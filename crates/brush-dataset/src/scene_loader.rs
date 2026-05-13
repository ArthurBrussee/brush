use std::sync::Arc;

use brush_async::{Actor, task};
use image::DynamicImage;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::scene::{Scene, SceneBatch, sample_to_packed_data, view_to_sample_image};

/// Cache budget for decoded source images. 6 GB on native; less on
/// wasm since the whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const CACHE_BUDGET_MB: usize = 6 * 1024;
#[cfg(target_family = "wasm")]
const CACHE_BUDGET_MB: usize = 2 * 1024;

/// Shared decoded-image cache. Each slot holds at most one image; once
/// the running total passes `budget_mb`, new images bypass the cache
/// and just get re-decoded on every visit.
struct ImageCache {
    slots: Vec<Option<Arc<DynamicImage>>>,
    used_mb: usize,
    budget_mb: usize,
}

impl ImageCache {
    fn new(n_views: usize) -> Self {
        Self {
            slots: vec![None; n_views],
            used_mb: 0,
            budget_mb: CACHE_BUDGET_MB,
        }
    }

    fn get(&self, index: usize) -> Option<Arc<DynamicImage>> {
        self.slots[index].clone()
    }

    fn insert(&mut self, index: usize, image: Arc<DynamicImage>) {
        if self.slots[index].is_some() {
            return;
        }
        let size_mb = image.as_bytes().len() / (1024 * 1024);
        if self.used_mb + size_mb < self.budget_mb {
            self.slots[index] = Some(image);
            self.used_mb += size_mb;
        }
    }
}

pub struct SceneLoader {
    rx: mpsc::Receiver<SceneBatch>,
    // Owns the loader actor threads. Dropping cancels them; their
    // senders then drop, the channel closes, and `next_batch` returns.
    _actors: Vec<Actor>,
}

impl SceneLoader {
    pub fn new(scene: &Scene, seed: u64) -> Self {
        // Prefetch buffer: at most 2 batches ahead of the trainer.
        let (tx, rx) = mpsc::channel(2);

        // Fan out only as many loaders as we have real parallelism.
        // Wasm shares one JS event loop, so extra actors just add
        // contention without overlapping I/O.
        let n_actors = if cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism().map_or(8, |p| p.get())
        };

        let views = scene.views.clone();
        let cache = Arc::new(Mutex::new(ImageCache::new(views.len())));

        let actors: Vec<Actor> = (0..n_actors)
            .map(|i| {
                let actor = Actor::new(&format!("dataloader-{i}"));
                let views = views.clone();
                let cache = cache.clone();
                let tx = tx.clone();
                let task_seed = seed.wrapping_add(i as u64);
                actor.spawn(move || run_loader(views, cache, tx, task_seed));
                actor
            })
            .collect();

        Self {
            rx,
            _actors: actors,
        }
    }

    pub async fn next_batch(&mut self) -> SceneBatch {
        self.rx
            .recv()
            .await
            .expect("Scene loader channel closed unexpectedly")
    }
}

async fn run_loader(
    views: Arc<Vec<crate::scene::SceneView>>,
    cache: Arc<Mutex<ImageCache>>,
    tx: mpsc::Sender<SceneBatch>,
    seed: u64,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut shuffled: Vec<usize> = Vec::new();

    loop {
        if shuffled.is_empty() {
            shuffled = (0..views.len()).collect();
            shuffled.shuffle(&mut rng);
        }
        let index = shuffled.pop().expect("Need at least one view in dataset");
        let view = &views[index];

        let sample = if let Some(image) = cache.lock().await.get(index) {
            image
        } else {
            let raw = view
                .image
                .load()
                .await
                .expect("Scene loader failed to load an image");
            let sample = Arc::new(view_to_sample_image(raw, view.image.alpha_mode()));
            cache.lock().await.insert(index, sample.clone());
            sample
        };

        let (img_packed, has_alpha) = sample_to_packed_data(sample.as_ref().clone());
        let batch = SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: view.image.alpha_mode(),
            camera: view.camera.clone(),
        };

        if tx.send(batch).await.is_err() {
            break;
        }
        task::yield_now().await;
    }
}
