use std::sync::Arc;

use brush_async::Actor;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::scene::{DepthSample, Scene, SceneBatch, sample_to_packed_data, view_to_sample_image};

/// Cache budget for prepared views. 6 GB on native; less on wasm since the
/// whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const CACHE_BUDGET_BYTES: usize = 6 * 1024 * 1024 * 1024;
#[cfg(target_family = "wasm")]
const CACHE_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// One fully-prepared view: packed GT image plus (when present) the `LiDAR`
/// depth sample, ready to clone into a [`SceneBatch`].
struct CachedView {
    img_packed: burn::tensor::TensorData,
    has_alpha: bool,
    depth: Option<DepthSample>,
}

impl CachedView {
    fn size_bytes(&self) -> usize {
        let depth_bytes = self
            .depth
            .as_ref()
            .map_or(0, |d| d.depth.as_bytes().len() + d.conf.as_bytes().len());
        self.img_packed.as_bytes().len() + depth_bytes
    }
}

/// Shared prepared-view cache. Each slot holds at most one view; once the
/// running total passes `budget_bytes`, new views bypass the cache and just
/// get re-prepared on every visit.
struct ViewCache {
    slots: Vec<Option<Arc<CachedView>>>,
    used_bytes: usize,
    budget_bytes: usize,
}

impl ViewCache {
    fn new(n_views: usize) -> Self {
        Self {
            slots: vec![None; n_views],
            used_bytes: 0,
            budget_bytes: CACHE_BUDGET_BYTES,
        }
    }

    fn get(&self, index: usize) -> Option<Arc<CachedView>> {
        self.slots[index].clone()
    }

    fn insert(&mut self, index: usize, view: Arc<CachedView>) {
        if self.slots[index].is_some() {
            return;
        }
        let size_bytes = view.size_bytes();
        if self.used_bytes + size_bytes < self.budget_bytes {
            self.slots[index] = Some(view);
            self.used_bytes += size_bytes;
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
        // Prefetch buffer: at most 4 batches ahead of the trainer.
        // Two tasks per actor share this buffer so one task's I/O can
        // overlap with the other's decode + GPU upload.
        let (tx, rx) = mpsc::channel(4);

        // Fan out only as many loaders as we have real parallelism.
        // Wasm shares one JS event loop, so extra actors just add
        // contention without overlapping I/O.
        let n_actors = if cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism().map_or(8, |p| p.get())
        };
        const TASKS_PER_ACTOR: usize = 2;

        let views = scene.views.clone();
        let cache = Arc::new(Mutex::new(ViewCache::new(views.len())));

        let mut task_idx: u64 = 0;
        let actors: Vec<Actor> = (0..n_actors)
            .map(|i| {
                let actor = Actor::new(&format!("dataloader-{i}"));
                for _ in 0..TASKS_PER_ACTOR {
                    let views = views.clone();
                    let cache = cache.clone();
                    let tx = tx.clone();
                    let task_seed = seed.wrapping_add(task_idx);
                    task_idx += 1;
                    actor
                        .run(move || run_loader(views, cache, tx, task_seed))
                        .detach();
                }
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

/// Load (or fetch from cache) a view's packed image + depth sample.
async fn load_cached(
    views: &Arc<Vec<crate::scene::SceneView>>,
    cache: &Arc<Mutex<ViewCache>>,
    index: usize,
) -> Arc<CachedView> {
    if let Some(cached) = cache.lock().await.get(index) {
        return cached;
    }
    let view = &views[index];
    let raw = view
        .image
        .load()
        .await
        .expect("Scene loader failed to load an image");
    let sample = view_to_sample_image(raw, view.image.alpha_mode());
    let (img_packed, has_alpha) = sample_to_packed_data(sample);
    let depth = match &view.depth {
        Some(d) => d.load().await.ok().map(|dd| DepthSample {
            depth: burn::tensor::TensorData::new(dd.depth, [dd.height, dd.width, 1]),
            // ARKit confidence 0/1/2 -> [0, 1], used as a per-pixel weight
            // in the depth loss (no hard threshold).
            conf: burn::tensor::TensorData::new(
                dd.conf.iter().map(|&c| c as f32 / 2.0).collect::<Vec<_>>(),
                [dd.height, dd.width, 1],
            ),
        }),
        None => None,
    };
    let cached = Arc::new(CachedView {
        img_packed,
        has_alpha,
        depth,
    });
    cache.lock().await.insert(index, cached.clone());
    cached
}

async fn run_loader(
    views: Arc<Vec<crate::scene::SceneView>>,
    cache: Arc<Mutex<ViewCache>>,
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

        let cached = load_cached(&views, &cache, index).await;
        let batch = SceneBatch {
            img_packed: cached.img_packed.clone(),
            has_alpha: cached.has_alpha,
            alpha_mode: view.image.alpha_mode(),
            camera: view.camera,
            depth: cached.depth.clone(),
        };

        if tx.send(batch).await.is_err() {
            break;
        }
        brush_async::yield_now().await;
    }
}
