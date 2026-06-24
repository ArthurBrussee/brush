use std::sync::Arc;

use brush_async::Actor;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::scene::{Scene, SceneBatch, sample_to_packed_data, view_to_sample_image};

/// Cache budget for packed scene batches. 6 GB on native; less on
/// wasm since the whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const CACHE_BUDGET_BYTES: usize = 6 * 1024 * 1024 * 1024;
#[cfg(target_family = "wasm")]
const CACHE_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Shared cache of GPU-ready scene batches. Each slot holds at most one
/// batch; once the running total passes `budget_bytes`, new batches bypass
/// the cache and just get re-decoded + re-packed on every visit.
///
/// Caching the packed batch (instead of the decoded `DynamicImage`) skips
/// the per-hit decode → premultiply → repack work: a cache hit is now a
/// single copy of the already-packed `[H, W]` u32 buffer.
struct BatchCache {
    slots: Vec<Option<Arc<SceneBatch>>>,
    used_bytes: usize,
    budget_bytes: usize,
}

impl BatchCache {
    fn new(n_views: usize) -> Self {
        Self {
            slots: vec![None; n_views],
            used_bytes: 0,
            budget_bytes: CACHE_BUDGET_BYTES,
        }
    }

    fn get(&self, index: usize) -> Option<Arc<SceneBatch>> {
        self.slots[index].clone()
    }

    fn insert(&mut self, index: usize, batch: Arc<SceneBatch>) {
        if self.slots[index].is_some() {
            return;
        }
        // Track exact bytes: rounding to whole MB let sub-MB images slip in
        // for free and bypass the budget entirely.
        let size_bytes = batch.img_packed.as_bytes().len();
        if self.used_bytes + size_bytes < self.budget_bytes {
            self.slots[index] = Some(batch);
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
        let cache = Arc::new(Mutex::new(BatchCache::new(views.len())));

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

async fn run_loader(
    views: Arc<Vec<crate::scene::SceneView>>,
    cache: Arc<Mutex<BatchCache>>,
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

        let batch = if let Some(batch) = cache.lock().await.get(index) {
            batch
        } else {
            let raw = view
                .image
                .load()
                .await
                .expect("Scene loader failed to load an image");
            let sample = view_to_sample_image(raw, view.image.alpha_mode());
            let (img_packed, has_alpha) = sample_to_packed_data(sample);

            let depth = if let Some(load_depth) = &view.depth {
                let [h, w] = [img_packed.shape[0], img_packed.shape[1]];
                Some(
                    load_depth
                        .load(h, w)
                        .await
                        .expect("Scene loader failed to load an image"),
                )
            } else {
                None
            };

            let batch = Arc::new(SceneBatch {
                img_packed,
                has_alpha,
                alpha_mode: view.image.alpha_mode(),
                depth,
                camera: view.camera,
            });
            cache.lock().await.insert(index, batch.clone());
            batch
        };

        // The channel takes an owned batch; clone the packed buffer out of
        // the shared cache entry.
        if tx.send(batch.as_ref().clone()).await.is_err() {
            break;
        }
        brush_async::yield_now().await;
    }
}
