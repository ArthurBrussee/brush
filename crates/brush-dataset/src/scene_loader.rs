use std::sync::Arc;

use brush_async::Actor;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::scene::{Scene, SceneBatch, sample_to_packed_data, view_to_sample_image};

/// Hard cap for the packed-batch cache budget.
const CACHE_BUDGET_CAP: usize = 6 * 1024 * 1024 * 1024;

/// Default cache budget when the user doesn't override it.
/// Native: `min(6 GiB, 1/4 of system RAM)` — on a 16 GiB machine that's ~4 GiB,
/// deliberately leaving the rest for the GPU working set (which shares the same
/// unified memory on Apple Silicon). Floored at 512 MiB so tiny machines still
/// cache something. Wasm: a flat 2 GiB, since the browser heap is already bounded.
#[cfg(not(target_family = "wasm"))]
fn default_cache_budget() -> usize {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let quarter = (sys.total_memory() / 4) as usize; // total_memory() is bytes
    quarter.clamp(512 * 1024 * 1024, CACHE_BUDGET_CAP)
}
#[cfg(target_family = "wasm")]
fn default_cache_budget() -> usize {
    2 * 1024 * 1024 * 1024
}

fn resolve_cache_budget(override_bytes: Option<usize>) -> usize {
    let budget = override_bytes.unwrap_or_else(default_cache_budget);
    log::info!(
        "Scene batch cache budget: {} MiB{}",
        budget / (1024 * 1024),
        if override_bytes.is_some() {
            " (user override)"
        } else {
            " (auto: min(6 GiB, 1/4 RAM))"
        }
    );
    budget
}

/// Shared cache of GPU-ready scene batches, one slot per view. Caching the
/// packed batch (not the decoded `DynamicImage`) makes a hit a single copy of
/// the already-packed `[H, W]` u32 buffer. When inserting would exceed the
/// budget, the **least-recently-used** cached views are evicted to make room —
/// so a budget smaller than the dataset keeps the *hot* views resident instead
/// of refusing new entries and re-decoding forever.
struct BatchCache {
    slots: Vec<Option<Arc<SceneBatch>>>,
    sizes: Vec<usize>,   // bytes held per slot (0 when empty)
    last_used: Vec<u64>, // logical-clock tick of last access per slot
    used_bytes: usize,
    budget_bytes: usize,
    tick: u64,
}

impl BatchCache {
    fn new(n_views: usize, budget_bytes: usize) -> Self {
        Self {
            slots: vec![None; n_views],
            sizes: vec![0; n_views],
            last_used: vec![0; n_views],
            used_bytes: 0,
            budget_bytes,
            tick: 0,
        }
    }

    fn get(&mut self, index: usize) -> Option<Arc<SceneBatch>> {
        let hit = self.slots[index].clone();
        if hit.is_some() {
            self.tick += 1;
            self.last_used[index] = self.tick;
        }
        hit
    }

    /// Evict the least-recently-used occupied slot (never `except`).
    /// Returns false if there's nothing else to evict.
    fn evict_lru(&mut self, except: usize) -> bool {
        let victim = (0..self.slots.len())
            .filter(|&i| i != except && self.slots[i].is_some())
            .min_by_key(|&i| self.last_used[i]);
        if let Some(i) = victim {
            self.slots[i] = None;
            self.used_bytes -= self.sizes[i];
            self.sizes[i] = 0;
            true
        } else {
            false
        }
    }

    fn insert(&mut self, index: usize, batch: Arc<SceneBatch>) {
        if self.slots[index].is_some() {
            return;
        }
        // Exact byte accounting (rounding to whole MB let sub-MB images slip in
        // free and bypass the budget).
        let size_bytes = batch.img_packed.as_bytes().len();
        // A single batch bigger than the whole budget is never cached.
        if size_bytes > self.budget_bytes {
            return;
        }
        // Evict cold slots until this batch fits.
        while self.used_bytes + size_bytes > self.budget_bytes && self.evict_lru(index) {}
        if self.used_bytes + size_bytes <= self.budget_bytes {
            self.tick += 1;
            self.last_used[index] = self.tick;
            self.sizes[index] = size_bytes;
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
    pub fn new(scene: &Scene, seed: u64, max_cache_bytes: Option<usize>) -> Self {
        // Prefetch buffer: at most 4 batches ahead of the trainer.
        // Two tasks per actor share this buffer so one task's I/O can
        // overlap with the other's decode + GPU upload.
        let (tx, rx) = mpsc::channel(4);

        // Fan out only as many loaders as we have real parallelism, but cap it:
        // each in-flight task can hold a full decoded `DynamicImage`, so peak
        // decode memory scales with the task count. With only a 4-deep prefetch
        // buffer (above), more than a handful of producers can't stay useful —
        // they just raise peak memory and thread contention. Capping at
        // MAX_LOADER_ACTORS keeps decode throughput while bounding that peak
        // (e.g. 10-core: 10→6 actors ⇒ 20→12 max concurrent decodes).
        // Wasm shares one JS event loop, so a single actor avoids contention.
        const MAX_LOADER_ACTORS: usize = 6;
        let n_actors = if cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism()
                .map_or(8, |p| p.get())
                .min(MAX_LOADER_ACTORS)
        };
        const TASKS_PER_ACTOR: usize = 2;

        let views = scene.views.clone();
        let cache = Arc::new(Mutex::new(BatchCache::new(
            views.len(),
            resolve_cache_budget(max_cache_bytes),
        )));

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
            let batch = Arc::new(SceneBatch {
                img_packed,
                has_alpha,
                alpha_mode: view.image.alpha_mode(),
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
