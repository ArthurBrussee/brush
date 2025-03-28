use std::sync::Arc;

use burn::prelude::Backend;
use image::DynamicImage;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::mpsc::Receiver;
use tokio::sync::{RwLock, mpsc};
use tokio_with_wasm::alias as tokio_wasm;

use crate::scene::{Scene, SceneBatch, sample_to_tensor, view_to_sample_image};

pub struct SceneLoader<B: Backend> {
    receiver: Receiver<SceneBatch<B>>,
}

struct ImageCache {
    states: Vec<Option<Arc<DynamicImage>>>,
    max_size: usize,
    size: usize,
}

// Cache at most some nr. of gigs of data.
// TODO: Not sure if this should be configurable or not.
#[cfg(not(target_family = "wasm"))]
const MAX_CACHE_MB: usize = 6 * 1024;

// On WASM, not much hope a big dataset will work anyway but let's not
// cache more than what fits in memory.
#[cfg(target_family = "wasm")]
const MAX_CACHE_MB: usize = 2 * 1024;

impl ImageCache {
    fn new(max_size: usize, n_images: usize) -> Self {
        Self {
            states: vec![None; n_images],
            max_size,
            size: 0,
        }
    }

    fn try_get(&self, index: usize) -> Option<Arc<DynamicImage>> {
        self.states[index].clone()
    }

    fn insert(&mut self, index: usize, data: Arc<DynamicImage>) {
        let data_size_mb = data.as_bytes().len() / (1024 * 1024);

        if self.size + data_size_mb < self.max_size && self.states[index].is_none() {
            self.states[index] = Some(data);
            self.size += data_size_mb;
        }
    }
}

impl<B: Backend> SceneLoader<B> {
    pub fn new(scene: &Scene, seed: u64, device: &B::Device) -> Self {
        let num_img_queue = 32;

        // The bounded size == number of batches to prefetch.
        let (send_img, mut rec_imag) = mpsc::channel(num_img_queue);

        // On wasm, there is little point to spawning multiple of these. In theory there would be
        // IF file reading truly was async, but since the zip archive is just in memory it isn't really
        // any faster.
        let parallelism = if cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(8)
                // Don't need more threads than the image queue can hold, most
                // threads would just sit around idling!
                .min(num_img_queue) as u64
        };
        let num_views = scene.views.len();

        let load_cache = Arc::new(RwLock::new(ImageCache::new(MAX_CACHE_MB, num_views)));

        for i in 0..parallelism {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed + i);
            let send_img = send_img.clone();
            let views = scene.views.clone();

            let load_cache = load_cache.clone();

            tokio_wasm::spawn(async move {
                let mut shuf_indices = vec![];

                loop {
                    let index = shuf_indices.pop().unwrap_or_else(|| {
                        shuf_indices = (0..num_views).collect();
                        shuf_indices.shuffle(&mut rng);
                        shuf_indices
                            .pop()
                            .expect("Need at least one view in dataset")
                    });

                    let view = &views[index];

                    let sample = if let Some(image) = load_cache.read().await.try_get(index) {
                        image
                    } else {
                        let image = view
                            .image
                            .load()
                            .await
                            .expect("Scene loader encountered an error while loading an image");
                        let image = Arc::new(image);
                        // Don't premultiply the image if it's a mask - treat as fully opaque.
                        let sample = view_to_sample_image(image, view.image.is_masked());
                        load_cache.write().await.insert(index, sample.clone());
                        sample
                    };

                    if send_img
                        .send((sample, view.image.is_masked(), view.camera.clone()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
        let (send_batch, rec_batch) = mpsc::channel(2);

        let device = device.clone();
        tokio_wasm::spawn(async move {
            while let Some(rec) = rec_imag.recv().await {
                let (sample, alpha_is_mask, camera) = rec;
                let img_tensor = sample_to_tensor(&sample, &device);

                if send_batch
                    .send(SceneBatch {
                        img_tensor,
                        alpha_is_mask,
                        camera,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        Self {
            receiver: rec_batch,
        }
    }

    pub async fn next_batch(&mut self) -> SceneBatch<B> {
        self.receiver
            .recv()
            .await
            .expect("Somehow lost data loading channel!")
    }
}
