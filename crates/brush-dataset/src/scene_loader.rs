use std::sync::Arc;

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::mpsc::Receiver;
use tokio::sync::{RwLock, mpsc};
use tokio_with_wasm::alias as tokio_wasm;

use crate::scene::{Scene, SceneBatch, view_to_sample_data};

pub struct SceneLoader<B: Backend> {
    receiver: Receiver<SceneBatch<B>>,
}

struct ImageCache {
    states: Vec<Option<TensorData>>,
    max_size: usize,
    active: usize,
}

const MAX_CACHE: usize = 512;

impl ImageCache {
    fn new(max_size: usize) -> Self {
        Self {
            states: vec![None; max_size],
            max_size,
            active: 0,
        }
    }

    fn try_get(&self, index: usize) -> Option<TensorData> {
        self.states[index].clone()
    }

    fn insert(&mut self, index: usize, data: &TensorData) {
        if self.active < self.max_size && self.states[index].is_none() {
            self.states[index] = Some(data.clone());
            self.active += 1;
        }
    }
}

impl<B: Backend> SceneLoader<B> {
    pub fn new(scene: &Scene, seed: u64, device: &B::Device) -> Self {
        // The bounded size == number of batches to prefetch.
        let (send_img, mut rec_imag) = mpsc::channel(64);

        // On wasm, there is little point to spawning multiple of these. In theory there would be
        // IF file reading truly was async, but since the zip archive is just in memory it isn't really
        // any faster.
        let parallelism = if cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(8) as u64
        };
        let num_views = scene.views.len();

        let load_cache = Arc::new(RwLock::new(ImageCache::new(MAX_CACHE)));

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

                    let sample_data = if let Some(data) = load_cache.read().await.try_get(index) {
                        data
                    } else {
                        let image = view
                            .image
                            .load()
                            .await
                            .expect("Scene loader encountered an error while loading an image");

                        // Don't premultiply the image if it's a mask - treat as fully opaque.
                        let sample = view_to_sample_data(&image, view.image.is_masked());
                        load_cache.write().await.insert(index, &sample);
                        sample
                    };

                    println!("Sending new image");

                    if send_img
                        .send((sample_data, view.image.is_masked(), view.camera.clone()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }

                println!("Shutting down data image loading");
            });
        }
        let (send_batch, rec_batch) = mpsc::channel(2);

        let device = device.clone();
        tokio_wasm::spawn(async move {
            while let Some(rec) = rec_imag.recv().await {
                let (sample, alpha_is_mask, camera) = rec;

                if send_batch
                    .send(SceneBatch {
                        img_tensor: Tensor::from_data(sample, &device),
                        alpha_is_mask,
                        camera,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }

            println!("Shutting down data tensor loading");
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
