use brush_train::image::view_to_sample;
use brush_train::scene::Scene;
use brush_train::train::SceneBatch;
use burn::prelude::Backend;
use rand::{seq::SliceRandom, SeedableRng};
use tokio::sync::mpsc;
use tokio::sync::mpsc::Receiver;
use tokio_with_wasm::alias as tokio_wasm;

pub struct SceneLoader<B: Backend> {
    receiver: Receiver<SceneBatch<B>>,
}

impl<B: Backend> SceneLoader<B> {
    pub fn new(scene: &Scene, seed: u64, device: &B::Device) -> Self {
        let scene = scene.clone();
        // The bounded size == number of batches to prefetch.
        let (tx, rx) = mpsc::channel(5);
        let device = device.clone();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let fut = async move {
            let mut shuf_indices = vec![];

            loop {
                let (gt_image, gt_view) = {
                    let index = shuf_indices.pop().unwrap_or_else(|| {
                        shuf_indices = (0..scene.views.len()).collect();
                        shuf_indices.shuffle(&mut rng);
                        shuf_indices
                            .pop()
                            .expect("Need at least one view in dataset")
                    });
                    let view = scene.views[index].clone();
                    (view_to_sample(&view, &device), view)
                };

                let scene_batch = SceneBatch { gt_image, gt_view };

                if tx.send(scene_batch).await.is_err() {
                    break;
                }
            }
        };

        tokio_wasm::spawn(fut);
        Self { receiver: rx }
    }

    pub async fn next_batch(&mut self) -> SceneBatch<B> {
        self.receiver
            .recv()
            .await
            .expect("Somehow lost data loading channel!")
    }
}
