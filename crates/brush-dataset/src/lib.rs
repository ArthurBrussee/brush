pub mod brush_vfs;
mod formats;
pub mod scene_loader;
pub mod splat_export;
pub mod splat_import;

pub use formats::load_dataset;

use async_fn_stream::fn_stream;
use brush_train::scene::{Scene, SceneView};
use image::DynamicImage;
use std::future::Future;

use tokio_stream::Stream;
use tokio_with_wasm::alias as tokio_wasm;

#[derive(Clone, Default, Debug)]
pub struct LoadDatasetArgs {
    pub max_frames: Option<usize>,
    pub max_resolution: Option<u32>,
    pub eval_split_every: Option<usize>,
    pub subsample_frames: Option<u32>,
    pub subsample_points: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct LoadInitArgs {
    pub sh_degree: u32,
}

impl Default for LoadInitArgs {
    fn default() -> Self {
        Self { sh_degree: 3 }
    }
}

#[derive(Clone)]
pub struct Dataset {
    pub train: Scene,
    pub eval: Option<Scene>,
}

impl Dataset {
    pub fn empty() -> Self {
        Self {
            train: Scene::new(vec![]),
            eval: None,
        }
    }

    pub fn from_views(train_views: Vec<SceneView>, eval_views: Vec<SceneView>) -> Self {
        Self {
            train: Scene::new(train_views),
            eval: if eval_views.is_empty() {
                None
            } else {
                Some(Scene::new(eval_views))
            },
        }
    }
}

pub(crate) fn clamp_img_to_max_size(image: DynamicImage, max_size: u32) -> DynamicImage {
    if image.width() <= max_size && image.height() <= max_size {
        return image;
    }

    let aspect_ratio = image.width() as f32 / image.height() as f32;
    let (new_width, new_height) = if image.width() > image.height() {
        (max_size, (max_size as f32 / aspect_ratio) as u32)
    } else {
        ((max_size as f32 * aspect_ratio) as u32, max_size)
    };
    image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

pub(crate) fn stream_fut_parallel<T: Send + 'static>(
    futures: Vec<impl Future<Output = T> + Send + 'static>,
) -> impl Stream<Item = T> {
    let parallel = if cfg!(target_family = "wasm") {
        1
    } else {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(8)
    };

    log::info!("Loading stream with {parallel} threads");

    let mut futures = futures;
    fn_stream(|emitter| async move {
        while !futures.is_empty() {
            // Spawn a batch of threads.
            let handles: Vec<_> = futures
                .drain(..futures.len().min(parallel))
                .map(|fut| tokio_wasm::spawn(fut))
                .collect();
            // Stream each of them.
            for handle in handles {
                emitter
                    .emit(handle.await.expect("Underlying stream panicked"))
                    .await;
            }
        }
    })
}
