pub mod brush_vfs;
mod formats;
pub mod scene_loader;
pub mod splat_export;
pub mod splat_import;

use burn::config::Config;
pub use formats::load_dataset;

use async_fn_stream::fn_stream;
use brush_train::scene::{Scene, SceneView};
use image::DynamicImage;
use std::future::Future;

use clap::Args;
use tokio_stream::Stream;
use tokio_with_wasm::alias as tokio_wasm;

#[derive(Config, Debug, Args)]
pub struct LoadDataseConfig {
    /// Max nr. of frames of dataset to load
    #[arg(long, help_heading = "Dataset Options")]
    pub max_frames: Option<usize>,
    /// Max resolution of images to load
    #[arg(long, help_heading = "Dataset Options")]
    pub max_resolution: Option<u32>,
    /// Create an eval dataset by selecting every nth image
    #[arg(long, help_heading = "Dataset Options")]
    pub eval_split_every: Option<usize>,
    /// Load only every nth frame
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_frames: Option<u32>,
    /// Load only every nth point from the initial sfm data
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_points: Option<u32>,
}

#[derive(Config, Debug, Args)]
pub struct ModelConfig {
    #[arg(
        long,
        help = "SH degree of splats",
        help_heading = "Model Options",
        default_value = "3"
    )]
    #[config(default = 3)]
    pub sh_degree: u32,
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
    futures: Vec<impl Future<Output = T> + WasmNotSend + 'static>,
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

// On wasm, lots of things aren't Send that are send on non-wasm.
// Non-wasm tokio requires :Send for futures, tokio_with_wasm doesn't.
// So, it can help to annotate futures/objects as send only on not-wasm.
#[cfg(target_family = "wasm")]
mod wasm_send {
    pub trait WasmNotSend {}
    impl<T> WasmNotSend for T {}
}

#[cfg(not(target_family = "wasm"))]
mod wasm_send {
    pub trait WasmNotSend: Send {}
    impl<T: Send> WasmNotSend for T {}
}

pub use wasm_send::*;
