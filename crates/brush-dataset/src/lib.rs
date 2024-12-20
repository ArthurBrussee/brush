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

use glam::Vec3;
use ndarray::{arr2, concatenate, s, Array2, Array3, Axis};
use ndarray_linalg::{Determinant, Eig};
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

    pub fn estimate_up(&self) -> Vec3 {
        // based on https://github.com/jonbarron/camp_zipnerf/blob/8e6d57e3aee34235faf3ef99decca0994efe66c9/camp_zipnerf/internal/camera_utils.py#L233
        let mut c2ws_all = vec![];
        for view in self.train.views.iter() {
            let c2w = view.camera.local_to_world().transpose().to_cols_array();
            c2ws_all.push(c2w);
        }
        if let Some(eval_scene) = &self.eval {
            for view in eval_scene.views.iter() {
                let c2w = view.camera.local_to_world().transpose().to_cols_array();
                c2ws_all.push(c2w);
            }
        }
        let c2ws: Array3<f32> = arr2(&c2ws_all).into_shape((c2ws_all.len(), 4, 4)).unwrap().to_owned();
        let mut t: Array2<f32> = c2ws.slice(s![.., ..3, 3]).to_owned();
        let mean_t = t.mean_axis(Axis(0)).unwrap();
        t -= &mean_t;
        let t_cov = t.t().dot(&t);
        let (eigvals, eigvecs) = t_cov.eig().unwrap();
        let eigvals = eigvals.mapv(|x| x.re);
        let mut eigvecs = eigvecs.mapv(|x| x.re);
        // Sort eigenvectors in order of largest to smallest eigenvalue.
        let mut inds: Vec<usize> = (0..eigvals.len()).collect();
        inds.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());
        eigvecs.assign(&eigvecs.select(Axis(1), &inds));
        let mut rot = eigvecs.t().to_owned();
        if rot.det().unwrap() < 0.0 {
            let diag = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
            rot = diag.dot(&rot);
        }
        let mut transform = concatenate![
            Axis(1),
            rot.to_owned(),
            rot.dot(&-mean_t.insert_axis(Axis(1)))
        ];
        let mut y_axis_z = 0.0;
        for i in 0..c2ws.len_of(Axis(0)) {
            let c2w = c2ws.slice(s![i, .., ..]).to_owned();
            y_axis_z += transform.dot(&c2w)[(2, 1)];
        }
        // Flip coordinate system if z component of y-axis is negative
        if y_axis_z < 0.0 {
            let flip_diag = arr2(&[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]);
            transform = flip_diag.dot(&transform);
        }
        Vec3::new(transform[(2, 0)], transform[(2, 1)], -transform[(2, 2)])
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
