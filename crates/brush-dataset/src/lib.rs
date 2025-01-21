pub mod brush_vfs;
mod formats;
pub mod scene_loader;
pub mod splat_export;
pub mod splat_import;

use burn::config::Config;
pub use formats::load_dataset;

use async_fn_stream::fn_stream;
use brush_train::scene::{Scene, SceneView};
use std::future::Future;

use clap::Args;
use glam::{Mat3, Mat4, Vec3};
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

fn solve_cubic(a: f32, b: f32, c: f32, d: f32) -> Vec<f32> {
    // Convert to depressed cubic t^3 + pt + q = 0
    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    // For symmetric matrices, we know D <= 0 (three real roots)
    let phi = (-q / (2.0 * f32::sqrt(-(p * p * p) / 27.0))).acos();
    let t1 = 2.0 * f32::sqrt(-p / 3.0) * f32::cos(phi / 3.0);
    let t2 = 2.0 * f32::sqrt(-p / 3.0) * f32::cos((phi + 2.0 * std::f32::consts::PI) / 3.0);
    let t3 = 2.0 * f32::sqrt(-p / 3.0) * f32::cos((phi + 4.0 * std::f32::consts::PI) / 3.0);
    // Convert back to original cubic
    let mut roots = vec![t1 - b / (3.0 * a), t2 - b / (3.0 * a), t3 - b / (3.0 * a)];
    roots.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Less)); // sort in descending order
    roots
}

#[allow(clippy::needless_range_loop)]
fn find_eigenvector(matrix: Mat3, eigenvalue: f32) -> Vec3 {
    // Create matrix (A - λI)
    let m = Mat3::from_cols(
        matrix.col(0) - Vec3::new(eigenvalue, 0.0, 0.0),
        matrix.col(1) - Vec3::new(0.0, eigenvalue, 0.0),
        matrix.col(2) - Vec3::new(0.0, 0.0, eigenvalue),
    );
    // Convert matrix to array for easier manipulation
    let mut m_arr = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m_arr[i][j] = m.col(j)[i];
        }
    }
    // Gaussian elimination with pivoting
    for i in 0..2 {
        // Find pivot
        let mut max_element = m_arr[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..3 {
            if m_arr[k][i].abs() > max_element {
                max_element = m_arr[k][i].abs();
                max_row = k;
            }
        }
        // Swap maximum row with current row
        if max_row != i {
            for j in 0..3 {
                let temp = m_arr[i][j];
                m_arr[i][j] = m_arr[max_row][j];
                m_arr[max_row][j] = temp;
            }
        }
        // Make all rows below this one 0 in current column
        for k in (i + 1)..3 {
            let c = -m_arr[k][i] / m_arr[i][i];
            for j in i..3 {
                if i == j {
                    m_arr[k][j] = 0.0;
                } else {
                    m_arr[k][j] += c * m_arr[i][j];
                }
            }
        }
    }
    // Back substitution
    let mut x = Vec3::new(0.0, 0.0, 1.0); // Set z = 1 as we have infinite solutions
    if m_arr[1][1].abs() > 1e-10 {
        x.y = -m_arr[1][2] / m_arr[1][1];
    }
    if m_arr[0][0].abs() > 1e-10 {
        x.x = -(m_arr[0][1] * x.y + m_arr[0][2] * x.z) / m_arr[0][0];
    }
    // Normalize eigenvector
    x.normalize()
}

pub fn compute_sorted_eigenvectors(matrix: Mat3) -> Vec<Vec3> {
    // Calculate coefficients of characteristic polynomial
    // det(A - λI) = -λ^3 + c2λ^2 + c1λ + c0
    let a = -1.0;
    let b = matrix.col(0).x + matrix.col(1).y + matrix.col(2).z;
    let c = matrix.col(1).z * matrix.col(2).y
        + matrix.col(0).z * matrix.col(2).x
        + matrix.col(0).y * matrix.col(1).x
        - matrix.col(0).x * matrix.col(1).y
        - matrix.col(1).y * matrix.col(2).z
        - matrix.col(0).x * matrix.col(2).z;
    let d = matrix.col(0).x * matrix.col(1).y * matrix.col(2).z
        + matrix.col(0).y * matrix.col(1).z * matrix.col(2).x
        + matrix.col(0).z * matrix.col(1).x * matrix.col(2).y
        - matrix.col(0).x * matrix.col(1).z * matrix.col(2).y
        - matrix.col(0).y * matrix.col(1).x * matrix.col(2).z
        - matrix.col(0).z * matrix.col(1).y * matrix.col(2).x;
    // Find eigenvalues
    let eigenvalues = solve_cubic(a, b, c, d);
    // Find eigenvectors
    let eigenvectors = eigenvalues
        .iter()
        .map(|&ev| find_eigenvector(matrix, ev))
        .collect();
    eigenvectors
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

    #[allow(clippy::needless_range_loop)]
    pub fn estimate_up(&self) -> Vec3 {
        // based on https://github.com/jonbarron/camp_zipnerf/blob/8e6d57e3aee34235faf3ef99decca0994efe66c9/camp_zipnerf/internal/camera_utils.py#L233
        let mut c2ws = vec![];
        let mut ts = vec![];
        for view in self.train.views.iter() {
            c2ws.push(view.camera.local_to_world());
            ts.push(view.camera.position);
        }
        if let Some(eval_scene) = &self.eval {
            for view in eval_scene.views.iter() {
                c2ws.push(view.camera.local_to_world());
                ts.push(view.camera.position);
            }
        }
        let mean_t = ts.iter().fold(Vec3::ZERO, |acc, x| acc + *x) / ts.len() as f32;
        for t in &mut ts {
            *t -= mean_t;
        }
        // Compute 3x3 covariance by t^T * t ((3, N) * (N, 3) -> (3, 3))
        let mut t_cov = Mat3::ZERO;
        for row in 0..3 {
            let mut cov_row = Vec3::ZERO;
            for n in 0..ts.len() {
                cov_row += ts[n][row] * ts[n];
            }
            *t_cov.col_mut(row) = cov_row;
        }
        t_cov = t_cov.transpose();
        let eigvecs = compute_sorted_eigenvectors(t_cov);
        let mut rot = Mat3::from_cols(eigvecs[0], eigvecs[1], eigvecs[2]).transpose();
        if rot.determinant() < 0.0 {
            let diag = Mat3::from_diagonal(Vec3::new(1.0, 1.0, -1.0));
            rot = diag.mul_mat3(&rot);
        }
        let mut transform = Mat4::from_cols(
            rot.col(0).extend(0.0),
            rot.col(1).extend(0.0),
            rot.col(2).extend(0.0),
            rot.mul_vec3(-mean_t).extend(1.0),
        );
        let mut y_axis_z = 0.0;
        for c2w in c2ws {
            y_axis_z += transform.mul_mat4(&Mat4::from(c2w)).col(1).z;
        }
        // Flip coordinate system if z component of y-axis is negative
        if y_axis_z < 0.0 {
            let scale = Mat4::from_scale(Vec3::new(1.0, -1.0, -1.0));
            transform = scale.mul_mat4(&transform);
        }
        Vec3::new(transform.col(0).z, transform.col(1).z, -transform.col(2).z)
    }
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
