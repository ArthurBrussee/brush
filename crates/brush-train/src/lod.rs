use brush_dataset::scene::{sample_to_tensor_data, view_to_sample_image};
use brush_render::{MainBackend, gaussian_splats::Splats};
use brush_render_bwd::render_splats;
use burn::{
    backend::Autodiff,
    prelude::Module,
    tensor::{Tensor, TensorData, TensorPrimitive, s},
};
use burn_cubecl::cubecl::wgpu::WgpuDevice;
use glam::Vec3;

type DiffBackend = Autodiff<MainBackend>;

/// Decimate splats to `target_count` using pre-computed per-Gaussian scores.
/// Higher scores are considered more important and kept.
pub async fn decimate_to_count(
    mut splats: Splats<MainBackend>,
    scores: &[f32],
    target_count: u32,
) -> Splats<MainBackend> {
    let num = splats.num_splats();
    if target_count >= num {
        return splats;
    }

    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let keep_indices: Vec<i32> = indexed[..target_count as usize]
        .iter()
        .map(|(i, _)| *i as i32)
        .collect();

    let device = splats.device();
    let keep_tensor = Tensor::from_data(
        TensorData::new(keep_indices, [target_count as usize]),
        &device,
    );

    splats.transforms = splats.transforms.map(|t| t.select(0, keep_tensor.clone()));
    splats.sh_coeffs = splats.sh_coeffs.map(|c| c.select(0, keep_tensor.clone()));
    splats.raw_opacities = splats
        .raw_opacities
        .map(|o| o.select(0, keep_tensor.clone()));

    splats
}

/// Log-determinant of a 6x6 positive semi-definite matrix via Cholesky decomposition.
/// Returns `f32::NEG_INFINITY` if the matrix is not positive definite.
fn log_det_6x6(m: &[f32; 36]) -> f32 {
    let mut l = [0.0f32; 36];
    for j in 0..6 {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * 6 + k] * l[j * 6 + k];
        }
        let diag = m[j * 6 + j] - sum;
        if diag <= 0.0 {
            return f32::NEG_INFINITY;
        }
        l[j * 6 + j] = diag.sqrt();
        for i in (j + 1)..6 {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * 6 + k] * l[j * 6 + k];
            }
            l[i * 6 + j] = (m[i * 6 + j] - sum) / l[j * 6 + j];
        }
    }
    let mut log_det = 0.0f32;
    for i in 0..6 {
        log_det += l[i * 6 + i].ln();
    }
    2.0 * log_det
}

/// Compute sensitivity-based pruning scores for all Gaussians.
///
/// Inspired by PUP 3D-GS (Hanson et al., CVPR 2025): <https://pup3dgs.github.io/>
///
/// Runs a single forward+backward pass over every training view, accumulating
/// the per-Gaussian Hessian approximation `H_i = sum(J_i * J_i^T)` where `J_i` is
/// the 6-element gradient vector `[d_mean, d_log_scale]`. The score is `log|det(H_i)|`.
pub async fn compute_pup_scores(
    splats: Splats<MainBackend>,
    scene: &brush_dataset::scene::Scene,
    device: &WgpuDevice,
) -> Vec<f32> {
    let num_splats = splats.num_splats() as usize;
    let mut hessian_accum: Tensor<MainBackend, 3> = Tensor::zeros([num_splats, 6, 6], device);

    for (vi, view) in scene.views.iter().enumerate() {
        log::info!("PUP scoring: view {}/{}", vi + 1, scene.views.len());

        let image = view
            .image
            .load()
            .await
            .expect("Failed to load image for PUP scoring");
        let sample = view_to_sample_image(image, view.image.alpha_mode());
        let img_size = glam::uvec2(sample.width(), sample.height());
        let gt_data = sample_to_tensor_data(sample);

        let mut splats: Splats<DiffBackend> = splats.clone().train();
        splats.transforms = splats
            .transforms
            .map(|t: Tensor<DiffBackend, 2>| t.require_grad());

        let diff_out = render_splats(splats.clone(), &view.camera, img_size, Vec3::ZERO).await;
        let pred_image = Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));

        let gt_tensor: Tensor<DiffBackend, 3> = Tensor::from_data(gt_data, device);
        let channels = pred_image.dims()[2].min(gt_tensor.dims()[2]);
        let pred_rgb = pred_image.slice(s![.., .., 0..channels]);
        let gt_rgb = gt_tensor.slice(s![.., .., 0..channels]);

        let loss = (pred_rgb - gt_rgb).abs().mean();
        let mut grads = loss.backward();

        let transforms_grad = splats
            .transforms
            .val()
            .grad_remove(&mut grads)
            .expect("Transform gradients required for PUP scoring");
        // Extract means (cols 0..3) and log_scales (cols 7..10) gradients for 6D Hessian
        let mean_grad = transforms_grad.clone().slice(s![.., 0..3]);
        let scale_grad = transforms_grad.slice(s![.., 7..10]);

        let j = Tensor::cat(vec![mean_grad, scale_grad], 1);
        let j_col = j.clone().unsqueeze_dim(2);
        let j_row = j.unsqueeze_dim(1);
        let outer = j_col.mul(j_row);
        hessian_accum = hessian_accum + outer;
    }

    let hessian_data = hessian_accum
        .into_data_async()
        .await
        .expect("Failed to read Hessian accumulator")
        .into_vec()
        .expect("Failed to convert Hessian data");

    hessian_data
        .as_chunks::<36>()
        .0
        .iter()
        .map(log_det_6x6)
        .collect()
}
