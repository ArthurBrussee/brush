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
pub async fn decimate_to_count_scored(
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

    splats.means = splats.means.map(|m| m.select(0, keep_tensor.clone()));
    splats.rotations = splats.rotations.map(|r| r.select(0, keep_tensor.clone()));
    splats.log_scales = splats.log_scales.map(|s| s.select(0, keep_tensor.clone()));
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
    let num_views = scene.views.len();

    let mut hessian_accum: Tensor<MainBackend, 3> = Tensor::zeros([num_splats, 6, 6], device);

    for (vi, view) in scene.views.iter().enumerate() {
        log::info!("PUP scoring: view {}/{}", vi + 1, num_views);

        let image = view
            .image
            .load()
            .await
            .expect("Failed to load image for PUP scoring");
        let sample = view_to_sample_image(image, view.image.alpha_mode());
        let img_size = glam::uvec2(sample.width(), sample.height());
        let gt_data = sample_to_tensor_data(sample);

        let mut splats_ad: Splats<DiffBackend> = splats.clone().train();
        splats_ad.means = splats_ad
            .means
            .map(|m: Tensor<DiffBackend, 2>| m.require_grad());
        splats_ad.log_scales = splats_ad
            .log_scales
            .map(|m: Tensor<DiffBackend, 2>| m.require_grad());

        let means_tensor = splats_ad.means.val();
        let scales_tensor = splats_ad.log_scales.val();

        let diff_out = render_splats(splats_ad, &view.camera, img_size, Vec3::ZERO).await;
        let pred_image = Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));

        let gt_tensor: Tensor<DiffBackend, 3> = Tensor::from_data(gt_data, device);
        let channels = pred_image.dims()[2].min(gt_tensor.dims()[2]);
        let pred_rgb = pred_image.slice(s![.., .., 0..channels]);
        let gt_rgb = gt_tensor.slice(s![.., .., 0..channels]);

        let loss = (pred_rgb - gt_rgb).abs().mean();
        let mut grads = loss.backward();

        let mean_grad = means_tensor
            .grad_remove(&mut grads)
            .expect("Mean gradients required for PUP scoring");
        let scale_grad = scales_tensor
            .grad_remove(&mut grads)
            .expect("Scale gradients required for PUP scoring");

        let j: Tensor<MainBackend, 2> = Tensor::cat(vec![mean_grad, scale_grad], 1);
        let j_col = j.clone().unsqueeze_dim(2);
        let j_row = j.unsqueeze_dim(1);
        let outer = j_col.mul(j_row);

        hessian_accum = hessian_accum + outer;
    }

    let hessian_data: Vec<f32> = hessian_accum
        .into_data_async()
        .await
        .expect("Failed to read Hessian accumulator")
        .into_vec()
        .expect("Failed to convert Hessian data");

    hessian_data
        .chunks_exact(36)
        .map(|chunk| log_det_6x6(chunk.try_into().unwrap()))
        .collect()
}
