use crate::ssim::Ssim;
use brush_dataset::scene_loader::SceneLoader;
use brush_render::{MainBackend, SplatOps, gaussian_splats::Splats};
use burn::{
    backend::wgpu::WgpuDevice,
    prelude::Backend,
    tensor::{Int, Tensor, TensorPrimitive, s},
};
use glam::Vec3;
use tracing::{Instrument, trace_span};

#[derive(Debug, Clone)]
pub struct GaussianScores<B: Backend> {
    pub importance_score: Tensor<B, 1>,
    pub pruning_score: Tensor<B, 1>,
}

fn compute_losses(
    pred: Tensor<MainBackend, 3>,
    gt: Tensor<MainBackend, 3>,
    ssim: &Ssim<MainBackend>,
) -> (Tensor<MainBackend, 3>, Tensor<MainBackend, 3>) {
    let l1_loss = (pred.clone() - gt.clone()).abs();

    let ssim_loss = 1.0 - ssim.ssim(pred, gt);
    let photometric_loss: Tensor<MainBackend, 3> = l1_loss.clone() * 0.8 + ssim_loss * 0.2;

    let l1_mean = l1_loss.mean_dim(2);

    let l1_min = l1_mean.clone().min().reshape([1, 1, 1]);
    let l1_max = l1_mean.clone().max().reshape([1, 1, 1]);

    let l1_norm = (l1_mean - l1_min.clone()) / (l1_max - l1_min).clamp_min(1e-9);

    (l1_norm.detach(), photometric_loss.detach())
}

async fn compute_view_metrics(
    batch_img_tensor: burn::tensor::TensorData,
    camera: &brush_render::camera::Camera,
    splats: Splats<MainBackend>,
    ssim: &Ssim<MainBackend>,
    device: &WgpuDevice,
    error_threshold: f32,
) -> (Tensor<MainBackend, 1>, Tensor<MainBackend, 1>) {
    let [img_h, img_w, _] = batch_img_tensor.shape.clone().try_into().unwrap();
    let gt_tensor = Tensor::<MainBackend, 3>::from_data(batch_img_tensor, device);
    let background = Vec3::ZERO;

    let project_output = MainBackend::project(
        camera,
        glam::uvec2(img_w as u32, img_h as u32),
        splats.means.val().into_primitive().tensor(),
        splats.log_scales.val().into_primitive().tensor(),
        splats.rotations.val().into_primitive().tensor(),
        splats.sh_coeffs.val().into_primitive().tensor(),
        splats.raw_opacities.val().into_primitive().tensor(),
        splats.render_mode,
    );

    let num_intersections = project_output.read_num_intersections().await;

    let (out_img, _, _) = MainBackend::rasterize(
        &project_output,
        num_intersections,
        background,
        true,
        false,
        None,
    );

    // Compute high error mask
    let pred_image = Tensor::from_primitive(TensorPrimitive::Float(out_img));
    let pred_rgb = pred_image.slice(s![.., .., 0..3]);
    let gt_rgb = gt_tensor.slice(s![.., .., 0..3]);

    let (l1_rgb_norm, photometric_loss) = compute_losses(pred_rgb, gt_rgb, ssim);

    let high_error_mask = l1_rgb_norm.greater_elem(error_threshold).float();
    let high_error_mask_primitive = high_error_mask.detach().into_primitive().tensor();

    // Compute high error count
    let (_, aux, _) = MainBackend::rasterize(
        &project_output,
        num_intersections,
        background,
        true,
        true,
        Some(&high_error_mask_primitive),
    );

    let high_error_count = Tensor::<MainBackend, 1, Int>::from_primitive(aux.high_error_count)
        .float()
        .detach();

    let loss_mean = photometric_loss.mean().detach();

    (high_error_count, loss_mean)
}

pub async fn compute_gaussian_score(
    dataloader: &mut SceneLoader,
    splats: Splats<MainBackend>,
    n_views: i32,
    error_threshold: f32,
) -> GaussianScores<MainBackend> {
    let device = splats.device();

    const SSIM_WINDOW_SIZE: usize = 11;
    let ssim = Ssim::new(SSIM_WINDOW_SIZE, 3, &device);

    let num_splats = splats.means.val().shape().dims[0];
    let mut accum_high_error_count = Tensor::<MainBackend, 1>::zeros([num_splats], &device);
    let mut accum_high_error_metric = Tensor::<MainBackend, 1>::zeros([num_splats], &device);

    for _ in 0..n_views {
        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("Wait for next data batch"))
            .await;

        let (high_error_count, photometric_loss) = compute_view_metrics(
            batch.img_tensor,
            &batch.camera,
            splats.clone(),
            &ssim,
            &device,
            error_threshold,
        )
        .await;

        accum_high_error_count = accum_high_error_count.add(high_error_count.clone());

        accum_high_error_metric =
            accum_high_error_metric.add(high_error_count.mul(photometric_loss));
    }

    let importance_score = accum_high_error_count / (n_views as f32);

    let min = accum_high_error_metric.clone().min();
    let max = accum_high_error_metric.clone().max();
    let pruning_score = (accum_high_error_metric - min.clone()) / (max - min).clamp_min(1e-9);

    GaussianScores {
        importance_score,
        pruning_score,
    }
}
