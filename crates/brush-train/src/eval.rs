use anyhow::Result;
use brush_dataset::scene::{SceneView, sample_to_tensor, view_to_sample_image};
use brush_render::SplatForward;
use brush_render::gaussian_splats::Splats;
use brush_render::render_aux::RenderAux;
use burn::prelude::Backend;
use burn::tensor::{Tensor, s};
use glam::Vec3;
use image::DynamicImage;

use crate::ssim::Ssim;

pub struct EvalSample<B: Backend> {
    pub gt_img: DynamicImage,
    pub rendered: Tensor<B, 3>,
    pub psnr: Tensor<B, 1>,
    pub ssim: Tensor<B, 1>,
    pub aux: RenderAux<B>,
}

pub async fn eval_stats<B: Backend + SplatForward<B>>(
    splats: Splats<B>,
    eval_view: &SceneView,
    device: &B::Device,
) -> Result<EvalSample<B>> {
    let gt_img = eval_view.image.load().await?;

    // Compare MSE in RGB only.
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let gt_tensor = sample_to_tensor(
        &view_to_sample_image(gt_img.clone(), eval_view.image.is_masked()),
        device,
    );

    let gt_rgb = gt_tensor.slice(s![.., .., 0..3]);

    // Render on reference black background.
    let (rendered, aux) = splats.render(&eval_view.camera, res, Vec3::ZERO, true);
    let render_rgb = rendered.slice(s![.., .., 0..3]);

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (render_rgb * 255.0).round() / 255.0;

    let mse = (render_rgb.clone() - gt_rgb.clone()).powi_scalar(2).mean();

    let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
    let ssim_measure = Ssim::new(11, 3, device);
    let ssim = ssim_measure.ssim(render_rgb.clone(), gt_rgb).mean();

    Ok(EvalSample {
        gt_img,
        psnr,
        ssim,
        rendered: render_rgb,
        aux,
    })
}
