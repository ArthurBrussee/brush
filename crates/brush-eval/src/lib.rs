use anyhow::Result;
use brush_dataset::scene::{SceneView, sample_to_tensor, view_to_sample_image};
use brush_render::gaussian_splats::Splats;
use brush_render::{RenderAux, SplatForward};
use brush_ssim::Ssim;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use image::DynamicImage;

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

    // Compare MSE in RGB only, not sure if this should include alpha.
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let gt_tensor = sample_to_tensor(
        &view_to_sample_image(gt_img.clone(), eval_view.image.is_masked()),
        device,
    );

    let gt_rgb = gt_tensor.slice([0..res.y as usize, 0..res.x as usize, 0..3]);

    let (rendered, aux) = splats.render(&eval_view.camera, res, true);
    let render_rgb = rendered.slice([0..res.y as usize, 0..res.x as usize, 0..3]);

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (render_rgb * 255.0).round() / 255.0;

    let mse = (render_rgb.clone() - gt_rgb.clone())
        .powf_scalar(2.0)
        .mean();

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
