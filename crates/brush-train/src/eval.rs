#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::{sample_to_tensor_data, view_to_sample_image};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::render_aux::RenderAux;
use brush_render::{AlphaMode, SplatForward};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorPrimitive, s};
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

pub fn eval_stats<B: Backend + SplatForward<B>>(
    splats: &Splats<B>,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &B::Device,
) -> Result<EvalSample<B>> {
    // Compare MSE in RGB only.
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let gt_tensor = sample_to_tensor_data(view_to_sample_image(gt_img.clone(), alpha_mode));
    let gt_tensor = Tensor::from_data(gt_tensor, device);
    let gt_rgb = gt_tensor.slice(s![.., .., 0..3]);

    // Render on reference black background.
    let (img, aux) = {
        let (img, aux) = B::render_splats(
            gt_cam,
            res,
            splats.means.val().into_primitive().tensor(),
            splats.log_scales.val().into_primitive().tensor(),
            splats.rotations.val().into_primitive().tensor(),
            splats.sh_coeffs.val().into_primitive().tensor(),
            splats.raw_opacities.val().into_primitive().tensor(),
            splats.render_mode,
            Vec3::ZERO,
            true,
        );
        (Tensor::from_primitive(TensorPrimitive::Float(img)), aux)
    };
    let render_rgb = img.slice(s![.., .., 0..3]);

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

impl<B: Backend> EvalSample<B> {
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        use image::Rgb32FImage;
        log::info!("Saving eval image to disk.");
        let img = self.rendered.clone();
        let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = img.clone().into_data_async().await?.into_vec::<f32>()?;
        let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
            .expect("Failed to create image from tensor")
            .into();
        let img: image::DynamicImage = img.into_rgb8().into();
        let parent = path.parent().expect("Eval must have a filename");
        tokio::fs::create_dir_all(parent).await?;
        log::info!("Saving eval view to {path:?}");
        img.save(path)?;
        Ok(())
    }
}
