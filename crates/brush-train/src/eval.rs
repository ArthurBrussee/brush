#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::{sample_to_tensor_data, view_to_sample_image};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, RenderAux, SplatOps};

#[cfg(target_family = "wasm")]
use brush_render::render::calc_tile_bounds;
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
    pub render_aux: RenderAux<B>,
}

pub fn eval_stats<B: Backend + SplatOps<B>>(
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

    // Render on reference black background using split pipeline.
    let (img, render_aux) = {
        // First pass: project
        let project_output = B::project(
            gt_cam,
            res,
            splats.means.val().into_primitive().tensor(),
            splats.log_scales.val().into_primitive().tensor(),
            splats.rotations.val().into_primitive().tensor(),
            splats.sh_coeffs.val().into_primitive().tensor(),
            splats.raw_opacities.val().into_primitive().tensor(),
            splats.render_mode,
        );

        // Sync readback of num_intersections
        #[cfg(not(target_family = "wasm"))]
        let num_intersections = project_output.num_intersections();

        #[cfg(target_family = "wasm")]
        let num_intersections = {
            use burn::tensor::ops::FloatTensorOps;
            let tile_bounds = calc_tile_bounds(res);
            let num_tiles = tile_bounds[0] * tile_bounds[1];
            let total_splats = splats.num_splats();
            let max_possible = num_tiles.saturating_mul(total_splats);
            max_possible.min(2 * 512 * 65535)
        };

        // Second pass: rasterize (with bwd_info = true for eval, drop compact_gid)
        let (out_img, render_aux, _) = B::rasterize(&project_output, num_intersections, Vec3::ZERO, true);

        (Tensor::from_primitive(TensorPrimitive::Float(out_img)), render_aux)
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
        render_aux,
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
