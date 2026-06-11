#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::{sample_to_packed_data, view_to_sample_image};
use brush_loss::{ImageLossConfig, image_loss_eval};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, RenderAux, TextureMode, render_splats};
use burn::tensor::{Device, Int, Tensor, s};
use glam::Vec3;
use image::DynamicImage;

pub struct EvalSample {
    pub gt_img: DynamicImage,
    pub rendered: Tensor<3>,
    pub psnr: Tensor<1>,
    pub ssim: Tensor<1>,
    pub render_aux: RenderAux,
}

pub async fn eval_stats(
    splats: Splats,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &Device,
    correction: Option<&(dyn Fn(Tensor<3>) -> Tensor<3> + Sync)>,
) -> Result<EvalSample> {
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let (gt_packed_data, _has_alpha) =
        sample_to_packed_data(view_to_sample_image(gt_img.clone(), alpha_mode));
    let gt_packed: Tensor<2, Int> = Tensor::from_data(gt_packed_data, device);

    // Render on reference black background.
    let (img, render_aux) =
        render_splats(splats, gt_cam, res, Vec3::ZERO, None, TextureMode::Float).await;
    let render_rgb = img.slice(s![.., .., 0..3]);

    // Apply the learned per-view appearance correction when scoring a
    // training view (`--train-on-eval`): without it, scores on
    // appearance-varying datasets mostly measure the splat <-> average
    // appearance offset rather than reconstruction quality.
    let render_rgb = match correction {
        Some(f) => f(render_rgb),
        None => render_rgb,
    };

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (render_rgb * 255.0).round() / 255.0;

    let cfg = |l1, ssim| ImageLossConfig {
        l1_weight: l1,
        ssim_weight: ssim,
        composite_bg: None,
        mask: false,
    };
    // MSE = mean(L1^2) since |a - b|^2 == (a - b)^2.
    let mse = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(1.0, 0.0))
        .powi_scalar(2)
        .mean();
    let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
    let ssim = image_loss_eval(render_rgb.clone(), gt_packed, cfg(0.0, 1.0)).mean();

    Ok(EvalSample {
        gt_img,
        psnr,
        ssim,
        rendered: render_rgb,
        render_aux,
    })
}

impl EvalSample {
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
