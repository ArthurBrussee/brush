use anyhow::Result;
use brush_dataset::scene::{SceneView, sample_to_tensor, view_to_sample_image};
use brush_render::SplatForward;
use brush_render::gaussian_splats::Splats;
use brush_render::render_aux::RenderAux;
use brush_ssim::Ssim;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use image::DynamicImage;
use std::path::Path;

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

impl<B: Backend> EvalSample<B> {
    pub async fn save_to_disk(&self, path: &Path) -> Result<()> {
        // TODO: FIgure out how to do this on WASM.
        #[cfg(not(target_family = "wasm"))]
        {
            use image::Rgb32FImage;
            log::info!("Saving eval image to disk.");

            let img = self.rendered.clone();
            let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
            let data = self
                .rendered
                .clone()
                .into_data_async()
                .await
                .into_vec::<f32>()
                .expect("Wrong type");

            let img: DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
                .expect("Failed to create image from tensor")
                .into();

            let parent = path.parent().expect("Eval must have a filename");
            tokio::fs::create_dir_all(parent).await?;
            log::info!("Saving eval view to {path:?}");
            img.save(path)?;
        }
        Ok(())
    }
}
