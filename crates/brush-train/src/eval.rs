use brush_render::gaussian_splats::Splats;
use brush_render::{RenderAux, SplatForward};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use rand::seq::IteratorRandom;

use crate::image::view_to_sample;
use crate::scene::{Scene, SceneView};
use crate::ssim::Ssim;

pub struct EvalSample<B: Backend> {
    pub index: usize,

    pub view: SceneView,
    pub rendered: Tensor<B, 3>,

    pub psnr: Tensor<B, 1>,
    pub ssim: Tensor<B, 1>,
    pub aux: RenderAux<B>,
}

pub fn eval_stats<B: Backend + SplatForward<B>>(
    splats: Splats<B>,
    eval_scene: &Scene,
    num_frames: Option<usize>,
    rng: &mut impl rand::Rng,
    device: &B::Device,
) -> impl Iterator<Item = EvalSample<B>> + 'static {
    let indices = if let Some(num) = num_frames {
        (0..eval_scene.views.len()).choose_multiple(rng, num)
    } else {
        (0..eval_scene.views.len()).collect()
    };

    let device = device.clone();
    let scene = eval_scene.clone();

    indices.into_iter().map(move |index| {
        let view = scene.views[index].clone();
        // Compare MSE in RGB only, not sure if this should include alpha.
        let res = glam::uvec2(view.image.width(), view.image.height());

        let gt_tensor = view_to_sample::<B>(&view, &device);
        let gt_rgb = gt_tensor.slice([0..res.y as usize, 0..res.x as usize, 0..3]);

        let (rendered, aux) = splats.render(&view.camera, res, false);

        let render_rgb = rendered.slice([0..res.y as usize, 0..res.x as usize, 0..3]);

        // Simulate 8-bit roundtrip for fair comparison.
        let render_rgb = (render_rgb * 255.0).round() / 255.0;

        let mse = (render_rgb.clone() - gt_rgb.clone())
            .powf_scalar(2.0)
            .mean();

        let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
        let ssim_measure = Ssim::new(11, 3, &device);
        let ssim = ssim_measure.ssim(render_rgb.clone(), gt_rgb).mean();

        EvalSample {
            index,
            view,
            psnr,
            ssim,
            rendered: render_rgb,
            aux,
        }
    })
}
