#![allow(unused_imports)]

use std::sync::Arc;

use brush_dataset::clamp_img_to_max_size;
use brush_render::gaussian_splats::Splats;
use brush_train::eval::EvalSample;
use brush_train::{image::tensor_into_image, scene::Scene, train::RefineStats};
use brush_train::{ssim::Ssim, train::TrainStepStats};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{activation::sigmoid, ElementConversion};

use anyhow::Result;

#[cfg(not(target_family = "wasm"))]
use brush_rerun::BurnToRerun;
use burn_jit::cubecl::MemoryUsage;

pub struct VisualizeTools {
    #[cfg(not(target_family = "wasm"))]
    rec: Option<rerun::RecordingStream>,
}

impl VisualizeTools {
    #[allow(unused_variables)]
    pub fn new(enabled: bool) -> Self {
        // Spawn rerun - creating this is already explicitly done by a user.
        #[cfg(not(target_family = "wasm"))]
        if enabled {
            Self {
                rec: rerun::RecordingStreamBuilder::new("Brush")
                    .connect_tcp()
                    .ok(),
            }
        } else {
            Self { rec: None }
        }

        #[cfg(target_family = "wasm")]
        Self {}
    }

    #[allow(unused_variables)]
    pub async fn log_splats<B: Backend>(&self, iter: u32, splats: Splats<B>) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);

                let means = splats
                    .means
                    .val()
                    .into_data_async()
                    .await
                    .to_vec::<f32>()
                    .expect("Wrong type");
                let means = means.chunks(3).map(|c| glam::vec3(c[0], c[1], c[2]));

                let base_rgb = splats
                    .sh_coeffs
                    .val()
                    .slice([0..splats.num_splats(), 0..1, 0..3])
                    * brush_render::render::SH_C0
                    + 0.5;

                let transparency = sigmoid(splats.raw_opacity.val());

                let colors = base_rgb
                    .into_data_async()
                    .await
                    .to_vec::<f32>()
                    .expect("Wrong type");
                let colors = colors.chunks(3).map(|c| {
                    rerun::Color::from_rgb(
                        (c[0] * 255.0) as u8,
                        (c[1] * 255.0) as u8,
                        (c[2] * 255.0) as u8,
                    )
                });

                // Visualize 2 sigma, and simulate some of the small covariance blurring.
                let radii = (splats.log_scales.val().exp() * transparency.unsqueeze_dim(1) * 2.0
                    + 0.004)
                    .into_data_async()
                    .await
                    .to_vec()
                    .expect("Wrong type");

                let rotations = splats
                    .rotation
                    .val()
                    .into_data_async()
                    .await
                    .to_vec::<f32>()
                    .expect("Wrong type");
                let rotations = rotations
                    .chunks(4)
                    .map(|q| glam::Quat::from_array([q[1], q[2], q[3], q[0]]));

                let radii = radii.chunks(3).map(|r| glam::vec3(r[0], r[1], r[2]));

                rec.log(
                    "world/splat/points",
                    &rerun::Ellipsoids3D::from_centers_and_half_sizes(means, radii)
                        .with_quaternions(rotations)
                        .with_colors(colors)
                        .with_fill_mode(rerun::FillMode::Solid),
                )?;
            }
        }
        Ok(())
    }

    #[allow(unused_variables)]
    pub fn log_scene(&self, scene: &Scene, max_img_size: u32) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN)?;

                for (i, view) in scene.views.iter().enumerate() {
                    let path = format!("world/dataset/camera/{i}");
                    let log_img = clamp_img_to_max_size(view.image.clone(), max_img_size);

                    let img_size = glam::uvec2(log_img.width(), log_img.height());

                    rec.log_static(
                        path.clone(),
                        &rerun::Pinhole::from_focal_length_and_resolution(
                            view.camera.focal(img_size),
                            img_size.as_vec2(),
                        ),
                    )?;
                    rec.log_static(
                        path.clone(),
                        &rerun::Transform3D::from_translation_rotation(
                            view.camera.position,
                            view.camera.rotation,
                        ),
                    )?;
                    rec.log_static(
                        path + "/image",
                        &rerun::Image::from_dynamic_image(log_img.as_ref().clone())?,
                    )?;
                }
            }
        }

        Ok(())
    }

    #[allow(unused_variables)]
    pub fn log_eval_stats(&self, iter: u32, avg_psnr: f32, avg_ssim: f32) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);
                rec.log("psnr/eval", &rerun::Scalar::new(avg_psnr as f64))?;
                rec.log("ssim/eval", &rerun::Scalar::new(avg_ssim as f64))?;
            }
        }
        Ok(())
    }

    #[allow(unused_variables)]
    pub async fn log_eval_sample<B: Backend>(&self, iter: u32, view: &EvalSample<B>) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);

                let eval_render = tensor_into_image(view.rendered.clone().into_data_async().await);
                let rendered = eval_render.to_rgb8();

                let [w, h] = [rendered.width(), rendered.height()];
                rec.log(
                    format!("world/eval/view_{}", view.index),
                    &rerun::Transform3D::from_translation_rotation(
                        view.view.camera.position,
                        view.view.camera.rotation,
                    ),
                )?;
                rec.log(
                    format!("world/eval/view_{}", view.index),
                    &rerun::Pinhole::from_focal_length_and_resolution(
                        view.view.camera.focal(glam::uvec2(w, h)),
                        glam::vec2(w as f32, h as f32),
                    ),
                )?;

                let gt_img = &view.view.image;
                let gt_rerun_img = if gt_img.color().has_alpha() {
                    rerun::Image::from_rgba32(gt_img.to_rgba8().into_vec(), [w, h])
                } else {
                    rerun::Image::from_rgb24(gt_img.to_rgb8().into_vec(), [w, h])
                };

                rec.log(
                    format!("world/eval/view_{}/ground_truth", view.index),
                    &gt_rerun_img,
                )?;
                rec.log(
                    format!("world/eval/view_{}/render", view.index),
                    &rerun::Image::from_rgb24(rendered.to_vec(), [w, h]),
                )?;
                // TODO: Whats a good place for this? Maybe in eval views?
                rec.log(
                    format!("world/eval/view_{}/tile_depth", view.index),
                    &view.aux.calc_tile_depth().into_rerun().await,
                )?;
            }
        }

        Ok(())
    }

    #[allow(unused_variables)]
    pub fn log_splat_stats<B: Backend>(&self, iter: u32, splats: &Splats<B>) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.clone() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);

                let num = splats.num_splats();
                rec.log("splats/num_splats", &rerun::Scalar::new(num as f64))?;
            }
        }
        Ok(())
    }

    #[allow(unused_variables)]
    pub async fn log_train_stats<B: AutodiffBackend>(
        &self,
        iter: u32,
        stats: TrainStepStats<B>,
    ) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);
                rec.log("lr/mean", &rerun::Scalar::new(stats.lr_mean))?;
                rec.log("lr/rotation", &rerun::Scalar::new(stats.lr_rotation))?;
                rec.log("lr/scale", &rerun::Scalar::new(stats.lr_scale))?;
                rec.log("lr/coeffs", &rerun::Scalar::new(stats.lr_coeffs))?;
                rec.log("lr/opac", &rerun::Scalar::new(stats.lr_opac))?;

                rec.log(
                    "splats/num_intersects",
                    &rerun::Scalar::new(
                        stats
                            .num_intersections
                            .into_scalar_async()
                            .await
                            .elem::<f64>(),
                    ),
                )?;
                rec.log(
                    "splats/splats_visible",
                    &rerun::Scalar::new(stats.num_visible.into_scalar_async().await.elem::<f64>()),
                )?;

                let [img_h, img_w, _] = stats.pred_image.dims();
                let pred_rgb = stats.pred_image.clone().slice([0..img_h, 0..img_w, 0..3]);
                let gt_rgb = stats.gt_images.clone().slice([0..img_h, 0..img_w, 0..3]);
                let mse = (pred_rgb.clone() - gt_rgb.clone()).powf_scalar(2.0).mean();
                let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
                rec.log(
                    "losses/main",
                    &rerun::Scalar::new(stats.loss.clone().into_scalar_async().await.elem::<f64>()),
                )?;

                let device = gt_rgb.device();

                // TODO: Bit annoyingly expensive to recalculate this here. Idk if train stats should be split into
                // "very cheap" and somewhat more expensive stats.
                let ssim_measure = Ssim::new(11, 3, &device);
                let ssim = ssim_measure
                    .ssim(pred_rgb.clone().unsqueeze(), gt_rgb.unsqueeze())
                    .mean();
                rec.log(
                    "ssim/train",
                    &rerun::Scalar::new(ssim.into_scalar_async().await.elem::<f64>()),
                )?;
            }
        }

        Ok(())
    }

    #[allow(unused_variables)]
    pub fn log_refine_stats(&self, iter: u32, refine: &RefineStats) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);

                let _ = rec.log(
                    "refine/num_split",
                    &rerun::Scalar::new(refine.num_split as f64),
                );
                let _ = rec.log(
                    "refine/num_cloned",
                    &rerun::Scalar::new(refine.num_cloned as f64),
                );
                let _ = rec.log(
                    "refine/num_transparent_pruned",
                    &rerun::Scalar::new(refine.num_transparent_pruned as f64),
                );
                let _ = rec.log(
                    "refine/num_scale_pruned",
                    &rerun::Scalar::new(refine.num_scale_pruned as f64),
                );
            }
        }

        Ok(())
    }

    #[allow(unused_variables)]
    pub fn log_memory(&self, iter: u32, memory: &MemoryUsage) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);

                let _ = rec.log(
                    "memory/used",
                    &rerun::Scalar::new(memory.bytes_in_use as f64),
                );

                let _ = rec.log(
                    "memory/reserved",
                    &rerun::Scalar::new(memory.bytes_reserved as f64),
                );

                let _ = rec.log(
                    "memory/allocs",
                    &rerun::Scalar::new(memory.number_allocs as f64),
                );
            }
        }

        Ok(())
    }
}
