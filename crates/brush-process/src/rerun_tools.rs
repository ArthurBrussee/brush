#![allow(unused_imports)]

use std::sync::Arc;

use brush_dataset::clamp_img_to_max_size;
use brush_dataset::scene::Scene;
use brush_render::gaussian_splats::Splats;
use brush_render::shaders::project_visible::SH_C0;
use brush_train::eval::EvalSample;
use brush_train::train::{RefineStats, TrainStepStats};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, activation::sigmoid};

use anyhow::Result;

#[cfg(not(target_family = "wasm"))]
use brush_rerun::BurnToRerun;
use burn_cubecl::cubecl::MemoryUsage;
use image::DynamicImage;

use crate::process_loop::tensor_into_image;

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

                let base_rgb =
                    splats
                        .sh_coeffs
                        .val()
                        .slice([0..splats.num_splats() as usize, 0..1, 0..3])
                        * SH_C0
                        + 0.5;

                let transparency = splats.opacities();

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
                rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
                for (i, view) in scene.views.iter().enumerate() {
                    let path = format!("world/dataset/camera/{i}");

                    let focal = view.camera.focal(glam::uvec2(1, 1));

                    rec.log_static(
                        path.clone(),
                        &rerun::Pinhole::from_fov_and_aspect_ratio(
                            view.camera.fov_y as f32,
                            focal.x / focal.y,
                        ),
                    )?;
                    rec.log_static(
                        path.clone(),
                        &rerun::Transform3D::from_translation_rotation(
                            view.camera.position,
                            view.camera.rotation,
                        ),
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
    pub async fn log_eval_sample<B: Backend>(
        &self,
        iter: u32,
        index: u32,
        eval: &EvalSample<B>,
    ) -> Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = self.rec.as_ref() {
            if rec.is_enabled() {
                rec.set_time_sequence("iterations", iter);

                let eval_render = tensor_into_image(eval.rendered.clone().into_data_async().await);
                let rendered = eval_render.to_rgb8();

                let [w, h] = [rendered.width(), rendered.height()];
                let gt_rerun_img = if eval.gt_img.color().has_alpha() {
                    rerun::Image::from_rgba32(eval.gt_img.to_rgba8().into_vec(), [w, h])
                } else {
                    rerun::Image::from_rgb24(eval.gt_img.to_rgb8().into_vec(), [w, h])
                };

                rec.log(
                    format!("world/eval/view_{index}/ground_truth"),
                    &gt_rerun_img,
                )?;
                rec.log(
                    format!("world/eval/view_{index}/render"),
                    &rerun::Image::from_rgb24(rendered.to_vec(), [w, h]),
                )?;
                rec.log(
                    format!("psnr/eval_{index}"),
                    &rerun::Scalar::new(
                        eval.psnr.clone().into_scalar_async().await.elem::<f32>() as f64
                    ),
                )?;
                rec.log(
                    format!("ssim/eval_{index}"),
                    &rerun::Scalar::new(
                        eval.ssim.clone().into_scalar_async().await.elem::<f32>() as f64
                    ),
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

                rec.log(
                    "losses/main",
                    &rerun::Scalar::new(stats.loss.clone().into_scalar_async().await.elem::<f64>()),
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
                    "refine/num_added",
                    &rerun::Scalar::new(refine.num_added as f64),
                );
                let _ = rec.log(
                    "refine/num_pruned",
                    &rerun::Scalar::new(refine.num_pruned as f64),
                );
                let _ = rec.log(
                    "refine/effective_growth",
                    &rerun::Scalar::new(refine.num_added as f64 - refine.num_pruned as f64),
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
