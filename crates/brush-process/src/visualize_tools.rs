#![allow(unused_imports)]

pub struct VisualizeTools {
    #[cfg(not(target_family = "wasm"))]
    rec: rerun::RecordingStream,
}

#[cfg(not(target_family = "wasm"))]
mod visualize_tools_impl {
    use std::sync::Arc;

    use brush_dataset::scene::Scene;
    use brush_render::gaussian_splats::Splats;
    use brush_render::shaders::project_visible::SH_C0;
    use brush_train::eval::EvalSample;
    use brush_train::msg::{RefineStats, TrainStepStats};
    use burn::prelude::Backend;
    use burn::tensor::backend::AutodiffBackend;
    use burn::tensor::{DType, TensorData, s};
    use burn::tensor::{ElementConversion, activation::sigmoid};

    use anyhow::Result;

    use burn_cubecl::cubecl::MemoryUsage;
    use image::{Rgb32FImage, Rgba32FImage};
    use rerun::external::glam;

    use super::VisualizeTools;

    impl VisualizeTools {
        #[allow(unused_variables)]
        pub fn new(enabled: bool) -> Self {
            if enabled {
                Self {
                    // Spawn rerun - creating this is already explicitly done by a user.
                    rec: rerun::RecordingStreamBuilder::new("Brush")
                        .spawn()
                        .expect("Failed to connect to rerun"),
                }
            } else {
                Self {
                    rec: rerun::RecordingStream::disabled(),
                }
            }
        }

        #[allow(unused_variables)]
        pub async fn log_splats<B: Backend>(&self, iter: u32, splats: Splats<B>) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);

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
                    .slice([0..splats.num_splats() as usize, 0..1])
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

                self.rec.log(
                    "world/splat/points",
                    &rerun::Ellipsoids3D::from_centers_and_half_sizes(means, radii)
                        .with_quaternions(rotations)
                        .with_colors(colors)
                        .with_fill_mode(rerun::FillMode::Solid),
                )?;
            }
            Ok(())
        }

        #[allow(unused_variables)]
        pub fn log_scene(&self, scene: &Scene, max_img_size: u32) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec
                    .log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
                for (i, view) in scene.views.iter().enumerate() {
                    let path = format!("world/dataset/camera/{i}");

                    let focal = view.camera.focal(glam::uvec2(1, 1));

                    self.rec.log_static(
                        path.clone(),
                        &rerun::Pinhole::from_fov_and_aspect_ratio(
                            view.camera.fov_y as f32,
                            focal.x / focal.y,
                        ),
                    )?;
                    self.rec.log_static(
                        path.clone(),
                        &rerun::Transform3D::from_translation_rotation(
                            view.camera.position,
                            view.camera.rotation,
                        ),
                    )?;
                }
            }

            Ok(())
        }

        #[allow(unused_variables)]
        pub fn log_eval_stats(&self, iter: u32, avg_psnr: f32, avg_ssim: f32) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);
                self.rec
                    .log("psnr/eval", &rerun::Scalars::new(vec![avg_psnr as f64]))?;
                self.rec
                    .log("ssim/eval", &rerun::Scalars::new(vec![avg_ssim as f64]))?;
            }
            Ok(())
        }

        #[allow(unused_variables)]
        pub async fn log_eval_sample<B: Backend>(
            &self,
            iter: u32,
            index: u32,
            eval: EvalSample<B>,
        ) -> Result<()> {
            if self.rec.is_enabled() {
                fn tensor_into_image(data: TensorData) -> image::DynamicImage {
                    let [h, w, c] = [data.shape[0], data.shape[1], data.shape[2]];

                    let img: image::DynamicImage = match data.dtype {
                        DType::F32 => {
                            let data = data.into_vec::<f32>().expect("Wrong type");
                            if c == 3 {
                                Rgb32FImage::from_raw(w as u32, h as u32, data)
                                    .expect("Failed to create image from tensor")
                                    .into()
                            } else if c == 4 {
                                Rgba32FImage::from_raw(w as u32, h as u32, data)
                                    .expect("Failed to create image from tensor")
                                    .into()
                            } else {
                                panic!("Unsupported number of channels: {c}");
                            }
                        }
                        _ => panic!("unsupported dtype {:?}", data.dtype),
                    };

                    img
                }

                self.rec.set_time_sequence("iterations", iter);

                let eval_render = tensor_into_image(eval.rendered.clone().into_data_async().await);
                let rendered = eval_render.into_rgb8();

                let [w, h] = [rendered.width(), rendered.height()];
                let gt_rerun_img = if eval.gt_img.color().has_alpha() {
                    rerun::Image::from_rgba32(eval.gt_img.into_rgba8().into_vec(), [w, h])
                } else {
                    rerun::Image::from_rgb24(eval.gt_img.into_rgb8().into_vec(), [w, h])
                };

                self.rec.log(
                    format!("world/eval/view_{index}/ground_truth"),
                    &gt_rerun_img,
                )?;
                self.rec.log(
                    format!("world/eval/view_{index}/render"),
                    &rerun::Image::from_rgb24(rendered.into_vec(), [w, h]),
                )?;
                self.rec.log(
                    format!("psnr/eval_{index}"),
                    &rerun::Scalars::new(vec![
                        eval.psnr.clone().into_scalar_async().await.elem::<f32>() as f64,
                    ]),
                )?;
                self.rec.log(
                    format!("ssim/eval_{index}"),
                    &rerun::Scalars::new(vec![
                        eval.ssim.clone().into_scalar_async().await.elem::<f32>() as f64,
                    ]),
                )?;
            }

            Ok(())
        }

        #[allow(unused_variables)]
        pub fn log_splat_stats<B: Backend>(&self, iter: u32, splats: &Splats<B>) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);

                let num = splats.num_splats();
                self.rec
                    .log("splats/num_splats", &rerun::Scalars::new(vec![num as f64]))?;
            }
            Ok(())
        }

        #[allow(unused_variables)]
        pub async fn log_train_stats<B: Backend>(
            &self,
            iter: u32,
            stats: TrainStepStats<B>,
        ) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);
                self.rec
                    .log("lr/mean", &rerun::Scalars::new(vec![stats.lr_mean]))?;
                self.rec
                    .log("lr/rotation", &rerun::Scalars::new(vec![stats.lr_rotation]))?;
                self.rec
                    .log("lr/scale", &rerun::Scalars::new(vec![stats.lr_scale]))?;
                self.rec
                    .log("lr/coeffs", &rerun::Scalars::new(vec![stats.lr_coeffs]))?;
                self.rec
                    .log("lr/opac", &rerun::Scalars::new(vec![stats.lr_opac]))?;

                self.rec.log(
                    "splats/num_intersects",
                    &rerun::Scalars::new(vec![
                        stats
                            .num_intersections
                            .into_scalar_async()
                            .await
                            .elem::<f64>(),
                    ]),
                )?;
                self.rec.log(
                    "splats/splats_visible",
                    &rerun::Scalars::new(vec![
                        stats.num_visible.into_scalar_async().await.elem::<f64>(),
                    ]),
                )?;

                let [img_h, img_w, _] = stats.pred_image.dims();
                let pred_rgb = stats.pred_image.clone().slice(s![.., .., 0..3]);

                self.rec.log(
                    "losses/main",
                    &rerun::Scalars::new(vec![
                        stats.loss.clone().into_scalar_async().await.elem::<f64>(),
                    ]),
                )?;
            }

            Ok(())
        }

        #[allow(unused_variables)]
        pub fn log_refine_stats(&self, iter: u32, refine: &RefineStats) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);
                self.rec.log(
                    "refine/num_added",
                    &rerun::Scalars::new(vec![refine.num_added as f64]),
                )?;
                self.rec.log(
                    "refine/num_pruned",
                    &rerun::Scalars::new(vec![refine.num_pruned as f64]),
                )?;
                self.rec.log(
                    "refine/effective_growth",
                    &rerun::Scalars::new(vec![refine.num_added as f64 - refine.num_pruned as f64]),
                )?;
            }
            Ok(())
        }

        pub fn log_memory(&self, iter: u32, memory: &MemoryUsage) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);

                self.rec.log(
                    "memory/used",
                    &rerun::Scalars::new(vec![memory.bytes_in_use as f64]),
                )?;

                self.rec.log(
                    "memory/reserved",
                    &rerun::Scalars::new(vec![memory.bytes_reserved as f64]),
                )?;

                self.rec.log(
                    "memory/allocs",
                    &rerun::Scalars::new(vec![memory.number_allocs as f64]),
                )?;
            }
            Ok(())
        }
    }
}

#[cfg(target_family = "wasm")]
mod visualize_tools_impl {
    use std::sync::Arc;

    use brush_dataset::scene::Scene;
    use brush_render::gaussian_splats::Splats;
    use brush_train::eval::EvalSample;
    use brush_train::msg::{RefineStats, TrainStepStats};
    use burn::prelude::Backend;
    use burn::tensor::backend::AutodiffBackend;
    use burn::tensor::{DType, TensorData};
    use burn::tensor::{ElementConversion, activation::sigmoid};

    use super::VisualizeTools;
    use anyhow::Result;
    use burn_cubecl::cubecl::MemoryUsage;

    impl VisualizeTools {
        pub fn new(_enabled: bool) -> Self {
            Self {}
        }

        pub async fn log_splats<B: Backend>(&self, _iter: u32, _splats: Splats<B>) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_scene(&self, _scene: &Scene, _max_img_size: u32) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_eval_stats(&self, _iter: u32, _avg_psnr: f32, _avg_ssim: f32) -> Result<()> {
            Ok(())
        }

        pub async fn log_eval_sample<B: Backend>(
            &self,
            _iter: u32,
            _index: u32,
            _eval: EvalSample<B>,
        ) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_splat_stats<B: Backend>(&self, _iter: u32, _splats: &Splats<B>) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        pub async fn log_train_stats<B: Backend>(
            &self,
            _iter: u32,
            _stats: TrainStepStats<B>,
        ) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_refine_stats(&self, _iter: u32, _refine: &RefineStats) -> Result<()> {
            Ok(())
        }

        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_memory(&self, _iter: u32, _memory: &MemoryUsage) -> Result<()> {
            Ok(())
        }
    }
}

pub use visualize_tools_impl::*;
