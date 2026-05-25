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
    use brush_render::shaders::SH_C0;
    use brush_train::eval::EvalSample;
    use brush_train::msg::{RefineStats, TrainStepStats};
    use burn::tensor::ElementConversion;
    use burn::tensor::{DType, TensorData, s};

    use anyhow::Result;

    use burn_cubecl::cubecl::MemoryUsage;
    use image::{Rgb32FImage, Rgba32FImage};
    use rerun::external::glam;

    use super::VisualizeTools;

    fn histogram_fixed(values: &[f32], min: f32, max: f32, num_bins: usize) -> Vec<u64> {
        let mut bins = vec![0u64; num_bins];
        let range = (max - min).max(f32::EPSILON);
        for &v in values {
            let t = ((v - min) / range).clamp(0.0, 1.0);
            let idx = ((t * num_bins as f32) as usize).min(num_bins - 1);
            bins[idx] += 1;
        }
        bins
    }

    fn bin_centers(min: f32, max: f32, num_bins: usize) -> Vec<f32> {
        let step = (max - min) / num_bins as f32;
        (0..num_bins)
            .map(|i| min + (i as f32 + 0.5) * step)
            .collect()
    }

    impl VisualizeTools {
        #[allow(unused_variables)]
        pub async fn new(enabled: bool) -> Self {
            let rec = tokio::task::spawn_blocking(move || {
                if enabled {
                    rerun::RecordingStreamBuilder::new("Brush")
                        .spawn()
                        .expect("Failed to spawn rerun")
                } else {
                    rerun::RecordingStream::disabled()
                }
            })
            .await
            .expect("Failed to spawn rerun");

            Self {
                // Spawn rerun - creating this is already explicitly done by a user.
                rec,
            }
        }

        #[allow(unused_variables)]
        pub async fn log_splats(&self, iter: u32, splats: Splats) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);

                let means = splats.means().into_data_async().await?.into_vec::<f32>()?;
                let means = means.chunks(3).map(|c| glam::vec3(c[0], c[1], c[2]));

                let base_rgb = splats.sh_coeffs.val().slice(s![.., 0..1]) * SH_C0 + 0.5;

                let transparency = splats.opacities();

                let colors = base_rgb.into_data_async().await?.into_vec::<f32>()?;
                let colors = colors.chunks(3).map(|c| {
                    rerun::Color::from_rgb(
                        (c[0] * 255.0) as u8,
                        (c[1] * 255.0) as u8,
                        (c[2] * 255.0) as u8,
                    )
                });

                // Visualize 2 sigma, and simulate some of the small covariance blurring.
                let radii = (splats.log_scales().exp() * transparency.unsqueeze_dim(1) * 2.0
                    + 0.004)
                    .into_data_async()
                    .await?
                    .into_vec()?;

                let rotations = splats
                    .rotations()
                    .into_data_async()
                    .await?
                    .into_vec::<f32>()?;
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

        pub fn send_default_blueprint(&self, num_eval_views: usize) -> Result<()> {
            use rerun::blueprint::{
                Blueprint, BlueprintActivation, ContainerLike, Grid, Horizontal, Spatial2DView,
                Spatial3DView, Tabs, TimeSeriesView, Vertical,
            };

            if !self.rec.is_enabled() {
                return Ok(());
            }

            let scene_view = Spatial3DView::new("Scene")
                .with_origin("world")
                .with_contents(["world/**"]);

            // Up to 4 eval views laid out as a 2-column grid (1 view stays as a single row).
            let visible_eval = num_eval_views.min(4);
            let eval_cells: Vec<ContainerLike> = (0..visible_eval)
                .map(|i| {
                    Horizontal::new([
                        Spatial2DView::new("Ground truth")
                            .with_origin(format!("eval/view_{i}/ground_truth"))
                            .with_contents(["$origin/**"])
                            .into(),
                        Spatial2DView::new("Render")
                            .with_origin(format!("eval/view_{i}/render"))
                            .with_contents(["$origin/**"])
                            .into(),
                    ])
                    .with_name(format!("view {i}"))
                    .into()
                })
                .collect();

            let main_row = match visible_eval {
                0 => Horizontal::new([scene_view.into()]),
                1 => Horizontal::new([
                    eval_cells.into_iter().next().expect("len 1"),
                    scene_view.into(),
                ])
                .with_column_shares([3.0, 1.0]),
                _ => Horizontal::new([
                    Grid::new(eval_cells).with_grid_columns(2).into(),
                    scene_view.into(),
                ])
                .with_column_shares([3.0, 1.0]),
            };

            // Default-visible: Quality (aggregate only), Splats, Memory.
            // Per-view PSNR/SSIM lives in the right-most tab strip alongside every
            // other graph so they're discoverable without crowding the default view.
            let graphs = Horizontal::new([
                TimeSeriesView::new("Quality")
                    .with_contents(["psnr/eval", "ssim/eval"])
                    .into(),
                TimeSeriesView::new("Splats")
                    .with_contents(["splats/**", "refine/effective_growth"])
                    .into(),
                TimeSeriesView::new("Memory")
                    .with_contents(["memory/**"])
                    .into(),
                Tabs::new([
                    TimeSeriesView::new("Loss")
                        .with_contents(["loss/**"])
                        .into(),
                    TimeSeriesView::new("Quality (aggregate)")
                        .with_contents(["psnr/eval", "ssim/eval"])
                        .into(),
                    TimeSeriesView::new("Quality (per view)")
                        .with_contents(["psnr/per_view/**", "ssim/per_view/**"])
                        .into(),
                    TimeSeriesView::new("Splats")
                        .with_contents(["splats/**", "refine/effective_growth"])
                        .into(),
                    TimeSeriesView::new("Memory")
                        .with_contents(["memory/**"])
                        .into(),
                    TimeSeriesView::new("Refine")
                        .with_contents(["refine/num_added", "refine/num_pruned"])
                        .into(),
                    TimeSeriesView::new("Throughput")
                        .with_contents(["train/step_ms"])
                        .into(),
                    TimeSeriesView::new("Learning rates")
                        .with_contents(["lr/**"])
                        .into(),
                ])
                .with_name("All graphs")
                .into(),
            ]);

            let root = Vertical::new([main_row.into(), graphs.into()]).with_row_shares([3.0, 2.0]);

            Blueprint::new(root)
                .with_auto_layout(false)
                .with_auto_views(false)
                .send(&self.rec, BlueprintActivation::default())?;

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
        pub async fn log_eval_sample(&self, iter: u32, index: u32, eval: EvalSample) -> Result<()> {
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

                let eval_render = tensor_into_image(eval.rendered.clone().into_data_async().await?);
                let rendered = eval_render.into_rgb8();

                let [w, h] = [rendered.width(), rendered.height()];
                let gt_rerun_img = if eval.gt_img.color().has_alpha() {
                    rerun::Image::from_rgba32(eval.gt_img.into_rgba8().into_vec(), [w, h])
                } else {
                    rerun::Image::from_rgb24(eval.gt_img.into_rgb8().into_vec(), [w, h])
                };

                self.rec
                    .log(format!("eval/view_{index}/ground_truth"), &gt_rerun_img)?;
                self.rec.log(
                    format!("eval/view_{index}/render"),
                    &rerun::Image::from_rgb24(rendered.into_vec(), [w, h]),
                )?;
                self.rec.log(
                    format!("psnr/per_view/{index}"),
                    &rerun::Scalars::new(vec![
                        eval.psnr.clone().into_scalar_async::<f32>().await? as f64,
                    ]),
                )?;
                self.rec.log(
                    format!("ssim/per_view/{index}"),
                    &rerun::Scalars::new(vec![
                        eval.ssim.clone().into_scalar_async::<f32>().await? as f64,
                    ]),
                )?;
            }

            Ok(())
        }

        #[allow(unused_variables)]
        pub fn log_splat_stats(&self, iter: u32, num_splats: u32) -> Result<()> {
            if self.rec.is_enabled() {
                self.rec.set_time_sequence("iterations", iter);
                self.rec.log(
                    "splats/num_splats",
                    &rerun::Scalars::new(vec![num_splats as f64]),
                )?;
            }
            Ok(())
        }

        pub fn is_enabled(&self) -> bool {
            self.rec.is_enabled()
        }

        pub async fn log_train_stats(
            &self,
            iter: u32,
            stats: &TrainStepStats,
            step_duration: std::time::Duration,
        ) -> Result<()> {
            if !self.rec.is_enabled() {
                return Ok(());
            }
            self.rec.set_time_sequence("iterations", iter);
            // Reading the loss scalar forces a GPU readback, so it's gated on
            // logging being enabled and only happens on logging iters (the
            // caller decides the cadence).
            let loss = stats.loss.clone().into_scalar_async::<f32>().await? as f64;
            self.rec
                .log("loss/total", &rerun::Scalars::new(vec![loss]))?;
            self.rec.log(
                "train/step_ms",
                &rerun::Scalars::new(vec![step_duration.as_secs_f64() * 1000.0]),
            )?;
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
                "splats/splats_visible",
                &rerun::Scalars::new(vec![stats.num_visible as f64]),
            )?;
            Ok(())
        }

        pub async fn log_histograms(&self, iter: u32, splats: &Splats) -> Result<()> {
            if !self.rec.is_enabled() {
                return Ok(());
            }
            self.rec.set_time_sequence("iterations", iter);

            let num_bins = 32usize;

            // Opacity is a [0, 1] sigmoid output — fixed-range histogram works directly.
            let opac = splats
                .opacities()
                .into_data_async()
                .await?
                .into_vec::<f32>()?;
            let opac_bins = histogram_fixed(&opac, 0.0, 1.0, num_bins);
            let opac_centers = bin_centers(0.0, 1.0, num_bins);
            self.rec.log(
                "histograms/opacity",
                &rerun::BarChart::new(opac_bins).with_abscissa(opac_centers),
            )?;

            // Log-scale is stored directly in the splat params; binning it in log
            // space gives a useful long-tail view of splat sizes.
            let log_scales_data = splats
                .log_scales()
                .into_data_async()
                .await?
                .into_vec::<f32>()?;
            let mean_log_scale: Vec<f32> = log_scales_data
                .chunks(3)
                .map(|c| (c[0] + c[1] + c[2]) / 3.0)
                .collect();
            let (scale_lo, scale_hi) = (-10.0_f32, 2.0_f32);
            let scale_bins = histogram_fixed(&mean_log_scale, scale_lo, scale_hi, num_bins);
            let scale_centers = bin_centers(scale_lo, scale_hi, num_bins);
            self.rec.log(
                "histograms/log_scale",
                &rerun::BarChart::new(scale_bins).with_abscissa(scale_centers),
            )?;

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
    use burn::tensor::{DType, TensorData};

    use super::VisualizeTools;
    use anyhow::Result;
    use burn_cubecl::cubecl::MemoryUsage;

    impl VisualizeTools {
        pub async fn new(_enabled: bool) -> Self {
            Self {}
        }

        pub async fn log_splats(&self, _iter: u32, _splats: Splats) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_scene(&self, _scene: &Scene, _max_img_size: u32) -> Result<()> {
            Ok(())
        }

        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn send_default_blueprint(&self, _num_eval_views: usize) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_eval_stats(&self, _iter: u32, _avg_psnr: f32, _avg_ssim: f32) -> Result<()> {
            Ok(())
        }

        pub async fn log_eval_sample(
            &self,
            _iter: u32,
            _index: u32,
            _eval: EvalSample,
        ) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_splat_stats(&self, _iter: u32, _num_splats: u32) -> Result<()> {
            Ok(())
        }

        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn is_enabled(&self) -> bool {
            false
        }

        #[allow(unused_variables)]
        pub async fn log_train_stats(
            &self,
            _iter: u32,
            _stats: &TrainStepStats,
            _step_duration: std::time::Duration,
        ) -> Result<()> {
            Ok(())
        }

        #[allow(unused_variables)]
        pub async fn log_histograms(&self, _iter: u32, _splats: &Splats) -> Result<()> {
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
