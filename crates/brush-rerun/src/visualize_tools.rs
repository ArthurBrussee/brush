#![allow(unused_imports)]

pub struct VisualizeTools {
    #[cfg(not(target_family = "wasm"))]
    rec: rerun::RecordingStream,
    /// Tracks which eval view indices have had their GT image logged. GT
    /// images never change, so we log them once as static instead of paying
    /// the full readback + send cost every eval iter.
    #[cfg(not(target_family = "wasm"))]
    gt_logged: std::sync::Mutex<std::collections::HashSet<u32>>,
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
    use image::imageops::FilterType;
    use rerun::external::glam;

    use super::VisualizeTools;

    fn resize_to_max(img: image::RgbImage, max_size: u32) -> image::RgbImage {
        let (w, h) = img.dimensions();
        let longest = w.max(h);
        if max_size == 0 || longest <= max_size {
            return img;
        }
        let scale = max_size as f32 / longest as f32;
        let new_w = ((w as f32 * scale).round() as u32).max(1);
        let new_h = ((h as f32 * scale).round() as u32).max(1);
        image::imageops::resize(&img, new_w, new_h, FilterType::Triangle)
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
                gt_logged: std::sync::Mutex::default(),
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

            // Override entity-path leaves with human-friendly legend labels.
            let set_name = |path: &str, name: &str| -> Result<()> {
                self.rec.log_static(
                    path,
                    &rerun::SeriesLines::new().with_names([name.to_owned()]),
                )?;
                Ok(())
            };
            set_name("loss/total", "Loss")?;
            set_name("train/step_ms", "Step time")?;
            set_name("psnr/eval", "Avg")?;
            set_name("ssim/eval", "Avg")?;
            for i in 0..num_eval_views {
                set_name(&format!("psnr/per_view/{i}"), &format!("View {i}"))?;
                set_name(&format!("ssim/per_view/{i}"), &format!("View {i}"))?;
            }
            set_name("splats/num_splats", "Total")?;
            set_name("splats/splats_visible", "Visible")?;
            set_name("lr/mean", "Means")?;
            set_name("lr/rotation", "Rotations")?;
            set_name("lr/scale", "Scales")?;
            set_name("lr/coeffs", "SH coeffs")?;
            set_name("lr/opac", "Opacity")?;
            set_name("memory/used", "Used")?;
            set_name("memory/reserved", "Reserved")?;
            set_name("memory/allocs", "Allocations")?;
            set_name("refine/num_added", "Added (total)")?;
            set_name("refine/num_split_oversized", "Split: oversized")?;
            set_name("refine/num_split_high_grad", "Split: high-grad")?;
            set_name("refine/num_pruned", "Pruned")?;
            set_name("refine/num_pruned_non_finite", "Pruned: non-finite")?;
            set_name("refine/effective_growth", "Effective growth")?;
            set_name("refine/duration_ms", "Refine duration")?;

            let scene_view = Spatial3DView::new("Scene")
                .with_origin("world")
                .with_contents(["world/**"]);

            // Each eval view = a Horizontal[GT, Render] cell. Groups of up to 4 are
            // laid out as a 2-column Grid; if there are more than 4 views, those
            // grids become switchable tabs ("views 0-3", "views 4-7", ...).
            let eval_cell = |i: usize| -> ContainerLike {
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
            };
            let eval_group = |start: usize, end: usize| -> ContainerLike {
                let cells: Vec<ContainerLike> = (start..end).map(eval_cell).collect();
                if cells.len() == 1 {
                    cells.into_iter().next().expect("len 1")
                } else {
                    let label = format!("views {start}-{}", end - 1);
                    Grid::new(cells)
                        .with_grid_columns(2)
                        .with_name(label)
                        .into()
                }
            };

            let main_row = if num_eval_views == 0 {
                Horizontal::new([scene_view.into()])
            } else {
                let group_size = 4;
                let eval_panel: ContainerLike = if num_eval_views <= group_size {
                    eval_group(0, num_eval_views)
                } else {
                    let num_groups = num_eval_views.div_ceil(group_size);
                    let groups = (0..num_groups).map(|g| {
                        let start = g * group_size;
                        let end = (start + group_size).min(num_eval_views);
                        eval_group(start, end)
                    });
                    Tabs::new(groups).with_name("Eval views").into()
                };
                Horizontal::new([eval_panel, scene_view.into()]).with_column_shares([3.0, 1.0])
            };

            // Default-visible graph row: Quality (PSNR + per-view PSNR + SSIM +
            // per-view SSIM + Loss as a tab strip), then Splats / Refine / Memory
            // each as their own view, then an "Other" tab for the rest.
            let quality_tabs = Tabs::new([
                TimeSeriesView::new("PSNR")
                    .with_contents(["psnr/eval"])
                    .into(),
                TimeSeriesView::new("PSNR per view")
                    .with_contents(["psnr/per_view/**"])
                    .into(),
                TimeSeriesView::new("SSIM")
                    .with_contents(["ssim/eval"])
                    .into(),
                TimeSeriesView::new("SSIM per view")
                    .with_contents(["ssim/per_view/**"])
                    .into(),
                TimeSeriesView::new("Loss")
                    .with_contents(["loss/**"])
                    .into(),
            ])
            .with_name("Quality");

            let splats_view = TimeSeriesView::new("Splats").with_contents(["splats/**"]);
            let refine_view = TimeSeriesView::new("Refine").with_contents([
                "refine/num_split_oversized",
                "refine/num_split_high_grad",
                "refine/num_pruned",
                "refine/num_pruned_non_finite",
                "refine/effective_growth",
            ]);
            let memory_view = TimeSeriesView::new("Memory").with_contents(["memory/**"]);

            let other_tabs = Tabs::new([
                TimeSeriesView::new("Throughput")
                    .with_contents(["train/step_ms", "refine/duration_ms"])
                    .into(),
                TimeSeriesView::new("Learning rates")
                    .with_contents(["lr/**"])
                    .into(),
            ])
            .with_name("Other");

            let graphs = Horizontal::new([
                quality_tabs.into(),
                splats_view.into(),
                refine_view.into(),
                memory_view.into(),
                other_tabs.into(),
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

        pub async fn log_eval_sample(
            &self,
            iter: u32,
            index: u32,
            eval: EvalSample,
            max_img_size: u32,
        ) -> Result<()> {
            if !self.rec.is_enabled() {
                return Ok(());
            }

            self.rec.set_time_sequence("iterations", iter);

            // Read the rendered f32 tensor and convert straight to u8 RGB,
            // skipping the intermediate Rgb32FImage allocation.
            let data = eval.rendered.clone().into_data_async().await?;
            let [h, w, c] = [data.shape[0], data.shape[1], data.shape[2]];
            assert!(
                c == 3,
                "Expected 3-channel eval render, got {c} (would need updating to log alpha)"
            );
            let f32_buf = data.into_vec::<f32>()?;
            let render_u8: Vec<u8> = f32_buf
                .into_iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect();
            let render_img = image::RgbImage::from_raw(w as u32, h as u32, render_u8)
                .expect("Failed to build RgbImage from rendered tensor");
            let render_img = resize_to_max(render_img, max_img_size);
            let rw = render_img.width();
            let rh = render_img.height();
            self.rec.log(
                format!("eval/view_{index}/render"),
                &rerun::Image::from_rgb24(render_img.into_vec(), [rw, rh]),
            )?;

            // GT never changes. Log it once as static per view.
            let first_gt = {
                let mut logged = self.gt_logged.lock().expect("gt_logged poisoned");
                logged.insert(index)
            };
            if first_gt {
                let gt_rgb = eval.gt_img.into_rgb8();
                let gt_img = resize_to_max(gt_rgb, max_img_size);
                let gw = gt_img.width();
                let gh = gt_img.height();
                self.rec.log_static(
                    format!("eval/view_{index}/ground_truth"),
                    &rerun::Image::from_rgb24(gt_img.into_vec(), [gw, gh]),
                )?;
            }

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

        pub fn log_refine_stats(
            &self,
            iter: u32,
            refine: &RefineStats,
            refine_duration: std::time::Duration,
        ) -> Result<()> {
            if !self.rec.is_enabled() {
                return Ok(());
            }
            self.rec.set_time_sequence("iterations", iter);
            self.rec.log(
                "refine/num_added",
                &rerun::Scalars::new(vec![refine.num_added as f64]),
            )?;
            self.rec.log(
                "refine/num_split_oversized",
                &rerun::Scalars::new(vec![refine.num_split_oversized as f64]),
            )?;
            self.rec.log(
                "refine/num_split_high_grad",
                &rerun::Scalars::new(vec![refine.num_split_high_grad as f64]),
            )?;
            self.rec.log(
                "refine/num_pruned",
                &rerun::Scalars::new(vec![refine.num_pruned as f64]),
            )?;
            self.rec.log(
                "refine/num_pruned_non_finite",
                &rerun::Scalars::new(vec![refine.num_pruned_non_finite as f64]),
            )?;
            self.rec.log(
                "refine/effective_growth",
                &rerun::Scalars::new(vec![refine.num_added as f64 - refine.num_pruned as f64]),
            )?;
            self.rec.log(
                "refine/duration_ms",
                &rerun::Scalars::new(vec![refine_duration.as_secs_f64() * 1000.0]),
            )?;
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
            _max_img_size: u32,
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
        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_refine_stats(
            &self,
            _iter: u32,
            _refine: &RefineStats,
            _refine_duration: std::time::Duration,
        ) -> Result<()> {
            Ok(())
        }

        #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
        pub fn log_memory(&self, _iter: u32, _memory: &MemoryUsage) -> Result<()> {
            Ok(())
        }
    }
}

pub use visualize_tools_impl::*;
