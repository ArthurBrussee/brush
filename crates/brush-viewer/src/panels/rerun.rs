use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    train_loop::TrainMessage,
    viewer::{self, ViewerContext, ViewerMessage},
    ViewerPanel,
};

use brush_rerun::BurnToRerun;
use burn_wgpu::WgpuDevice;

use brush_render::{gaussian_splats::Splats, AutodiffBackend, Backend};
use brush_train::{image::tensor_into_image, scene::Scene};
use brush_train::{ssim::Ssim, train::TrainStepStats};
use burn::tensor::{activation::sigmoid, ElementConversion};
use rerun::{Color, FillMode, RecordingStream};
use tokio::{sync::mpsc::UnboundedSender, task};

pub struct VisualizeTools {
    rec: Option<RecordingStream>,
    task_queue: UnboundedSender<Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>>,
}

impl VisualizeTools {
    pub fn new() -> Self {
        // Spawn rerun - creating this is already explicitly done by a user.
        let rec = rerun::RecordingStreamBuilder::new("Brush").spawn().ok();

        let (queue_send, mut queue_receive) = tokio::sync::mpsc::unbounded_channel();

        // Spawn a task to handle futures one by one as they come in.
        task::spawn(async move {
            while let Some(fut) = queue_receive.recv().await {
                if let Err(e) = fut.await {
                    log::error!("Error logging to rerun: {}", e);
                }
            }
        });

        Self {
            rec,
            task_queue: queue_send,
        }
    }

    fn queue_task(&self, fut: impl Future<Output = anyhow::Result<()>> + Send + 'static) {
        // Ignore this error - if the channel is closed we just don't do anything and drop
        // the future.
        let _ = self.task_queue.send(Box::pin(fut));
    }

    pub(crate) fn log_splats<B: Backend>(self: Arc<Self>, splats: Splats<B>) {
        let Some(rec) = self.rec.clone() else {
            return;
        };

        if !rec.is_enabled() {
            return;
        }

        self.queue_task(async move {
            let means = splats
                .means
                .val()
                .into_data_async()
                .await
                .to_vec::<f32>()
                .unwrap();
            let means = means.chunks(3).map(|c| glam::vec3(c[0], c[1], c[2]));

            let base_rgb = splats
                .sh_coeffs
                .val()
                .slice([0..splats.num_splats(), 0..1, 0..3])
                * brush_render::render::SH_C0
                + 0.5;

            let transparency = sigmoid(splats.raw_opacity.val());

            let colors = base_rgb.into_data_async().await.to_vec::<f32>().unwrap();
            let colors = colors.chunks(3).map(|c| {
                Color::from_rgb(
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
                .unwrap();

            let rotations = splats
                .rotation
                .val()
                .into_data_async()
                .await
                .to_vec::<f32>()
                .unwrap();
            let rotations = rotations
                .chunks(4)
                .map(|q| glam::Quat::from_array([q[1], q[2], q[3], q[0]]));

            let radii = radii.chunks(3).map(|r| glam::vec3(r[0], r[1], r[2]));

            rec.log(
                "world/splat/points",
                &rerun::Ellipsoids3D::from_centers_and_half_sizes(means, radii)
                    .with_quaternions(rotations)
                    .with_colors(colors)
                    .with_fill_mode(FillMode::Solid),
            )?;
            Ok(())
        });
    }

    pub(crate) fn log_scene(self: Arc<Self>, scene: Scene) {
        let Some(rec) = self.rec.clone() else {
            return;
        };

        if !rec.is_enabled() {
            return;
        }

        self.queue_task(async move {
            rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN)?;

            for (i, view) in scene.views.iter().enumerate() {
                let path = format!("world/dataset/camera/{i}");
                let (width, height) = (view.image.width(), view.image.height());
                let vis_size = glam::uvec2(width, height);
                rec.log_static(
                    path.clone(),
                    &rerun::Pinhole::from_focal_length_and_resolution(
                        view.camera.focal(vis_size),
                        glam::vec2(vis_size.x as f32, vis_size.y as f32),
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
                    &rerun::Image::from_dynamic_image(view.image.as_ref().clone())?,
                )?;
            }

            Ok(())
        });
    }

    pub fn log_eval_stats<B: Backend>(
        self: Arc<Self>,
        iter: u32,
        stats: brush_train::eval::EvalStats<B>,
    ) {
        let Some(rec) = self.rec.clone() else {
            return;
        };

        if !rec.is_enabled() {
            return;
        }

        self.queue_task(async move {
            rec.set_time_sequence("iterations", iter);

            let avg_psnr =
                stats.samples.iter().map(|s| s.psnr).sum::<f32>() / (stats.samples.len() as f32);
            let avg_ssim =
                stats.samples.iter().map(|s| s.ssim).sum::<f32>() / (stats.samples.len() as f32);

            rec.log("psnr/eval", &rerun::Scalar::new(avg_psnr as f64))?;
            rec.log("ssim/eval", &rerun::Scalar::new(avg_ssim as f64))?;

            for (i, samp) in stats.samples.into_iter().enumerate() {
                let eval_render = tensor_into_image(samp.rendered.into_data_async().await);

                let rendered = eval_render.to_rgb8();

                let [w, h] = [rendered.width(), rendered.height()];
                rec.log(
                    format!("world/eval/view_{i}"),
                    &rerun::Transform3D::from_translation_rotation(
                        samp.view.camera.position,
                        samp.view.camera.rotation,
                    ),
                )?;
                rec.log(
                    format!("world/eval/view_{i}"),
                    &rerun::Pinhole::from_focal_length_and_resolution(
                        samp.view.camera.focal(glam::uvec2(w, h)),
                        glam::vec2(w as f32, h as f32),
                    ),
                )?;

                let gt_img = samp.view.image;
                let gt_rerun_img = if gt_img.color().has_alpha() {
                    rerun::Image::from_rgba32(gt_img.to_rgba8().into_vec(), [w, h])
                } else {
                    rerun::Image::from_rgb24(gt_img.to_rgb8().into_vec(), [w, h])
                };

                rec.log(format!("world/eval/view_{i}/ground_truth"), &gt_rerun_img)?;
                rec.log(
                    format!("world/eval/view_{i}/render"),
                    &rerun::Image::from_rgb24(rendered.to_vec(), [w, h]),
                )?;
                // TODO: Whats a good place for this? Maybe in eval views?
                rec.log(
                    format!("world/eval/view_{i}/tile_depth"),
                    &samp.aux.read_tile_depth().into_rerun().await,
                )?;
            }

            Ok(())
        });
    }

    pub fn log_splat_stats<B: AutodiffBackend>(&self, splats: Splats<B>) {
        let Some(rec) = self.rec.clone() else {
            return;
        };

        if !rec.is_enabled() {
            return;
        }

        self.queue_task(async move {
            rec.log(
                "splats/num_splats",
                &rerun::Scalar::new(splats.num_splats() as f64).clone(),
            )?;

            Ok(())
        });
    }

    pub fn log_train_stats<B: AutodiffBackend>(
        self: Arc<Self>,
        iter: u32,
        stats: TrainStepStats<B>,
    ) {
        let Some(rec) = self.rec.clone() else {
            return;
        };

        if !rec.is_enabled() {
            return;
        }

        self.queue_task(async move {
            rec.set_time_sequence("iterations", iter);
            rec.log("lr/mean", &rerun::Scalar::new(stats.lr_mean))?;
            rec.log("lr/rotation", &rerun::Scalar::new(stats.lr_rotation))?;
            rec.log("lr/scale", &rerun::Scalar::new(stats.lr_scale))?;
            rec.log("lr/coeffs", &rerun::Scalar::new(stats.lr_coeffs))?;
            rec.log("lr/opac", &rerun::Scalar::new(stats.lr_opac))?;

            let [batch_size, img_h, img_w, _] = stats.pred_images.dims();
            let pred_rgb =
                stats
                    .pred_images
                    .clone()
                    .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);
            let gt_rgb = stats
                .gt_images
                .clone()
                .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);
            let mse = (pred_rgb.clone() - gt_rgb.clone()).powf_scalar(2.0).mean();
            let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
            rec.log(
                "losses/main",
                &rerun::Scalar::new(stats.loss.clone().into_scalar_async().await.elem::<f64>()),
            )?;
            rec.log(
                "psnr/train",
                &rerun::Scalar::new(psnr.into_scalar_async().await.elem::<f64>()),
            )?;
            let device = gt_rgb.device();

            // TODO: Bit annoyingly expensive to recalculate this here. Idk if train stats should be split into
            // "very cheap" and somewhat more expensive stats.
            let ssim_measure = Ssim::new(11, 3, &device);
            let ssim = ssim_measure.ssim(pred_rgb.clone().unsqueeze(), gt_rgb.unsqueeze());
            rec.log(
                "ssim/train",
                &rerun::Scalar::new(ssim.into_scalar_async().await.elem::<f64>()),
            )?;

            // Not sure what's best here, atm let's just log the first batch render only.
            // Maybe could do an average instead?
            let main_aux = &stats.auxes[0];

            rec.log(
                "splats/num_intersects",
                &rerun::Scalar::new(main_aux.read_num_intersections().await as f64),
            )?;
            rec.log(
                "splats/splats_visible",
                &rerun::Scalar::new(main_aux.read_num_visible().await as f64),
            )?;

            if let Some(refine) = stats.refine {
                rec.log(
                    "refine/num_split",
                    &rerun::Scalar::new(refine.num_split as f64),
                )?;
                rec.log(
                    "refine/num_cloned",
                    &rerun::Scalar::new(refine.num_cloned as f64),
                )?;
                rec.log(
                    "refine/num_transparent_pruned",
                    &rerun::Scalar::new(refine.num_transparent_pruned as f64),
                )?;
                rec.log(
                    "refine/num_scale_pruned",
                    &rerun::Scalar::new(refine.num_scale_pruned as f64),
                )?;
            }
            Ok(())
        });
    }
}

pub(crate) struct RerunPanel {
    visualize: Option<Arc<VisualizeTools>>,
    device: WgpuDevice,
    eval_every: u32,
    eval_view_count: Option<usize>,
    log_train_stats_every: u32,
    visualize_splats_every: Option<u32>,
    ready_to_log_dataset: bool,
}

impl RerunPanel {
    pub(crate) fn new(device: WgpuDevice) -> Self {
        RerunPanel {
            visualize: None,
            eval_every: 1000,
            eval_view_count: None,
            log_train_stats_every: 50,
            visualize_splats_every: None,
            device,
            ready_to_log_dataset: false,
        }
    }
}

impl RerunPanel {}

impl ViewerPanel for RerunPanel {
    fn title(&self) -> String {
        "Rerun".to_owned()
    }

    fn on_message(&mut self, message: &ViewerMessage, context: &mut ViewerContext) {
        match message {
            ViewerMessage::StartLoading { training } => {
                if *training {
                    if self.visualize.is_some() {
                        self.visualize = Some(Arc::new(VisualizeTools::new()));
                    }
                } else {
                    self.visualize = None;
                }
            }
            ViewerMessage::DoneLoading { training } => {
                if *training {
                    self.ready_to_log_dataset = true;
                }
            }
            ViewerMessage::Splats { iter, splats } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                if let Some(every) = self.visualize_splats_every {
                    if iter % every == 0 {
                        visualize.log_splats(*splats.clone());
                    }
                }
            }
            viewer::ViewerMessage::TrainStep {
                stats,
                iter,
                timestamp: _,
            } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                // If needed, start an eval run.
                if iter % self.eval_every == 0 {
                    context.send_train_message(TrainMessage::Eval {
                        view_count: self.eval_view_count,
                    });
                }

                // Log out train stats.
                // HACK: Always log on a refine step, as they can happen off beat.
                // Not sure how to best handle this properly.
                if iter % self.log_train_stats_every == 0 || stats.refine.is_some() {
                    visualize.log_train_stats(*iter, *stats.clone());
                }
            }
            ViewerMessage::EvalResult { iter, eval } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                visualize.log_eval_stats(*iter, eval.clone());
            }

            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        let Some(visualize) = self.visualize.clone() else {
            if ui.button("Enable rerun").clicked() {
                self.visualize = Some(Arc::new(VisualizeTools::new()));
            }

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.label("Stream data to ");
                ui.hyperlink_to("Rerun.io", "https://rerun.io");
                ui.label(" for visualization");
            });

            ui.label("Install the viewer to get started.");
            ui.label("Will open the viewer if it isn't open yet. Open the viewer before enabling rerun to keep data.");
            ui.label("Open the brush_blueprint.rbl in the rerun viewer for a good default layout.");
            return;
        };

        if self.ready_to_log_dataset {
            visualize.log_scene(context.dataset.train.clone());
            self.ready_to_log_dataset = false;
        }

        ui.horizontal(|ui| {
            ui.label("Log train stats");
            ui.add(
                egui::Slider::new(&mut self.log_train_stats_every, 1..=1000)
                    .prefix("every ")
                    .suffix(" steps"),
            );
        });

        ui.horizontal(|ui| {
            ui.label("Evaluate");
            ui.add(
                egui::Slider::new(&mut self.eval_every, 1..=5000)
                    .prefix("every ")
                    .suffix(" steps"),
            );
        });

        let mut limit_eval_views = self.eval_view_count.is_some();
        ui.checkbox(&mut limit_eval_views, "Limit eval views");
        if limit_eval_views != self.eval_view_count.is_some() {
            self.eval_view_count = if limit_eval_views { Some(4) } else { None };
        }

        if let Some(count) = self.eval_view_count.as_mut() {
            ui.add(egui::Slider::new(count, 1..=100).text("Eval view count"));
        }

        let mut visualize_splats = self.visualize_splats_every.is_some();
        ui.checkbox(&mut visualize_splats, "Visualize splats");
        if visualize_splats != self.visualize_splats_every.is_some() {
            self.visualize_splats_every = if visualize_splats { Some(500) } else { None };
        }

        if let Some(every) = self.visualize_splats_every.as_mut() {
            ui.add(egui::Slider::new(every, 1..=5000).text("Visualize splats every"));
        }
    }
}
