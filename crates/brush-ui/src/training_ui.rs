use crate::ui_process::UiProcess;
use brush_process::config::TrainStreamConfig;
use brush_process::message::TrainMessage;
use brush_render::AlphaMode;
use brush_render::{MainBackend, gaussian_splats::Splats};
use egui::{Align2, Slider, Ui};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot::Sender;
use web_time::Duration;

pub(crate) struct SettingsPopup {
    send_args: Option<Sender<TrainStreamConfig>>,
    args: TrainStreamConfig,
}

impl SettingsPopup {
    pub(crate) fn new(send_args: Sender<TrainStreamConfig>) -> Self {
        Self {
            send_args: Some(send_args),
            args: TrainStreamConfig::default(),
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        let Some(sender) = &self.send_args else {
            return true;
        };
        sender.is_closed()
    }

    pub(crate) fn ui(&mut self, ui: &egui::Ui) {
        fn slider<T>(
            ui: &mut Ui,
            value: &mut T,
            range: std::ops::RangeInclusive<T>,
            text: &str,
            logarithmic: bool,
        ) where
            T: egui::emath::Numeric,
        {
            let mut s = Slider::new(value, range).clamping(egui::SliderClamping::Never);
            if logarithmic {
                s = s.logarithmic(true);
            }
            if !text.is_empty() {
                s = s.text(text);
            }
            ui.add(s);
        }

        if self.send_args.is_none() {
            return;
        }

        egui::Window::new("Settings")
        .resizable(true)
        .collapsible(false)
        .default_pos(ui.ctx().screen_rect().center())
        .default_size([300.0, 700.0])
        .pivot(Align2::CENTER_CENTER)
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Training");
            slider(ui, &mut self.args.train_config.total_steps, 1..=50000, " steps", false);

            ui.label("Max Splats");
            ui.add(Slider::new(&mut self.args.train_config.max_splats, 1000000..=10000000)
                .custom_formatter(|n, _| format!("{:.0}k", n as f32 / 1000.0))
                .custom_parser(|str| {
                    str.trim()
                        .strip_suffix('k')
                        .and_then(|s| s.parse::<f64>().ok().map(|n| n * 1000.0))
                        .or_else(|| str.trim().parse::<f64>().ok())
                })
                .clamping(egui::SliderClamping::Never));

            ui.collapsing("Learning rates", |ui| {
                let tc = &mut self.args.train_config;
                slider(ui, &mut tc.lr_mean, 1e-7..=1e-4, "Mean learning rate start", true);
                slider(ui, &mut tc.lr_mean_end, 1e-7..=1e-4, "Mean learning rate end", true);
                slider(ui, &mut tc.mean_noise_weight, 0.0..=200.0, "Mean noise weight", true);
                slider(ui, &mut tc.lr_coeffs_dc, 1e-4..=1e-2, "SH coefficients", true);
                slider(ui, &mut tc.lr_coeffs_sh_scale, 1.0..=50.0, "SH division for higher orders", false);
                slider(ui, &mut tc.lr_opac, 1e-3..=1e-1, "opacity", true);
                slider(ui, &mut tc.lr_scale, 1e-3..=1e-1, "scale", true);
                slider(ui, &mut tc.lr_scale_end, 1e-4..=1e-2, "scale (end)", true);
                slider(ui, &mut tc.lr_rotation, 1e-4..=1e-2, "rotation", true);
            });

            ui.collapsing("Growth & refinement", |ui| {
                let tc = &mut self.args.train_config;
                slider(ui, &mut tc.refine_every, 50..=300, "Refinement frequency", false);
                slider(ui, &mut tc.growth_grad_threshold, 0.0001..=0.001, "Growth threshold", true);
                slider(ui, &mut tc.growth_select_fraction, 0.01..=0.2, "Growth selection fraction", false);
                slider(ui, &mut tc.growth_stop_iter, 5000..=20000, "Growth stop iteration", false);
            });

            ui.collapsing("Losses", |ui| {
                let tc = &mut self.args.train_config;
                slider(ui, &mut tc.ssim_weight, 0.0..=1.0, "ssim weight", false);
                slider(ui, &mut tc.opac_decay, 0.0..=0.01, "Splat opacity decay", true);
                slider(ui, &mut tc.scale_decay, 0.0..=0.01, "Splat scale decay", true);
                slider(ui, &mut tc.match_alpha_weight, 0.01..=1.0, "Alpha match weight", false);
            });

            ui.add_space(15.0);

            ui.heading("Model");
            ui.label("Spherical Harmonics Degree:");
            ui.add(Slider::new(&mut self.args.model_config.sh_degree, 0..=4));

            ui.add_space(15.0);

            ui.heading("Dataset");
            ui.label("Max image resolution");
            slider(ui, &mut self.args.load_config.max_resolution, 32..=2048, "", false);


            let mut limit_frames = self.args.load_config.max_frames.is_some();
            if ui.checkbox(&mut limit_frames, "Limit max frames").clicked() {
                self.args.load_config.max_frames = if limit_frames { Some(32) } else { None };
            }
            if let Some(max_frames) = self.args.load_config.max_frames.as_mut() {
                slider(ui, max_frames, 1..=256, "", false);
            }

            let mut use_eval_split = self.args.load_config.eval_split_every.is_some();
            if ui.checkbox(&mut use_eval_split, "Split dataset for evaluation").clicked() {
                self.args.load_config.eval_split_every = if use_eval_split { Some(8) } else { None };
            }
            if let Some(eval_split) = self.args.load_config.eval_split_every.as_mut() {
                ui.add(Slider::new(eval_split, 2..=32).clamping(egui::SliderClamping::Never)
                    .prefix("1 out of ").suffix(" frames"));
            }

            let mut subsample_frames = self.args.load_config.subsample_frames.is_some();
            if ui.checkbox(&mut subsample_frames, "Subsample frames").clicked() {
                self.args.load_config.subsample_frames = if subsample_frames { Some(2) } else { None };
            }
            if let Some(subsample) = self.args.load_config.subsample_frames.as_mut() {
                ui.add(Slider::new(subsample, 2..=20).clamping(egui::SliderClamping::Never)
                    .prefix("Load every 1/").suffix(" frames"));
            }

            let mut subsample_points = self.args.load_config.subsample_points.is_some();
            if ui.checkbox(&mut subsample_points, "Subsample points").clicked() {
                self.args.load_config.subsample_points = if subsample_points { Some(2) } else { None };
            }
            if let Some(subsample) = self.args.load_config.subsample_points.as_mut() {
                ui.add(Slider::new(subsample, 2..=20).clamping(egui::SliderClamping::Never)
                    .prefix("Load every 1/").suffix(" points"));
            }

            let mut alpha_mode_enabled = self.args.load_config.alpha_mode.is_some();
            if ui.checkbox(&mut alpha_mode_enabled, "Force alpha mode").clicked() {
                self.args.load_config.alpha_mode = if alpha_mode_enabled {

                    Some(AlphaMode::default())
                } else {
                    None
                };
            }

            if alpha_mode_enabled {
                let mut alpha_mode = self.args.load_config.alpha_mode.unwrap_or_default();
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut alpha_mode, AlphaMode::Masked, "Masked");
                    ui.selectable_value(&mut alpha_mode, AlphaMode::Transparent, "Transparent");
                });
                self.args.load_config.alpha_mode = Some(alpha_mode);
            }

            ui.add_space(15.0);

            ui.heading("Process");
            ui.label("Random seed:");
            let mut seed_str = self.args.process_config.seed.to_string();
            if ui.text_edit_singleline(&mut seed_str).changed()
                && let Ok(seed) = seed_str.parse::<u64>() {
                    self.args.process_config.seed = seed;
                }

            ui.label("Start at iteration:");
            slider(ui, &mut self.args.process_config.start_iter, 0..=10000, "", false);

            #[cfg(not(target_family = "wasm"))]
            ui.collapsing("Export", |ui| {
                fn text_input(ui: &mut Ui, label: &str, text: &mut String) {
                    let label = ui.label(label);
                    ui.text_edit_singleline(text).labelled_by(label.id);
                }

                let pc = &mut self.args.process_config;
                ui.add(Slider::new(&mut pc.export_every, 1..=15000)
                    .clamping(egui::SliderClamping::Never).prefix("every ").suffix(" steps"));
                text_input(ui, "Export path:", &mut pc.export_path);
                text_input(ui, "Export filename:", &mut pc.export_name);
            });

            ui.collapsing("Evaluate", |ui| {
                let pc = &mut self.args.process_config;
                ui.add(Slider::new(&mut pc.eval_every, 1..=5000)
                    .clamping(egui::SliderClamping::Never).prefix("every ").suffix(" steps"));
                ui.checkbox(&mut pc.eval_save_to_disk, "Save Eval images to disk");
            });

            ui.add_space(15.0);

            #[cfg(all(not(target_family = "wasm"), not(target_os = "android")))]
            {
                ui.add(egui::Hyperlink::from_label_and_url(
                    egui::RichText::new("Rerun.io").heading(), "https://rerun.io"));

                let rc = &mut self.args.rerun_config;
                ui.checkbox(&mut rc.rerun_enabled, "Enable rerun");

                if rc.rerun_enabled {
                    ui.label("Open the brush_blueprint.rbl in the rerun viewer for a good default layout.");

                    ui.label("Log train stats");
                    ui.add(Slider::new(&mut rc.rerun_log_train_stats_every, 1..=1000)
                        .clamping(egui::SliderClamping::Never).prefix("every ").suffix(" steps"));

                    let mut visualize_splats = rc.rerun_log_splats_every.is_some();
                    ui.checkbox(&mut visualize_splats, "Visualize splats");
                    if visualize_splats != rc.rerun_log_splats_every.is_some() {
                        rc.rerun_log_splats_every = if visualize_splats { Some(500) } else { None };
                    }
                    if let Some(every) = rc.rerun_log_splats_every.as_mut() {
                        slider(ui, every, 1..=5000, "Visualize splats every", false);
                    }

                    ui.label("Max image log size");
                    ui.add(Slider::new(&mut rc.rerun_max_img_size, 128..=2048)
                        .clamping(egui::SliderClamping::Never).suffix(" px"));
                }

                ui.add_space(15.0);
            }

            ui.add_space(10.0);
            ui.vertical_centered_justified(|ui| {
                if ui.add(egui::Button::new("Start")
                    .min_size(egui::vec2(150.0, 40.0))
                    .fill(egui::Color32::from_rgb(70, 130, 180))
                    .corner_radius(5.0)).clicked() {
                    self.send_args.take().expect("Must be some").send(self.args.clone()).ok();
                }
            });
            });
        });
    }
}

pub struct TrainingState {
    pub train_progress: Option<(u32, u32)>, // (current_iter, total_steps, elapsed)
    pub last_train_step: Option<(Duration, u32)>, // (elapsed, iter) for calculating iter/s
    pub train_iter_per_s: f32,
    iter_per_s_samples: u32, // number of samples for smoothing ramp-up
    pub train_config: Option<TrainStreamConfig>,
    pub manual_export_iters: Vec<u32>,
    pub popup: Option<SettingsPopup>,
    pub current_splats: Option<Splats<MainBackend>>,
    pub export_channel: (
        UnboundedSender<anyhow::Error>,
        UnboundedReceiver<anyhow::Error>,
    ),
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            train_progress: None,
            last_train_step: None,
            train_iter_per_s: 0.0,
            iter_per_s_samples: 0,
            train_config: None,
            manual_export_iters: Vec::new(),
            popup: None,
            current_splats: None,
            export_channel: tokio::sync::mpsc::unbounded_channel(),
        }
    }

    pub fn reset(&mut self) {
        self.train_progress = None;
        self.last_train_step = None;
        self.train_iter_per_s = 0.0;
        self.iter_per_s_samples = 0;
        self.train_config = None;
        self.manual_export_iters.clear();
        self.current_splats = None;
    }

    pub fn on_train_message(&mut self, message: &TrainMessage) {
        match message {
            TrainMessage::TrainConfig { config } => {
                self.train_config = Some(*config.clone());
            }
            TrainMessage::TrainStep {
                iter,
                total_steps,
                total_elapsed,
                splats,
                ..
            } => {
                self.train_progress = Some((*iter, *total_steps));
                self.current_splats = Some(splats.as_ref().clone());

                // Calculate smoothed iter/s
                if let Some((last_elapsed, last_iter)) = self.last_train_step
                    && let Some(elapsed_diff) = total_elapsed.checked_sub(last_elapsed)
                {
                    let iter_diff = iter - last_iter;
                    if iter_diff > 0 && elapsed_diff.as_secs_f32() > 0.0 {
                        let current_iter_per_s = iter_diff as f32 / elapsed_diff.as_secs_f32();
                        // Gradually increase smoothing confidence over first 20 samples
                        let smoothing = (self.iter_per_s_samples as f32 / 20.0).min(1.0) * 0.95;
                        self.train_iter_per_s = smoothing * self.train_iter_per_s
                            + (1.0 - smoothing) * current_iter_per_s;
                        self.iter_per_s_samples += 1;
                    }
                }
                self.last_train_step = Some((*total_elapsed, *iter));
            }
            TrainMessage::DoneTraining => {
                if let Some((_, total)) = self.train_progress {
                    self.train_progress = Some((total, total));
                }
            }
            _ => {}
        }
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

async fn export(splat: Splats<MainBackend>) -> Result<(), anyhow::Error> {
    let data = brush_serde::splat_to_ply(splat).await?;
    rrfd::save_file("export.ply", data).await?;
    Ok(())
}

const PIN_STEM: f32 = 5.0;
const PIN_RADIUS: f32 = 3.5;
const PIN_HOVER_RADIUS: f32 = 4.5;

fn draw_pin(
    ui: &egui::Ui,
    x: f32,
    row_top: f32,
    color: egui::Color32,
    filled: bool,
    tooltip: &str,
) {
    let pin_total_height = PIN_STEM + PIN_RADIUS * 2.0;
    let hit_rect = egui::Rect::from_min_max(
        egui::pos2(x - 6.0, row_top),
        egui::pos2(x + 6.0, row_top + pin_total_height + 2.0),
    );
    let response = ui.interact(hit_rect, ui.id().with(tooltip), egui::Sense::hover());
    let radius = if response.hovered() {
        PIN_HOVER_RADIUS
    } else {
        PIN_RADIUS
    };

    // Stem
    let stem_bottom = row_top + PIN_STEM;
    ui.painter().line_segment(
        [egui::pos2(x, row_top), egui::pos2(x, stem_bottom)],
        egui::Stroke::new(1.5, color),
    );

    // Circle (radio button style)
    let circle_center = egui::pos2(x, stem_bottom + radius);
    ui.painter()
        .circle_stroke(circle_center, radius, egui::Stroke::new(1.5, color));
    if filled {
        ui.painter()
            .circle_filled(circle_center, radius * 0.5, color);
    }

    response.on_hover_text(tooltip);
}

pub fn draw_training_progress(ui: &mut egui::Ui, state: &mut TrainingState, process: &UiProcess) {
    let Some((iter, total)) = state.train_progress else {
        return;
    };

    ui.add_space(ui.available_width());

    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let progress = iter as f32 / total as f32;
        let percent = (progress * 100.0) as u32;

        let eta_text = if state.train_iter_per_s > 0.0 {
            let remaining_iters = total.saturating_sub(iter);
            let remaining_secs = (remaining_iters as f32 / state.train_iter_per_s) as u64;
            let remaining = Duration::from_secs(remaining_secs);
            format!("ETA {}", humantime::format_duration(remaining))
        } else {
            "ETA --".to_owned()
        };

        if let Some(splats) = state.current_splats.clone() {
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("⬆ Export")
                            .size(13.0)
                            .color(egui::Color32::WHITE),
                    )
                    .min_size(egui::vec2(70.0, 22.0))
                    .fill(egui::Color32::from_rgb(80, 140, 80)),
                )
                .clicked()
            {
                state.manual_export_iters.push(iter);
                let sender = state.export_channel.0.clone();
                let ctx = ui.ctx().clone();
                tokio_with_wasm::alias::task::spawn(async move {
                    if let Err(e) = export(splats).await {
                        let _ = sender.send(e);
                        ctx.request_repaint();
                    }
                });
            }
            ui.add_space(8.0);
        }

        let bar_response = ui.add(
            egui::ProgressBar::new(progress)
                .desired_width(550.0)
                .desired_height(22.0),
        );

        let bar_rect = bar_response.rect;
        let padding = 10.0;

        if let Some(config) = &state.train_config {
            let export_every = config.process_config.export_every;
            let export_color = egui::Color32::from_rgb(100, 150, 255);
            let manual_export_color = egui::Color32::from_rgb(100, 200, 100);
            let next_export = ((iter / export_every) + 1) * export_every;
            let row_top = bar_rect.bottom() + 3.0;

            let mut export_iter = export_every;
            while export_iter <= total {
                let x = bar_rect.left() + (export_iter as f32 / total as f32) * bar_rect.width();
                let completed = iter >= export_iter;
                let is_next = export_iter == next_export;
                let alpha = if completed || is_next { 1.0 } else { 0.4 };

                draw_pin(
                    ui,
                    x,
                    row_top,
                    export_color.gamma_multiply(alpha),
                    completed,
                    &format!("Export at iteration {export_iter}"),
                );
                export_iter += export_every;
            }

            for &manual_iter in &state.manual_export_iters {
                let x = bar_rect.left() + (manual_iter as f32 / total as f32) * bar_rect.width();
                draw_pin(
                    ui,
                    x,
                    row_top,
                    manual_export_color,
                    true,
                    &format!("Manual export at iteration {manual_iter}"),
                );
            }
        }

        ui.painter().text(
            egui::pos2(bar_rect.left() + padding, bar_rect.center().y),
            egui::Align2::LEFT_CENTER,
            format!("{percent}%"),
            egui::FontId::proportional(13.0),
            egui::Color32::WHITE,
        );

        let iter_text = format!("{:.1} it/s", state.train_iter_per_s);
        let dim_color = egui::Color32::from_rgb(200, 200, 200);
        let bright_color = egui::Color32::WHITE;

        let galley_eta = ui.painter().layout_no_wrap(
            eta_text.clone(),
            egui::FontId::proportional(12.0),
            bright_color,
        );
        let galley_iter = ui.painter().layout_no_wrap(
            iter_text.clone(),
            egui::FontId::proportional(11.0),
            dim_color,
        );

        let eta_width = galley_eta.size().x;
        let iter_width = galley_iter.size().x;
        let separator_width = 24.0;

        ui.painter().text(
            egui::pos2(bar_rect.right() - padding, bar_rect.center().y),
            egui::Align2::RIGHT_CENTER,
            eta_text,
            egui::FontId::proportional(12.0),
            bright_color,
        );

        ui.painter().text(
            egui::pos2(
                bar_rect.right() - padding - eta_width - separator_width / 2.0,
                bar_rect.center().y,
            ),
            egui::Align2::CENTER_CENTER,
            "-",
            egui::FontId::proportional(11.0),
            dim_color,
        );

        ui.painter().text(
            egui::pos2(
                bar_rect.right() - padding - eta_width - separator_width - iter_width,
                bar_rect.center().y,
            ),
            egui::Align2::LEFT_CENTER,
            iter_text,
            egui::FontId::proportional(11.0),
            dim_color,
        );

        ui.add_space(16.0);

        if iter == total {
            ui.label(
                egui::RichText::new("Done")
                    .size(15.0)
                    .strong()
                    .color(egui::Color32::from_rgb(100, 200, 100)),
            );
        } else {
            let paused = process.is_train_paused();
            let training_on_color = egui::Color32::from_rgb(70, 130, 180);
            let training_off_color = egui::Color32::from_rgb(80, 80, 80);
            let train_button = egui::Button::new(
                egui::RichText::new(if paused {
                    "⏸ Training"
                } else {
                    "⏵ Training"
                })
                .size(13.0)
                .color(egui::Color32::WHITE),
            )
            .min_size(egui::vec2(90.0, 26.0))
            .corner_radius(13.0)
            .fill(if paused {
                training_off_color
            } else {
                training_on_color
            });

            if ui.add(train_button).clicked() {
                process.set_train_paused(!paused);
            }

            ui.add_space(4.0);

            let live_update = process.is_live_update();
            let live_on_color = egui::Color32::from_rgb(140, 50, 50);
            let live_off_color = egui::Color32::from_rgb(80, 80, 80);
            let live_button = egui::Button::new(
                egui::RichText::new("🔴 Live view")
                    .size(13.0)
                    .color(egui::Color32::WHITE),
            )
            .min_size(egui::vec2(85.0, 26.0))
            .corner_radius(13.0)
            .fill(if live_update {
                live_on_color
            } else {
                live_off_color
            });

            if ui.add(live_button).clicked() {
                process.set_live_update(!live_update);
            }
        }
    });
}

pub fn draw_settings_popup(ui: &egui::Ui, state: &mut TrainingState, process: &UiProcess) {
    if let Some(popup) = &mut state.popup
        && process.is_loading()
    {
        popup.ui(ui);

        if popup.is_done() {
            state.popup = None;
        }
    }
}
