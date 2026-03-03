use std::ops::RangeInclusive;
use std::path::PathBuf;

use brush_process::config::TrainStreamConfig;
use brush_render::AlphaMode;
use brush_render::gaussian_splats::SplatRenderMode;
use egui::{Align2, Slider, Ui};
use tokio::sync::oneshot::Sender;

pub(crate) struct SettingsPopup {
    send_args: Option<Sender<TrainStreamConfig>>,
    args: TrainStreamConfig,
    // Unique ID per instance so window state isn't persisted across popup opens
    window_id: egui::Id,
    // Path to save args.txt (directory where args.txt should be saved)
    pub(crate) base_path: Option<PathBuf>,
    // Status message for save feedback
    save_status: Option<(String, web_time::Instant)>,
}

fn slider<T>(ui: &mut Ui, value: &mut T, range: RangeInclusive<T>, text: &str, logarithmic: bool, enabled: bool)
where
    T: egui::emath::Numeric,
{
    let mut s = Slider::new(value, range).clamping(egui::SliderClamping::Never);
    if logarithmic {
        s = s.logarithmic(true);
    }
    if !text.is_empty() {
        s = s.text(text);
    }
    ui.add_enabled(enabled, s);
}

/// Draw all settings controls for a `TrainStreamConfig`.
/// When `enabled` is false, individual widgets are greyed out and non-interactive,
/// but collapsing sections and layout remain fully functional.
pub(crate) fn draw_settings(ui: &mut Ui, args: &mut TrainStreamConfig, enabled: bool) {
    ui.heading("Training");
    slider(ui, &mut args.train_config.total_steps, 1..=50000, " steps", false, enabled);

    ui.label("Max Splats Cap");
    ui.add_enabled(enabled, Slider::new(&mut args.train_config.max_splats, 1000000..=10000000)
        .custom_formatter(|n, _| format!("{:.0}k", n as f32 / 1000.0))
        .custom_parser(|str| {
            str.trim()
                .strip_suffix('k')
                .and_then(|s| s.parse::<f64>().ok().map(|n| n * 1000.0))
                .or_else(|| str.trim().parse::<f64>().ok())
        })
        .clamping(egui::SliderClamping::Never));

    ui.collapsing("Learning rates", |ui| {
        let tc = &mut args.train_config;
        slider(ui, &mut tc.lr_mean, 1e-7..=1e-4, "Mean learning rate start", true, enabled);
        slider(ui, &mut tc.lr_mean_end, 1e-7..=1e-4, "Mean learning rate end", true, enabled);
        slider(ui, &mut tc.mean_noise_weight, 0.0..=200.0, "Mean noise weight", true, enabled);
        slider(ui, &mut tc.lr_coeffs_dc, 1e-4..=1e-2, "SH coefficients", true, enabled);
        slider(ui, &mut tc.lr_coeffs_sh_scale, 1.0..=50.0, "SH division for higher orders", false, enabled);
        slider(ui, &mut tc.lr_opac, 1e-3..=1e-1, "opacity", true, enabled);
        slider(ui, &mut tc.lr_scale, 1e-3..=1e-1, "scale", true, enabled);
        slider(ui, &mut tc.lr_scale_end, 1e-4..=1e-2, "scale (end)", true, enabled);
        slider(ui, &mut tc.lr_rotation, 1e-4..=1e-2, "rotation", true, enabled);
    });

    ui.collapsing("Growth & refinement", |ui| {
        let tc = &mut args.train_config;
        slider(ui, &mut tc.refine_every, 50..=300, "Refinement frequency", false, enabled);
        slider(ui, &mut tc.growth_grad_threshold, 0.0001..=0.001, "Growth threshold", true, enabled);
        slider(ui, &mut tc.growth_select_fraction, 0.01..=0.2, "Growth selection fraction", false, enabled);
        slider(ui, &mut tc.growth_stop_iter, 5000..=20000, "Growth stop iteration", false, enabled);
    });

    ui.collapsing("Losses", |ui| {
        let tc = &mut args.train_config;
        slider(ui, &mut tc.ssim_weight, 0.0..=1.0, "ssim weight", false, enabled);
        slider(ui, &mut tc.opac_decay, 0.0..=0.01, "Splat opacity decay", true, enabled);
        slider(ui, &mut tc.scale_decay, 0.0..=0.01, "Splat scale decay", true, enabled);
        slider(ui, &mut tc.match_alpha_weight, 0.01..=1.0, "Alpha match weight", false, enabled);
    });

    ui.add_space(16.0);

    ui.heading("Model");
    ui.label("Spherical Harmonics Degree:");
    ui.add_enabled(enabled, Slider::new(&mut args.model_config.sh_degree, 0..=4));

    let mut render_mode_enabled = args.train_config.render_mode.is_some();
    ui.add_enabled(enabled, egui::Checkbox::new(&mut render_mode_enabled, "Render mode"));
    if enabled && render_mode_enabled != args.train_config.render_mode.is_some() {
        args.train_config.render_mode = if render_mode_enabled {
            Some(SplatRenderMode::Mip)
        } else {
            None
        };
    }
    if let Some(ref mut render_mode) = args.train_config.render_mode {
        ui.add_enabled_ui(enabled, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(render_mode, SplatRenderMode::Default, "Default");
                ui.selectable_value(render_mode, SplatRenderMode::Mip, "Mip");
            });
        });
    }

    ui.add_space(16.0);

    ui.heading("Dataset");
    ui.label("Max image resolution");
    slider(ui, &mut args.load_config.max_resolution, 32..=2048, "", false, enabled);

    let mut limit_frames = args.load_config.max_frames.is_some();
    ui.add_enabled(enabled, egui::Checkbox::new(&mut limit_frames, "Limit max frames"));
    if enabled && limit_frames != args.load_config.max_frames.is_some() {
        args.load_config.max_frames = if limit_frames { Some(32) } else { None };
    }
    if let Some(max_frames) = args.load_config.max_frames.as_mut() {
        slider(ui, max_frames, 1..=256, "", false, enabled);
    }

    let mut use_eval_split = args.load_config.eval_split_every.is_some();
    ui.add_enabled(enabled, egui::Checkbox::new(&mut use_eval_split, "Split dataset for evaluation"));
    if enabled && use_eval_split != args.load_config.eval_split_every.is_some() {
        args.load_config.eval_split_every = if use_eval_split { Some(8) } else { None };
    }
    if let Some(eval_split) = args.load_config.eval_split_every.as_mut() {
        ui.add_enabled(enabled, Slider::new(eval_split, 2..=32).clamping(egui::SliderClamping::Never)
            .prefix("1 out of ").suffix(" frames"));
    }

    let mut subsample_frames = args.load_config.subsample_frames.is_some();
    ui.add_enabled(enabled, egui::Checkbox::new(&mut subsample_frames, "Subsample frames"));
    if enabled && subsample_frames != args.load_config.subsample_frames.is_some() {
        args.load_config.subsample_frames = if subsample_frames { Some(2) } else { None };
    }
    if let Some(subsample) = args.load_config.subsample_frames.as_mut() {
        ui.add_enabled(enabled, Slider::new(subsample, 2..=20).clamping(egui::SliderClamping::Never)
            .prefix("Load every 1/").suffix(" frames"));
    }

    let mut subsample_points = args.load_config.subsample_points.is_some();
    ui.add_enabled(enabled, egui::Checkbox::new(&mut subsample_points, "Subsample points"));
    if enabled && subsample_points != args.load_config.subsample_points.is_some() {
        args.load_config.subsample_points = if subsample_points { Some(2) } else { None };
    }
    if let Some(subsample) = args.load_config.subsample_points.as_mut() {
        ui.add_enabled(enabled, Slider::new(subsample, 2..=20).clamping(egui::SliderClamping::Never)
            .prefix("Load every 1/").suffix(" points"));
    }

    let mut alpha_mode_enabled = args.load_config.alpha_mode.is_some();
    ui.add_enabled(enabled, egui::Checkbox::new(&mut alpha_mode_enabled, "Force alpha mode"));
    if enabled && alpha_mode_enabled != args.load_config.alpha_mode.is_some() {
        args.load_config.alpha_mode = if alpha_mode_enabled {
            Some(AlphaMode::default())
        } else {
            None
        };
    }

    if alpha_mode_enabled {
        let mut alpha_mode = args.load_config.alpha_mode.unwrap_or_default();
        ui.add_enabled_ui(enabled, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut alpha_mode, AlphaMode::Masked, "Masked");
                ui.selectable_value(&mut alpha_mode, AlphaMode::Transparent, "Transparent");
            });
        });
        if enabled {
            args.load_config.alpha_mode = Some(alpha_mode);
        }
    }

    ui.add_space(16.0);

    ui.heading("Process");
    ui.label("Random seed:");
    let mut seed_str = args.process_config.seed.to_string();
    ui.add_enabled(enabled, egui::TextEdit::singleline(&mut seed_str));
    if enabled
        && let Ok(seed) = seed_str.parse::<u64>() {
            args.process_config.seed = seed;
        }

    ui.label("Start at iteration:");
    slider(ui, &mut args.process_config.start_iter, 0..=10000, "", false, enabled);

    #[cfg(not(target_family = "wasm"))]
    ui.collapsing("Export", |ui| {
        fn text_input(ui: &mut Ui, label: &str, text: &mut String, enabled: bool) {
            let label = ui.label(label);
            ui.add_enabled(enabled, egui::TextEdit::singleline(text)).labelled_by(label.id);
        }

        let pc = &mut args.process_config;
        ui.add_enabled(enabled, Slider::new(&mut pc.export_every, 1..=15000)
            .clamping(egui::SliderClamping::Never).prefix("every ").suffix(" steps"));
        text_input(ui, "Export path:", &mut pc.export_path, enabled);
        text_input(ui, "Export filename:", &mut pc.export_name, enabled);
    });

    ui.collapsing("Evaluate", |ui| {
        let pc = &mut args.process_config;
        ui.add_enabled(enabled, Slider::new(&mut pc.eval_every, 1..=5000)
            .clamping(egui::SliderClamping::Never).prefix("every ").suffix(" steps"));
        ui.add_enabled(enabled, egui::Checkbox::new(&mut pc.eval_save_to_disk, "Save Eval images to disk"));
    });

    ui.add_space(15.0);

    #[cfg(all(not(target_family = "wasm"), not(target_os = "android")))]
    {
        ui.add(egui::Hyperlink::from_label_and_url(
            egui::RichText::new("Rerun.io").heading(), "https://rerun.io"));

        let rc = &mut args.rerun_config;
        ui.add_enabled(enabled, egui::Checkbox::new(&mut rc.rerun_enabled, "Enable rerun"));

        if rc.rerun_enabled {
            ui.label("Open the brush_blueprint.rbl in the rerun viewer for a good default layout.");

            ui.label("Log train stats");
            ui.add_enabled(enabled, Slider::new(&mut rc.rerun_log_train_stats_every, 1..=1000)
                .clamping(egui::SliderClamping::Never).prefix("every ").suffix(" steps"));

            let mut visualize_splats = rc.rerun_log_splats_every.is_some();
            ui.add_enabled(enabled, egui::Checkbox::new(&mut visualize_splats, "Visualize splats"));
            if enabled && visualize_splats != rc.rerun_log_splats_every.is_some() {
                rc.rerun_log_splats_every = if visualize_splats { Some(500) } else { None };
            }
            if let Some(every) = rc.rerun_log_splats_every.as_mut() {
                slider(ui, every, 1..=5000, "Visualize splats every", false, enabled);
            }

            ui.label("Max image log size");
            ui.add_enabled(enabled, Slider::new(&mut rc.rerun_max_img_size, 128..=2048)
                .clamping(egui::SliderClamping::Never).suffix(" px"));
        }

        ui.add_space(16.0);
    }
}

impl SettingsPopup {
    pub(crate) fn new() -> Self {
        Self {
            send_args: None,
            args: TrainStreamConfig::default(),
            window_id: egui::Id::new(rand::random::<u64>()),
            base_path: None,
            save_status: None,
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        let Some(sender) = &self.send_args else {
            return true;
        };
        sender.is_closed()
    }

    pub(crate) fn ui(&mut self, ui: &egui::Ui, center: egui::Pos2) {
        if self.is_done() {
            return;
        }

        // Show save confirmation popup
        if let Some((msg, time)) = &self.save_status
            && time.elapsed().as_secs() < 2
        {
            let popup_id = self.window_id.with("save_popup");
            egui::Window::new("Saved")
                .id(popup_id)
                .collapsible(false)
                .resizable(false)
                .title_bar(false)
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.label(egui::RichText::new(msg).size(14.0));
                });
        }

        egui::Window::new("Settings")
        .id(self.window_id)
        .resizable(true)
        .collapsible(false)
        .default_pos(center)
        .default_size([300.0, 750.0])
        .pivot(Align2::CENTER_CENTER)
        .title_bar(false)
        .show(ui.ctx(), |ui| {
            // Custom title bar with save button
            ui.horizontal(|ui| {
                ui.heading("Settings");

                if !cfg!(target_family = "wasm") {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some(save_dir) = &self.base_path
                            && ui.small_button("💾 Save").clicked()
                        {
                            let args_path = save_dir.join("args.txt");
                            let config: &TrainStreamConfig = &self.args;
                            let args = brush_process::args_file::config_to_args(config);

                            match std::fs::write(&args_path, args.join(" ")) {
                                Ok(()) => {
                                    self.save_status = Some((
                                        format!("Saved to {}", args_path.display()),
                                        web_time::Instant::now(),
                                    ));
                                }
                                Err(e) => {
                                    self.save_status = Some((
                                        format!("Failed: {e}"),
                                        web_time::Instant::now(),
                                    ));
                                }
                            }
                        }
                    });
                }
            });

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
            draw_settings(ui, &mut self.args, true);

            ui.add_space(12.0);

            ui.vertical_centered_justified(|ui| {
                if ui.add(egui::Button::new(egui::RichText::new("Start").size(14.0))
                    .min_size(egui::vec2(150.0, 36.0))
                    .fill(egui::Color32::from_rgb(70, 130, 180))
                    .corner_radius(6.0)).clicked() {
                    self.send_args.take().expect("Must be some").send(self.args.clone()).ok();
                }
            });
            });
        });
    }

    pub(crate) fn start_pick(
        &mut self,
        initial: TrainStreamConfig,
    ) -> impl Future<Output = TrainStreamConfig> + use<> {
        self.args = initial;
        let (sender, receiver) = tokio::sync::oneshot::channel();
        self.send_args = Some(sender);
        async move { receiver.await.expect("Must be some") }
    }
}
