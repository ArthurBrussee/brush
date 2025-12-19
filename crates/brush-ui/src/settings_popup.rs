use std::ops::RangeInclusive;

use brush_process::config::TrainStreamConfig;
use brush_render::AlphaMode;
use egui::{Align2, Slider, Ui};
use tokio::sync::oneshot::Sender;

pub(crate) struct SettingsPopup {
    send_args: Option<Sender<TrainStreamConfig>>,
    args: TrainStreamConfig,
}

fn slider<T>(ui: &mut Ui, value: &mut T, range: RangeInclusive<T>, text: &str, logarithmic: bool)
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
    ui.add(s);
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

    pub(crate) fn ui(&mut self, ui: &egui::Ui, center: egui::Pos2) {
        if self.send_args.is_none() {
            return;
        }

        egui::Window::new("Settings")
        .resizable(true)
        .collapsible(false)
        .default_pos(center)
        .default_size([300.0, 700.0])
        .pivot(Align2::CENTER_CENTER)
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Training");
            slider(ui, &mut self.args.train_config.total_steps, 1..=50000, " steps", false);

            ui.label("Max Splats Cap");
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

            ui.add_space(16.0);

            ui.heading("Model");
            ui.label("Spherical Harmonics Degree:");
            ui.add(Slider::new(&mut self.args.model_config.sh_degree, 0..=4));

            ui.add_space(16.0);

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

            ui.add_space(16.0);

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

                ui.add_space(16.0);
            }

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
}
