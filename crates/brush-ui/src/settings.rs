use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use brush_process::config::ProcessArgs;
use brush_vfs::DataSource;
use egui::{Align2, Slider, Ui};
use tokio::sync::oneshot::Sender;

pub struct SettingsPanel {
    args: ProcessArgs,
    url: String,
    send_args: Option<Sender<ProcessArgs>>,
    show_url_dialog: bool,
}

impl SettingsPanel {
    pub(crate) fn new() -> Self {
        Self {
            args: ProcessArgs::default(),
            url: "splat.com/example.ply".to_owned(),
            send_args: None,
            show_url_dialog: false,
        }
    }

    fn ui_window(&mut self, ui: &egui::Ui) {
        // Check if the receiver is closed and set send_args to None if it is
        if let Some(sender) = &self.send_args {
            if sender.is_closed() {
                self.send_args = None;
            }
        }

        if self.send_args.is_none() {
            return;
        }

        egui::Window::new("Settings")
            .resizable(true)
            .collapsible(false)
            .default_pos(ui.ctx().screen_rect().center())
            .pivot(Align2::CENTER_CENTER)
            .show(ui.ctx(), |ui| {
                // Training
                ui.heading("Training");
                slider(ui, &mut self.args.train_config.total_steps, 1..=50000, " steps", false);

                ui.label("Max Splats");
                ui.add(Slider::new(&mut self.args.train_config.max_splats, 1000000..=10000000)
                    .custom_formatter(|n, _| format!("{:.0}k", n as f32 / 1000.0))
                    .clamping(egui::SliderClamping::Never));

                ui.collapsing("Learning rates", |ui| {
                    let tc = &mut self.args.train_config;
                    slider(ui, &mut tc.lr_mean, 1e-7..=1e-4, "Mean learning rate start", true);
                    slider(ui, &mut tc.lr_mean, 1e-7..=1e-4, "Mean learning rate end", true);
                    slider(ui, &mut tc.mean_noise_weight, 1e3..=1e5, "Mean noise weight", true);
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
                    slider(ui, &mut tc.opac_loss_weight, 1e-9..=1e-7, "Splat opacity loss weight", true);
                    slider(ui, &mut tc.match_alpha_weight, 0.01..=1.0, "Alpha match weight", false);
                });

                ui.add_space(15.0);

                // Model
                ui.heading("Model");
                ui.label("Spherical Harmonics Degree:");
                ui.add(Slider::new(&mut self.args.model_config.sh_degree, 0..=4));

                ui.add_space(15.0);

                // Dataset
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

                ui.add_space(15.0);

                // Process
                ui.heading("Process");
                ui.label("Random seed:");
                let mut seed_str = self.args.process_config.seed.to_string();
                if ui.text_edit_singleline(&mut seed_str).changed() {
                    if let Ok(seed) = seed_str.parse::<u64>() {
                        self.args.process_config.seed = seed;
                    }
                }

                ui.label("Start at iteration:");
                slider(ui, &mut self.args.process_config.start_iter, 0..=10000, "", false);

                #[cfg(not(target_family = "wasm"))]
                ui.collapsing("Export", |ui| {
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

                // Rerun
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

                // Start button
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
    }
}

impl AppPane for SettingsPanel {
    fn title(&self) -> String {
        "Settings".to_owned()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(20.0);

            // Create a nice loading options UI
            let mut load_option = None;

            ui.label(
                egui::RichText::new("Load Data:")
                    .heading()
                    .color(egui::Color32::from_rgb(70, 130, 180)),
            );
            ui.add_space(5.0);

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                if ui
                    .add(
                        egui::Button::new("File")
                            .min_size(egui::vec2(50.0, 32.0))
                            .fill(egui::Color32::from_rgb(70, 130, 180))
                            .stroke(egui::Stroke::NONE),
                    )
                    .clicked()
                {
                    load_option = Some(DataSource::PickFile);
                }

                let can_pick_dir = !cfg!(target_family = "wasm") && !cfg!(target_os = "android");
                if can_pick_dir
                    && ui
                        .add(
                            egui::Button::new("Directory")
                                .min_size(egui::vec2(70.0, 32.0))
                                .fill(egui::Color32::from_rgb(70, 130, 180))
                                .stroke(egui::Stroke::NONE),
                        )
                        .clicked()
                {
                    load_option = Some(DataSource::PickDirectory);
                }

                let can_url = !cfg!(target_os = "android");
                if can_url
                    && ui
                        .add(
                            egui::Button::new("URL")
                                .min_size(egui::vec2(50.0, 32.0))
                                .fill(egui::Color32::from_rgb(70, 130, 180))
                                .stroke(egui::Stroke::NONE),
                        )
                        .clicked()
                {
                    self.show_url_dialog = true;
                }
            });

            ui.add_space(15.0);

            // URL dialog window
            if self.show_url_dialog {
                egui::Window::new("Load from URL")
                    .resizable(false)
                    .collapsible(false)
                    .default_pos(ui.ctx().screen_rect().center())
                    .pivot(Align2::CENTER_CENTER)
                    .show(ui.ctx(), |ui| {
                        ui.vertical(|ui| {
                            ui.label("Enter URL:");
                            ui.add_space(5.0);

                            let url_response = ui.add(
                                egui::TextEdit::singleline(&mut self.url)
                                    .desired_width(300.0)
                                    .hint_text("e.g., splat.com/example.ply"),
                            );

                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                if ui.button("Load").clicked() && !self.url.trim().is_empty() {
                                    load_option = Some(DataSource::Url(self.url.clone()));
                                    self.show_url_dialog = false;
                                }
                                if ui.button("Cancel").clicked() {
                                    self.show_url_dialog = false;
                                }
                            });

                            if url_response.lost_focus()
                                && ui.input(|i| i.key_pressed(egui::Key::Enter))
                                && !self.url.trim().is_empty()
                            {
                                load_option = Some(DataSource::Url(self.url.clone()));
                                self.show_url_dialog = false;
                            }
                        });
                    });
            }

            if let Some(source) = load_option {
                let (sender, receiver) = tokio::sync::oneshot::channel();
                self.send_args = Some(sender);
                process.start_new_process(source, receiver);
            }
        });

        // Draw settings window if we're loading something (if loading a ply
        // this wont' do anything, only if process args are needed).
        if process.is_loading() {
            self.ui_window(ui);
        }
    }
}

// Helper functions to reduce repetition
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

#[allow(unused)]
fn text_input(ui: &mut Ui, label: &str, text: &mut String) {
    let label = ui.label(label);
    ui.text_edit_singleline(text).labelled_by(label.id);
}
