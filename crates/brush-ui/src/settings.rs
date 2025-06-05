use crate::{BrushUiProcess, panels::AppPanel};
use brush_process::config::ProcessArgs;
use brush_vfs::DataSource;
use egui::{Align2, Slider};
use tokio::sync::oneshot::Sender;

pub struct SettingsPanel {
    args: ProcessArgs,
    url: String,

    send_args: Option<Sender<ProcessArgs>>,
}

impl SettingsPanel {
    pub(crate) fn new() -> Self {
        Self {
            // Nb: Important to just start with the default values here, so CLI and UI match defaults.
            args: ProcessArgs::default(),
            url: "splat.com/example.ply".to_owned(),
            send_args: None,
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

        // TODO: In the ideal world we'd reflect things here and generate this based on some attributes, but idk.
        egui::Window::new("Settings")
            .resizable(true)
            .collapsible(false)
            .default_pos(ui.ctx().screen_rect().center())
            .pivot(Align2::CENTER_CENTER)
            .show(ui.ctx(), |ui| {
                ui.heading("Training");
                ui.add(
                    egui::Slider::new(&mut self.args.train_config.total_steps, 1..=50000)
                        .clamping(egui::SliderClamping::Never)
                        .suffix(" steps"),
                );

                ui.label("Max Splats");
                ui.add(
                    Slider::new(&mut self.args.train_config.max_splats, 1000000..=10000000)
                        .custom_formatter(|n, _| {
                            let k_value = n as f32 / 1000.0;
                            format!("{k_value:.0}k")
                        })
                        .clamping(egui::SliderClamping::Never),
                );

                ui.collapsing("Learning rates", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.lr_mean, 1e-7..=1e-4)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("Mean learning rate start"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.lr_mean, 1e-7..=1e-4)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("Mean learning rate end"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.mean_noise_weight, 1e3..=1e5)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("Mean noise weight"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.lr_coeffs_dc, 1e-4..=1e-2)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("SH coefficients"),
                    );

                    ui.add(
                        egui::Slider::new(
                            &mut self.args.train_config.lr_coeffs_sh_scale,
                            1.0..=50.0,
                        )
                        .clamping(egui::SliderClamping::Never)
                        .text("SH division for higher orders"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.lr_opac, 1e-3..=1e-1)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("opacity"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.lr_scale, 1e-3..=1e-1)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("scale"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.lr_scale_end, 1e-4..=1e-2)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("scale (end)"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.lr_rotation, 1e-4..=1e-2)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true)
                            .text("rotation"),
                    );
                });

                ui.collapsing("Growth & refinement", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.refine_every, 50..=300)
                            .clamping(egui::SliderClamping::Never)
                            .text("Refinement frequency"),
                    );

                    ui.add(
                        egui::Slider::new(
                            &mut self.args.train_config.growth_grad_threshold,
                            0.0001..=0.001,
                        )
                        .clamping(egui::SliderClamping::Never)
                        .logarithmic(true)
                        .text("Growth threshold"),
                    );

                    ui.add(
                        egui::Slider::new(
                            &mut self.args.train_config.growth_select_fraction,
                            0.01..=0.2,
                        )
                        .clamping(egui::SliderClamping::Never)
                        .text("Growth selection fraction"),
                    );

                    ui.add(
                        egui::Slider::new(
                            &mut self.args.train_config.growth_stop_iter,
                            5000..=20000,
                        )
                        .clamping(egui::SliderClamping::Never)
                        .text("Growth stop iteration"),
                    );
                });

                ui.collapsing("Losses", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.args.train_config.ssim_weight, 0.0..=1.0)
                            .clamping(egui::SliderClamping::Never)
                            .text("ssim weight"),
                    );

                    ui.add(
                        egui::Slider::new(
                            &mut self.args.train_config.opac_loss_weight,
                            1e-9..=1e-7,
                        )
                        .clamping(egui::SliderClamping::Never)
                        .logarithmic(true)
                        .text("Splat opacity loss weight"),
                    );

                    ui.add(
                        egui::Slider::new(
                            &mut self.args.train_config.match_alpha_weight,
                            0.01..=1.0,
                        )
                        .clamping(egui::SliderClamping::Never)
                        .text("Alpha match weight"),
                    );
                });

                ui.add_space(15.0);

                ui.heading("Model");
                ui.label("Spherical Harmonics Degree:");
                ui.add(Slider::new(&mut self.args.model_config.sh_degree, 0..=4));

                ui.add_space(15.0);

                ui.heading("Dataset");
                ui.label("Max image resolution");
                ui.add(
                    Slider::new(&mut self.args.load_config.max_resolution, 32..=2048)
                        .clamping(egui::SliderClamping::Never),
                );

                let mut limit_frames = self.args.load_config.max_frames.is_some();
                if ui.checkbox(&mut limit_frames, "Limit max frames").clicked() {
                    self.args.load_config.max_frames = if limit_frames { Some(32) } else { None };
                }

                if let Some(max_frames) = self.args.load_config.max_frames.as_mut() {
                    ui.add(Slider::new(max_frames, 1..=256).clamping(egui::SliderClamping::Never));
                }

                let mut use_eval_split = self.args.load_config.eval_split_every.is_some();
                if ui
                    .checkbox(&mut use_eval_split, "Split dataset for evaluation")
                    .clicked()
                {
                    self.args.load_config.eval_split_every =
                        if use_eval_split { Some(8) } else { None };
                }

                if let Some(eval_split) = self.args.load_config.eval_split_every.as_mut() {
                    ui.add(
                        Slider::new(eval_split, 2..=32)
                            .clamping(egui::SliderClamping::Never)
                            .prefix("1 out of ")
                            .suffix(" frames"),
                    );
                }

                let mut subsample_frames = self.args.load_config.subsample_frames.is_some();
                if ui
                    .checkbox(&mut subsample_frames, "Subsample frames")
                    .clicked()
                {
                    self.args.load_config.subsample_frames =
                        if subsample_frames { Some(2) } else { None };
                }

                if let Some(subsample) = self.args.load_config.subsample_frames.as_mut() {
                    ui.add(
                        Slider::new(subsample, 2..=20)
                            .clamping(egui::SliderClamping::Never)
                            .prefix("Load every 1/")
                            .suffix(" frames"),
                    );
                }

                let mut subsample_points = self.args.load_config.subsample_points.is_some();
                if ui
                    .checkbox(&mut subsample_points, "Subsample points")
                    .clicked()
                {
                    self.args.load_config.subsample_points =
                        if subsample_points { Some(2) } else { None };
                }

                if let Some(subsample) = self.args.load_config.subsample_points.as_mut() {
                    ui.add(
                        Slider::new(subsample, 2..=20)
                            .clamping(egui::SliderClamping::Never)
                            .prefix("Load every 1/")
                            .suffix(" points"),
                    );
                }

                ui.add_space(15.0);

                ui.heading("Process");

                ui.label("Random seed:");
                let mut seed_str = self.args.process_config.seed.to_string();
                if ui.text_edit_singleline(&mut seed_str).changed() {
                    if let Ok(seed) = seed_str.parse::<u64>() {
                        self.args.process_config.seed = seed;
                    }
                }

                ui.label("Start at iteration:");
                ui.add(
                    egui::Slider::new(&mut self.args.process_config.start_iter, 0..=10000)
                        .clamping(egui::SliderClamping::Never),
                );

                #[cfg(not(target_family = "wasm"))]
                {
                    ui.collapsing("Export", |ui| {
                        ui.add(
                            egui::Slider::new(
                                &mut self.args.process_config.export_every,
                                1..=15000,
                            )
                            .clamping(egui::SliderClamping::Never)
                            .prefix("every ")
                            .suffix(" steps"),
                        );

                        let label = ui.label("Export path:");
                        ui.text_edit_singleline(&mut self.args.process_config.export_path).labelled_by(label.id);

                        let label = ui.label("Export filename:");
                        ui.text_edit_singleline(&mut self.args.process_config.export_name).labelled_by(label.id);
                    });
                }

                ui.collapsing("Evaluate", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.args.process_config.eval_every, 1..=5000)
                            .clamping(egui::SliderClamping::Never)
                            .prefix("every ")
                            .suffix(" steps"),
                    );

                    ui.checkbox(
                        &mut self.args.process_config.eval_save_to_disk,
                        "Save Eval images to disk",
                    );
                });

                #[cfg(all(not(target_family = "wasm"), not(target_os = "android")))]
                {
                    ui.add_space(15.0);

                    ui.add(egui::Hyperlink::from_label_and_url(egui::RichText::new("Rerun.io").heading(), "https://rerun.io"));

                    let rerun_config = &mut self.args.rerun_config;
                    ui.checkbox(&mut rerun_config.rerun_enabled, "Enable rerun");

                    if rerun_config.rerun_enabled {
                        ui.label(
                            "Open the brush_blueprint.rbl in the rerun viewer for a good default layout.",
                        );

                        ui.label("Log train stats");
                        ui.add(
                            egui::Slider::new(
                                &mut rerun_config.rerun_log_train_stats_every,
                                1..=1000,
                            )
                            .clamping(egui::SliderClamping::Never)
                            .prefix("every ")
                            .suffix(" steps"),
                        );

                        let mut visualize_splats = rerun_config.rerun_log_splats_every.is_some();
                        ui.checkbox(&mut visualize_splats, "Visualize splats");
                        if visualize_splats != rerun_config.rerun_log_splats_every.is_some() {
                            rerun_config.rerun_log_splats_every =
                                if visualize_splats { Some(500) } else { None };
                        }

                        if let Some(every) = rerun_config.rerun_log_splats_every.as_mut() {
                            ui.add(
                                egui::Slider::new(every, 1..=5000)
                                    .clamping(egui::SliderClamping::Never)
                                    .text("Visualize splats every"),
                            );
                        }

                        ui.label("Max image log size");
                        ui.add(
                            egui::Slider::new(&mut rerun_config.rerun_max_img_size, 128..=2048)
                                .clamping(egui::SliderClamping::Never)
                                .suffix(" px"),
                        );
                    }
                }


                ui.add_space(25.0);

                ui.vertical_centered_justified(|ui| {
                    if ui.add(egui::Button::new("Start")
                        .min_size(egui::vec2(150.0, 40.0))
                        .fill(egui::Color32::from_rgb(70, 130, 180))
                        .corner_radius(5.0))
                        .clicked() {
                        self.send_args
                            .take()
                            .expect("Must be some")
                            .send(self.args.clone())
                            .ok();
                    }
                });
            });
    }
}

impl AppPanel for SettingsPanel {
    fn title(&self) -> String {
        "Settings".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &dyn BrushUiProcess) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(20.0);

            let file = ui.button("Load file").clicked();

            let can_pick_dir = !cfg!(target_family = "wasm") && !cfg!(target_os = "android");
            let dir = can_pick_dir && ui.button("Load directory").clicked();

            ui.add_space(10.0);
            ui.text_edit_singleline(&mut self.url);

            let url = ui.button("Load URL").clicked();

            ui.add_space(10.0);

            if file || dir || url {
                let source = if file {
                    DataSource::PickFile
                } else if dir {
                    DataSource::PickDirectory
                } else {
                    DataSource::Url(self.url.clone())
                };

                let (sender, receiver) = tokio::sync::oneshot::channel();
                self.send_args = Some(sender);
                process.start_new_process(source, receiver);
            }
            ui.add_space(10.0);
        });

        // Draw settings window if we're loading something (if loading a ply
        // this wont' do anything, only if process args are needed).
        if process.is_loading() {
            self.ui_window(ui);
        }
    }
}
