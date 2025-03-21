use crate::{
    app::{AppContext, AppPanel},
    running_process::start_process,
};
use brush_dataset::{LoadDataseConfig, ModelConfig};
use brush_process::{
    data_source::DataSource,
    process_loop::{ProcessArgs, ProcessConfig, RerunConfig},
};
use brush_train::train::TrainConfig;
use egui::Slider;

pub(crate) struct SettingsPanel {
    args: ProcessArgs,
    url: String,
}

impl SettingsPanel {
    pub(crate) fn new() -> Self {
        Self {
            // Nb: Important to just start with the default values here, so CLI and UI match defaults.
            args: ProcessArgs::new(
                TrainConfig::new(),
                ModelConfig::new(),
                LoadDataseConfig::new(),
                ProcessConfig::new(),
                RerunConfig::new(),
            ),
            url: "splat.com/example.ply".to_owned(),
        }
    }
}

impl AppPanel for SettingsPanel {
    fn title(&self) -> String {
        "Settings".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut AppContext) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Model Settings");
            ui.label("Spherical Harmonics Degree:");
            ui.add(Slider::new(&mut self.args.model_config.sh_degree, 0..=4));

            ui.label("Max image resolution");
            ui.add(
                Slider::new(&mut self.args.load_config.max_resolution, 32..=2048)
                    .clamping(egui::SliderClamping::Never),
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

            ui.heading("Training Settings");

            ui.horizontal(|ui| {
                ui.label("Train");

                ui.add(
                    egui::Slider::new(&mut self.args.train_config.total_steps, 1..=50000)
                        .clamping(egui::SliderClamping::Never)
                        .suffix(" steps"),
                );
            });

            ui.heading("Process Settings");

            ui.horizontal(|ui| {
                ui.label("Evaluate");
                ui.add(
                    egui::Slider::new(&mut self.args.process_config.eval_every, 1..=5000)
                        .clamping(egui::SliderClamping::Never)
                        .prefix("every ")
                        .suffix(" steps"),
                );
            });

            #[cfg(not(target_family = "wasm"))]
            {
                ui.horizontal(|ui| {
                    ui.label("Export");
                    ui.add(
                        egui::Slider::new(&mut self.args.process_config.export_every, 1..=15000)
                            .clamping(egui::SliderClamping::Never)
                            .prefix("every ")
                            .suffix(" steps"),
                    );
                });
            }

            #[cfg(all(not(target_family = "wasm"), not(target_os = "android")))]
            {
                ui.heading("Rerun Settings");

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.hyperlink_to("Rerun.io", "https://rerun.io");
                    ui.label(" settings");
                });
                let rerun_config = &mut self.args.rerun_config;
                ui.checkbox(&mut rerun_config.rerun_enabled, "Enable rerun");

                if rerun_config.rerun_enabled {
                    ui.label(
                    "Open the brush_blueprint.rbl in the rerun viewer for a good default layout.",
                );

                    ui.horizontal(|ui| {
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
                    });

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
                }
            }

            ui.add_space(20.0);

            ui.label("Select a .ply to visualize, or a .zip with training data.");

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
                context.connect_to(start_process(
                    source,
                    self.args.clone(),
                    context.device.clone(),
                    ui.ctx().clone(),
                ));
            }

            ui.add_space(10.0);
        });
    }
}
