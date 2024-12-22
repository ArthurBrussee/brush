use crate::app::{AppContext, AppPanel};
use brush_dataset::{LoadDataseConfig, ModelConfig};
use brush_process::{
    data_source::DataSource,
    process_loop::{start_process, ProcessArgs, ProcessConfig, RerunConfig},
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
            args: ProcessArgs {
                // Super high resolutions are a bit sketchy. Limit to at least
                // some size.
                load_config: LoadDataseConfig {
                    max_resolution: Some(1920),
                    max_frames: None,
                    eval_split_every: None,
                    subsample_frames: None,
                    subsample_points: None,
                },
                train_config: TrainConfig::new(),
                model_config: ModelConfig::new(),
                process_config: ProcessConfig::new(),
                rerun_config: RerunConfig::new(),
            },
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

            ui.heading("Dataset Settings");
            let mut limit_res = self.args.load_config.max_resolution.is_some();
            if ui
                .checkbox(&mut limit_res, "Limit training resolution")
                .clicked()
            {
                self.args.load_config.max_resolution = if limit_res { Some(800) } else { None };
            }

            if let Some(target_res) = self.args.load_config.max_resolution.as_mut() {
                ui.add(Slider::new(target_res, 32..=2048));
            }

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

            let mut use_frame_subsample = self.args.load_config.subsample_frames.is_some();
            if ui
                .checkbox(&mut use_frame_subsample, "Subsample frames")
                .clicked()
            {
                self.args.load_config.subsample_frames =
                    if use_frame_subsample { Some(2) } else { None };
            }

            if let Some(subsample_frames) = self.args.load_config.subsample_frames.as_mut() {
                ui.add(
                    Slider::new(subsample_frames, 2..=32)
                        .clamping(egui::SliderClamping::Never)
                        .prefix("Keep 1 out of ")
                        .suffix(" frames"),
                );
            }

            let mut use_point_subsample = self.args.load_config.subsample_points.is_some();
            if ui
                .checkbox(&mut use_point_subsample, "Subsample points")
                .clicked()
            {
                self.args.load_config.subsample_points =
                    if use_point_subsample { Some(2) } else { None };
            }

            if let Some(subsample_points) = self.args.load_config.subsample_points.as_mut() {
                ui.add(
                    Slider::new(subsample_points, 2..=32)
                        .clamping(egui::SliderClamping::Never)
                        .prefix("Keep 1 out of ")
                        .suffix(" points"),
                );
            }

            ui.heading("Training Settings");

            ui.add(
                egui::Slider::new(&mut self.args.train_config.total_steps, 1..=50000)
                    .clamping(egui::SliderClamping::Never)
                    .suffix(" steps"),
            );

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
                ));
            }

            ui.add_space(10.0);
        });
    }
}
