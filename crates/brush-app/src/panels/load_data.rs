use crate::{
    app::{AppContext, AppPanel},
    data_source::DataSource,
    process_loop::{start_process, ProcessArgs},
};
use brush_dataset::{LoadDatasetArgs, LoadInitArgs};
use brush_train::train::TrainConfig;
use egui::Slider;

pub(crate) struct LoadDataPanel {
    args: ProcessArgs,
    url: String,
}

impl LoadDataPanel {
    pub(crate) fn new() -> Self {
        Self {
            args: ProcessArgs {
                // Super high resolutions are a bit sketchy. Limit to at least
                // some size.
                load_args: LoadDatasetArgs {
                    max_resolution: Some(1920),
                    ..Default::default()
                },
                train_config: TrainConfig::default(),
                init_args: LoadInitArgs::default(),
                source: DataSource::PickFile,
            },
            url: "splat.com/example.ply".to_owned(),
        }
    }
}

impl AppPanel for LoadDataPanel {
    fn title(&self) -> String {
        "Load data".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut AppContext) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label("Select a .ply to visualize, or a .zip with training data.");

            let file = ui.button("Load file").clicked();

            let can_pick_dir = !cfg!(target_family = "wasm") && !cfg!(target_os = "android");
            let dir = can_pick_dir && ui.button("Load directory").clicked();

            ui.add_space(10.0);
            ui.text_edit_singleline(&mut self.url);

            let url = ui.button("Load URL").clicked();

            ui.add_space(10.0);

            if file || dir || url {
                self.args.source = if file {
                    DataSource::PickFile
                } else if dir {
                    DataSource::PickDirectory
                } else {
                    DataSource::Url(self.url.clone())
                };
                context.connect_to(start_process(self.args.clone(), context.device.clone()));
            }

            ui.add_space(10.0);
            ui.heading("Train settings");

            ui.label("Spherical Harmonics Degree:");
            ui.add(Slider::new(&mut self.args.init_args.sh_degree, 0..=4));

            let mut limit_res = self.args.load_args.max_resolution.is_some();
            if ui
                .checkbox(&mut limit_res, "Limit training resolution")
                .clicked()
            {
                self.args.load_args.max_resolution = if limit_res { Some(800) } else { None };
            }

            if let Some(target_res) = self.args.load_args.max_resolution.as_mut() {
                ui.add(Slider::new(target_res, 32..=2048));
            }

            let mut limit_frames = self.args.load_args.max_frames.is_some();
            if ui.checkbox(&mut limit_frames, "Limit max frames").clicked() {
                self.args.load_args.max_frames = if limit_frames { Some(32) } else { None };
            }

            if let Some(max_frames) = self.args.load_args.max_frames.as_mut() {
                ui.add(Slider::new(max_frames, 1..=256));
            }

            let mut use_eval_split = self.args.load_args.eval_split_every.is_some();
            if ui
                .checkbox(&mut use_eval_split, "Split dataset for evaluation")
                .clicked()
            {
                self.args.load_args.eval_split_every = if use_eval_split { Some(8) } else { None };
            }

            if let Some(eval_split) = self.args.load_args.eval_split_every.as_mut() {
                ui.add(
                    Slider::new(eval_split, 2..=32)
                        .prefix("1 out of ")
                        .suffix(" frames"),
                );
            }

            ui.horizontal(|ui| {
                ui.label("Evaluate");
                ui.add(
                    egui::Slider::new(&mut self.args.train_config.eval_every, 1..=5000)
                        .prefix("every ")
                        .suffix(" steps"),
                );
            });

            let mut use_frame_subsample = self.args.load_args.subsample_frames.is_some();
            if ui
                .checkbox(&mut use_frame_subsample, "Subsample frames")
                .clicked()
            {
                self.args.load_args.subsample_frames =
                    if use_frame_subsample { Some(2) } else { None };
            }

            if let Some(subsample_frames) = self.args.load_args.subsample_frames.as_mut() {
                ui.add(
                    Slider::new(subsample_frames, 2..=32)
                        .prefix("Keep 1 out of ")
                        .suffix(" frames"),
                );
            }

            let mut use_point_subsample = self.args.load_args.subsample_points.is_some();
            if ui
                .checkbox(&mut use_point_subsample, "Subsample points")
                .clicked()
            {
                self.args.load_args.subsample_points =
                    if use_point_subsample { Some(2) } else { None };
            }

            if let Some(subsample_points) = self.args.load_args.subsample_points.as_mut() {
                ui.add(
                    Slider::new(subsample_points, 2..=32)
                        .prefix("Keep 1 out of ")
                        .suffix(" points"),
                );
            }

            #[cfg(not(target_family = "wasm"))]
            if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }
}
