use crate::{viewer::ViewerContext, ViewerPanel};
use brush_dataset::{LoadDatasetArgs, LoadInitArgs};
use brush_train::train::TrainConfig;
use egui::Slider;

enum Quality {
    Low,
    Normal,
}

pub(crate) struct LoadDataPanel {
    load_args: LoadDatasetArgs,
    init_args: LoadInitArgs,
    quality: Quality,
    url: String,
}

impl LoadDataPanel {
    pub(crate) fn new() -> Self {
        Self {
            // Super high resolutions are a bit sketchy. Limit to at least
            // some size.
            load_args: LoadDatasetArgs {
                max_resolution: Some(1920),
                ..Default::default()
            },
            init_args: LoadInitArgs::default(),
            quality: Quality::Normal,
            url: "splat.com/example.ply".to_owned(),
        }
    }
}

impl ViewerPanel for LoadDataPanel {
    fn title(&self) -> String {
        "Load data".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label("Select a .ply to visualize, or a .zip with training data.");

            let file = ui.button("Load file").clicked();

            ui.add_space(10.0);
            ui.text_edit_singleline(&mut self.url);

            let url = ui.button("Load URL").clicked();

            ui.add_space(10.0);

            if file || url {
                let mut config = TrainConfig::default();
                if matches!(self.quality, Quality::Low) {
                    config = config
                        .with_densify_grad_thresh(0.0003)
                        .with_refine_every(150)
                        .with_ssim_weight(0.0)
                        .with_cull_opacity(0.01);
                }

                let source = if file {
                    crate::viewer::DataSource::PickFile
                } else {
                    crate::viewer::DataSource::Url(self.url.to_string())
                };
                context.start_data_load(
                    source,
                    self.load_args.clone(),
                    self.init_args.clone(),
                    config,
                );
            }

            ui.add_space(10.0);
            ui.heading("Train settings");

            ui.label("Spherical Harmonics Degree:");
            ui.add(Slider::new(&mut self.init_args.sh_degree, 0..=4));

            ui.horizontal(|ui| {
                ui.label("Quality:");
                if ui
                    .selectable_label(matches!(self.quality, Quality::Low), "Low")
                    .clicked()
                {
                    self.quality = Quality::Low;
                }
                if ui
                    .selectable_label(matches!(self.quality, Quality::Normal), "Normal")
                    .clicked()
                {
                    self.quality = Quality::Normal;
                }
            });

            let mut limit_res = self.load_args.max_resolution.is_some();
            if ui
                .checkbox(&mut limit_res, "Limit training resolution")
                .clicked()
            {
                self.load_args.max_resolution = if limit_res { Some(800) } else { None };
            }

            if let Some(target_res) = self.load_args.max_resolution.as_mut() {
                ui.add(Slider::new(target_res, 32..=2048));
            }

            let mut limit_frames = self.load_args.max_frames.is_some();
            if ui.checkbox(&mut limit_frames, "Limit max frames").clicked() {
                self.load_args.max_frames = if limit_frames { Some(32) } else { None };
            }

            if let Some(max_frames) = self.load_args.max_frames.as_mut() {
                ui.add(Slider::new(max_frames, 1..=256));
            }

            let mut use_eval_split = self.load_args.eval_split_every.is_some();
            if ui
                .checkbox(&mut use_eval_split, "Split dataset for evaluation")
                .clicked()
            {
                self.load_args.eval_split_every = if use_eval_split { Some(8) } else { None };
            }

            if let Some(eval_split) = self.load_args.eval_split_every.as_mut() {
                ui.add(
                    Slider::new(eval_split, 2..=32)
                        .prefix("1 out of ")
                        .suffix(" frames"),
                );
            }

            let mut use_frame_subsample = self.load_args.subsample_frames.is_some();
            if ui
                .checkbox(&mut use_frame_subsample, "Subsample frames")
                .clicked()
            {
                self.load_args.subsample_frames = if use_frame_subsample { Some(2) } else { None };
            }

            if let Some(subsample_frames) = self.load_args.subsample_frames.as_mut() {
                ui.add(
                    Slider::new(subsample_frames, 2..=32)
                        .prefix("Keep 1 out of ")
                        .suffix(" frames"),
                );
            }

            let mut use_point_subsample = self.load_args.subsample_points.is_some();
            if ui
                .checkbox(&mut use_point_subsample, "Subsample points")
                .clicked()
            {
                self.load_args.subsample_points = if use_point_subsample { Some(2) } else { None };
            }

            if let Some(subsample_points) = self.load_args.subsample_points.as_mut() {
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
