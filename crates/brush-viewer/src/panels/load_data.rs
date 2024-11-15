use crate::{viewer::ViewerContext, ViewerPanel};
use brush_dataset::{LoadDatasetArgs, LoadInitArgs};
use brush_train::train::TrainConfig;
use egui::Slider;

enum Quality {
    Low,
    Normal,
}

pub(crate) struct LoadDataPanel {
    max_train_resolution: Option<u32>,
    max_frames: Option<usize>,
    eval_split_every: Option<usize>,
    sh_degree: u32,
    quality: Quality,
    proxy: bool,
    url: String,
}

impl LoadDataPanel {
    pub(crate) fn new() -> Self {
        Self {
            // High resolution performance just isn't great at the moment... limit this for now by default.
            max_train_resolution: None,
            max_frames: None,
            eval_split_every: Some(8),
            sh_degree: 3,
            quality: Quality::Normal,
            proxy: false,
            url: "splat.com/example.ply".to_owned(),
        }
    }
}

impl ViewerPanel for LoadDataPanel {
    fn title(&self) -> String {
        "Load data".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        ui.label("Select a .ply to visualize, or a .zip with training data.");

        let file = ui.button("Load file").clicked();

        ui.add_space(10.0);

        ui.checkbox(&mut self.proxy, "Proxy proxy.brush-splat.workers.dev/")
            .on_hover_text("File hosting services often don't allow client-side requests. Using a proxy can solve this. In particular this makes google drive share links work!");

        ui.text_edit_singleline(&mut self.url);

        let url = ui.button("Load URL").clicked();

        ui.add_space(10.0);

        if file || url {
            let load_data_args = LoadDatasetArgs {
                max_frames: self.max_frames,
                max_resolution: self.max_train_resolution,
                eval_split_every: self.eval_split_every,
            };
            let load_init_args = LoadInitArgs {
                sh_degree: self.sh_degree,
            };

            let mut config = TrainConfig::default();
            if matches!(self.quality, Quality::Low) {
                config = config
                    .with_densify_grad_thresh(0.00035)
                    .with_refine_every(200);
            }

            let source = if file {
                crate::viewer::DataSource::PickFile
            } else {
                let url = if !self.proxy {
                    self.url.to_string()
                } else {
                    format!("https://proxy.brush-splat.workers.dev/{}", self.url)
                };
                crate::viewer::DataSource::Url(url)
            };
            context.start_data_load(source, load_data_args, load_init_args, config);
        }

        ui.add_space(10.0);
        ui.heading("Train settings");

        ui.label("Spherical Harmonics Degree:");
        ui.add(Slider::new(&mut self.sh_degree, 0..=4));

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

        let mut limit_res = self.max_train_resolution.is_some();
        if ui
            .checkbox(&mut limit_res, "Limit training resolution")
            .clicked()
        {
            self.max_train_resolution = if limit_res { Some(800) } else { None };
        }

        if let Some(target_res) = self.max_train_resolution.as_mut() {
            ui.add(Slider::new(target_res, 32..=2048));
        }

        let mut limit_frames = self.max_frames.is_some();
        if ui.checkbox(&mut limit_frames, "Limit max frames").clicked() {
            self.max_frames = if limit_frames { Some(32) } else { None };
        }

        if let Some(max_frames) = self.max_frames.as_mut() {
            ui.add(Slider::new(max_frames, 1..=256));
        }

        let mut use_eval_split = self.eval_split_every.is_some();
        if ui
            .checkbox(&mut use_eval_split, "Split dataset for evaluation")
            .clicked()
        {
            self.eval_split_every = if use_eval_split { Some(8) } else { None };
        }

        if let Some(eval_split) = self.eval_split_every.as_mut() {
            ui.add(
                Slider::new(eval_split, 2..=32)
                    .prefix("1 out of ")
                    .suffix(" frames"),
            );
        }

        #[cfg(not(target_family = "wasm"))]
        if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
