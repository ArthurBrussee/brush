use crate::{
    viewer::{ProcessMessage, ViewerContext},
    ViewerPanel,
};
use burn_jit::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use std::{sync::Arc, time::Duration};
use web_time::Instant;
use wgpu::AdapterInfo;

pub(crate) struct StatsPanel {
    device: WgpuDevice,

    last_train_step: (Instant, u32),
    train_iter_per_s: f32,
    last_eval_psnr: Option<f32>,

    training_started: bool,
    num_splats: usize,
    frames: usize,

    start_load_time: Instant,
    adapter_info: AdapterInfo,
}

impl StatsPanel {
    pub(crate) fn new(device: WgpuDevice, adapter: Arc<wgpu::Adapter>) -> Self {
        let adapter_info = adapter.get_info();
        Self {
            device,
            last_train_step: (Instant::now(), 0),
            train_iter_per_s: 0.0,
            last_eval_psnr: None,
            training_started: false,
            num_splats: 0,
            frames: 0,
            start_load_time: Instant::now(),
            adapter_info,
        }
    }
}

fn bytes_format(bytes: u64) -> String {
    let unit = 1000;

    if bytes < unit {
        format!("{} B", bytes)
    } else {
        let size = bytes as f64;
        let exp = match size.log(1000.0).floor() as usize {
            0 => 1,
            e => e,
        };
        let unit_prefix = "KMGTPEZY".as_bytes();
        format!(
            "{:.2} {}B",
            (size / unit.pow(exp as u32) as f64),
            unit_prefix[exp - 1] as char,
        )
    }
}

impl ViewerPanel for StatsPanel {
    fn title(&self) -> String {
        "Stats".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, _: &mut ViewerContext) {
        match message {
            ProcessMessage::StartLoading { training } => {
                self.start_load_time = Instant::now();
                self.last_train_step = (Instant::now(), 0);
                self.train_iter_per_s = 0.0;
                self.num_splats = 0;
                self.last_eval_psnr = None;
                self.training_started = *training;
            }
            ProcessMessage::ViewSplats {
                up_axis: _,
                splats,
                frame,
            } => {
                self.num_splats = splats.num_splats();
                self.frames = *frame;
            }
            ProcessMessage::TrainStep {
                splats,
                stats: _,
                iter,
                timestamp,
            } => {
                self.num_splats = splats.num_splats();

                let current_iter_per_s = (iter - self.last_train_step.1) as f32
                    / (*timestamp - self.last_train_step.0).as_secs_f32();

                self.train_iter_per_s = 0.95 * self.train_iter_per_s + 0.05 * current_iter_per_s;
                self.last_train_step = (*timestamp, *iter);
            }
            ProcessMessage::EvalResult { iter: _, eval } => {
                let avg_psnr =
                    eval.samples.iter().map(|s| s.psnr).sum::<f32>() / (eval.samples.len() as f32);
                self.last_eval_psnr = Some(avg_psnr);
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &mut ViewerContext) {
        egui::Grid::new("stats_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Splats");
                ui.label(format!("{}", self.num_splats));
                ui.end_row();

                if self.frames > 0 {
                    ui.label("Frames");
                    ui.label(format!("{}", self.frames));
                    ui.end_row();
                }

                if self.training_started {
                    ui.label("Train step");
                    ui.label(format!("{}", self.last_train_step.1));
                    ui.end_row();

                    ui.label("Steps/s");
                    ui.label(format!("{:.1}", self.train_iter_per_s));
                    ui.end_row();

                    ui.label("Last eval PSNR");
                    ui.label(if let Some(psnr) = self.last_eval_psnr {
                        format!("{:.}", psnr)
                    } else {
                        "--".to_owned()
                    });
                    ui.end_row();

                    ui.label("Training time");
                    // Round duration to seconds.
                    let elapsed =
                        Duration::from_secs((Instant::now() - self.start_load_time).as_secs());
                    ui.label(format!("{}", humantime::Duration::from(elapsed)));
                    ui.end_row();
                }

                let client = WgpuRuntime::client(&self.device);
                let memory = client.memory_usage();

                ui.label("GPU memory");
                ui.end_row();

                ui.label("Bytes in use");
                ui.label(bytes_format(memory.bytes_in_use));
                ui.end_row();

                ui.label("Bytes reserved");
                ui.label(bytes_format(memory.bytes_reserved));
                ui.end_row();

                ui.label("Active allocations");
                ui.label(format!("{}", memory.number_allocs));
                ui.end_row();
            });

        // On WASM, adapter info is mostly private, not worth showing.
        if !cfg!(target_family = "wasm") {
            egui::Grid::new("gpu_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("GPU");
                    ui.end_row();

                    ui.label("Name");
                    ui.label(&self.adapter_info.name);
                    ui.end_row();

                    ui.label("Type");
                    ui.label(format!("{:?}", self.adapter_info.device_type));
                    ui.end_row();

                    ui.label("Driver");
                    ui.label(format!(
                        "{}, {}",
                        self.adapter_info.driver, self.adapter_info.driver_info
                    ));
                    ui.end_row();
                });
        }
    }
}
