use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use brush_dataset::Dataset;
use brush_process::message::ProcessMessage;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use web_time::Duration;
use wgpu::AdapterInfo;

pub struct StatsPanel {
    device: WgpuDevice,
    last_train_step: (Duration, u32),
    train_iter_per_s: f32,
    last_eval: Option<String>,
    cur_sh_degree: u32,
    num_splats: u32,
    frames: u32,
    cur_dataset: Dataset,
    adapter_info: AdapterInfo,
}

impl StatsPanel {
    pub(crate) fn new(device: WgpuDevice, adapter_info: AdapterInfo) -> Self {
        Self {
            device,
            last_train_step: (Duration::from_secs(0), 0),
            train_iter_per_s: 0.0,
            last_eval: None,
            num_splats: 0,
            frames: 0,
            cur_sh_degree: 0,
            cur_dataset: Dataset::empty(),
            adapter_info,
        }
    }
}

fn bytes_format(bytes: u64) -> String {
    let unit = 1000;

    if bytes < unit {
        format!("{bytes} B")
    } else {
        let size = bytes as f64;
        let exp = match size.log(1000.0).floor() as usize {
            0 => 1,
            e => e,
        };
        let unit_prefix = b"KMGTPEZY";
        format!(
            "{:.2} {}B",
            (size / unit.pow(exp as u32) as f64),
            unit_prefix[exp - 1] as char,
        )
    }
}

impl AppPane for StatsPanel {
    fn title(&self) -> String {
        "Stats".to_owned()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default
    }

    fn on_message(&mut self, message: &ProcessMessage, _: &UiProcess) {
        match message {
            ProcessMessage::NewSource => {
                *self = Self::new(self.device.clone(), self.adapter_info.clone());
            }
            ProcessMessage::StartLoading { .. } => {
                self.train_iter_per_s = 0.0;
                self.num_splats = 0;
                self.cur_sh_degree = 0;
                self.last_eval = None;
            }
            ProcessMessage::ViewSplats {
                up_axis: _,
                splats,
                frame,
                total_frames: _,
            } => {
                self.num_splats = splats.num_splats();
                self.frames = *frame;
                self.cur_sh_degree = splats.sh_degree();
            }
            ProcessMessage::TrainStep {
                splats,
                stats: _,
                iter,
                total_elapsed,
            } => {
                self.cur_sh_degree = splats.sh_degree();
                self.num_splats = splats.num_splats();
                let current_iter_per_s = (iter - self.last_train_step.1) as f32
                    / (*total_elapsed - self.last_train_step.0).as_secs_f32();
                self.train_iter_per_s = if *iter < 16 {
                    current_iter_per_s
                } else {
                    0.95 * self.train_iter_per_s + 0.05 * current_iter_per_s
                };
                self.last_train_step = (*total_elapsed, *iter);
            }
            ProcessMessage::Dataset { dataset } => {
                self.cur_dataset = dataset.clone();
            }
            ProcessMessage::EvalResult {
                iter: _,
                avg_psnr,
                avg_ssim,
            } => {
                self.last_eval = Some(format!("{avg_psnr:.2} PSNR, {avg_ssim:.3} SSIM"));
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        ui.vertical(|ui| {
            // Model Stats
            ui.heading("Model Stats");
            ui.separator();

            let first_col_width = ui.available_width() * 0.4;
            egui::Grid::new("model_stats_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .striped(true)
                .min_col_width(first_col_width)
                .max_col_width(first_col_width)
                .show(ui, |ui| {
                    ui.label("Splats");
                    ui.label(format!("{}", self.num_splats));
                    ui.end_row();

                    ui.label("SH Degree");
                    ui.label(format!("{}", self.cur_sh_degree));
                    ui.end_row();

                    if self.frames > 0 {
                        ui.label("Frames");
                        ui.label(format!("{}", self.frames));
                        ui.end_row();
                    }
                });

            if process.is_training() {
                ui.add_space(10.0);
                ui.heading("Training Stats");
                ui.separator();

                let first_col_width = ui.available_width() * 0.4;
                egui::Grid::new("training_stats_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .striped(true)
                    .min_col_width(first_col_width)
                    .max_col_width(first_col_width)
                    .show(ui, |ui| {
                        ui.label("Train step");
                        ui.label(format!("{}", self.last_train_step.1));
                        ui.end_row();

                        ui.label("Steps/s");
                        ui.label(format!("{:.1}", self.train_iter_per_s));
                        ui.end_row();

                        ui.label("Last eval");
                        ui.label(if let Some(eval) = self.last_eval.as_ref() {
                            eval
                        } else {
                            "--"
                        });
                        ui.end_row();

                        ui.label("Training time");
                        ui.label(format!(
                            "{}",
                            humantime::format_duration(Duration::from_secs(
                                self.last_train_step.0.as_secs()
                            ))
                        ));
                        ui.end_row();

                        ui.label("Dataset train size");
                        ui.label(format!("{}", self.cur_dataset.train.views.len()));
                        ui.end_row();

                        ui.label("Dataset eval");
                        ui.label(format!(
                            "{}",
                            self.cur_dataset
                                .eval
                                .as_ref()
                                .map_or(0, |eval| eval.views.len())
                        ));
                        ui.end_row();
                    });
            }

            ui.add_space(10.0);
            ui.heading("GPU");
            ui.separator();

            let client = WgpuRuntime::client(&self.device);
            let memory = client.memory_usage();

            let first_col_width = ui.available_width() * 0.4;
            egui::Grid::new("memory_stats_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .striped(true)
                .min_col_width(first_col_width)
                .max_col_width(first_col_width)
                .show(ui, |ui| {
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
                let first_col_width = ui.available_width() * 0.4;
                egui::Grid::new("gpu_info_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .striped(true)
                    .min_col_width(first_col_width)
                    .max_col_width(first_col_width)
                    .show(ui, |ui| {
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
        });
    }
}
