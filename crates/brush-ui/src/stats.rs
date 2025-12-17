use std::sync::Arc;

use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use eframe::egui_wgpu::Renderer;
use egui::mutex::RwLock;
use web_time::Duration;
use wgpu::AdapterInfo;

#[derive(Default)]
pub struct StatsPanel {
    device: Option<WgpuDevice>,
    last_eval: Option<String>,
    cur_sh_degree: u32,
    num_splats: u32,
    frames: u32,
    adapter_info: Option<AdapterInfo>,
    last_train_step: (Duration, u32),
    train_eval_views: (u32, u32),
    training_complete: bool,
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
    fn title(&self) -> egui::WidgetText {
        "Stats".into()
    }

    fn init(
        &mut self,
        _device: wgpu::Device,
        _queue: wgpu::Queue,
        _renderer: Arc<RwLock<Renderer>>,
        burn_device: burn_wgpu::WgpuDevice,
        adapter_info: wgpu::AdapterInfo,
    ) {
        self.device = Some(burn_device);
        self.adapter_info = Some(adapter_info);
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default && process.is_training()
    }

    fn on_message(&mut self, message: &ProcessMessage, _: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                self.last_eval = None;
                self.cur_sh_degree = 0;
                self.num_splats = 0;
                self.frames = 0;
                self.last_train_step = (Duration::from_secs(0), 0);
                self.train_eval_views = (0, 0);
                self.training_complete = false;
            }
            ProcessMessage::StartLoading { .. } => {
                self.num_splats = 0;
                self.cur_sh_degree = 0;
                self.last_eval = None;
            }
            ProcessMessage::ViewSplats { splats, frame, .. } => {
                self.num_splats = splats.num_splats();
                self.frames = *frame;
                self.cur_sh_degree = splats.sh_degree();
            }
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainStep {
                    splats,
                    iter,
                    total_elapsed,
                    ..
                } => {
                    self.cur_sh_degree = splats.sh_degree();
                    self.num_splats = splats.num_splats();
                    self.last_train_step = (*total_elapsed, *iter);
                }
                TrainMessage::Dataset { dataset } => {
                    self.train_eval_views = (
                        dataset.train.views.len() as u32,
                        dataset
                            .eval
                            .as_ref()
                            .map_or(0, |eval| eval.views.len() as u32),
                    );
                }
                TrainMessage::EvalResult {
                    iter: _,
                    avg_psnr,
                    avg_ssim,
                } => {
                    self.last_eval = Some(format!("{avg_psnr:.2} PSNR, {avg_ssim:.3} SSIM"));
                }
                TrainMessage::DoneTraining => {
                    self.training_complete = true;
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        ui.vertical(|ui| {
            let _ = process;

            // Model Stats
            ui.heading(if self.training_complete {
                "Final Model Stats"
            } else {
                "Model Stats"
            });
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

                        ui.label("Dataset views");
                        ui.label(format!("{}", self.train_eval_views.0));
                        ui.end_row();

                        ui.label("Dataset eval views");
                        ui.label(format!("{}", self.train_eval_views.1));
                        ui.end_row();
                    });
            }

            if let Some(device) = &self.device {
                ui.add_space(10.0);
                ui.heading("GPU");
                ui.separator();

                let client = WgpuRuntime::client(device);
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
                if !cfg!(target_family = "wasm")
                    && let Some(adapter_info) = &self.adapter_info
                {
                    let first_col_width = ui.available_width() * 0.4;
                    egui::Grid::new("gpu_info_grid")
                        .num_columns(2)
                        .spacing([20.0, 4.0])
                        .striped(true)
                        .min_col_width(first_col_width)
                        .max_col_width(first_col_width)
                        .show(ui, |ui| {
                            ui.label("Name");
                            ui.label(&adapter_info.name);
                            ui.end_row();

                            ui.label("Type");
                            ui.label(format!("{:?}", adapter_info.device_type));
                            ui.end_row();

                            ui.label("Driver");
                            ui.label(format!(
                                "{}, {}",
                                adapter_info.driver, adapter_info.driver_info
                            ));
                            ui.end_row();
                        });
                }
            }
        });
    }
}
