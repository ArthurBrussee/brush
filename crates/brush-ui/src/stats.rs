use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use eframe::egui_wgpu::RenderState;
use web_time::Duration;
use wgpu::AdapterInfo;

#[derive(Default)]
pub struct StatsPanel {
    device: Option<WgpuDevice>,
    last_eval: Option<String>,
    frames: u32,
    adapter_info: Option<AdapterInfo>,
    last_train_step: (Duration, u32),
    train_eval_views: (u32, u32),
    training_complete: bool,
    num_splats: u32,
    sh_degree: u32,
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

/// Helper to display a stat row - vertical stacks label above value, horizontal shows side-by-side
fn stat_row(ui: &mut egui::Ui, label: &str, value: impl Into<String>, vertical: bool) {
    if vertical {
        ui.label(label);
        ui.end_row();
        ui.strong(value.into());
        ui.end_row();
    } else {
        ui.label(label);
        ui.label(value.into());
        ui.end_row();
    }
}

/// Creates a stats grid with responsive layout
fn stats_grid(ui: &mut egui::Ui, id: &str, add_contents: impl FnOnce(&mut egui::Ui, bool)) {
    let use_vertical = ui.available_width() < 200.0;
    let first_col_width = ui.available_width() * 0.4;

    let mut grid = egui::Grid::new(id)
        .num_columns(if use_vertical { 1 } else { 2 })
        .spacing([20.0, 4.0]);

    if !use_vertical {
        grid = grid
            .striped(true)
            .min_col_width(first_col_width)
            .max_col_width(first_col_width);
    }

    grid.show(ui, |ui| add_contents(ui, use_vertical));
}

impl AppPane for StatsPanel {
    fn title(&self) -> egui::WidgetText {
        "Stats".into()
    }

    fn init(&mut self, state: &RenderState, burn_device: burn_wgpu::WgpuDevice) {
        self.device = Some(burn_device);
        self.adapter_info = Some(state.adapter.get_info());
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default && process.is_training()
    }

    fn on_message(&mut self, message: &ProcessMessage, _: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                self.last_eval = None;
                self.frames = 0;
                self.last_train_step = (Duration::from_secs(0), 0);
                self.train_eval_views = (0, 0);
                self.training_complete = false;
                self.num_splats = 0;
                self.sh_degree = 0;
            }
            ProcessMessage::StartLoading { .. } => {
                self.last_eval = None;
            }
            ProcessMessage::SplatsUpdated {
                num_splats,
                sh_degree,
                ..
            } => {
                self.num_splats = *num_splats;
                self.sh_degree = *sh_degree;
            }
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    ..
                } => {
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
        egui::ScrollArea::vertical().show(ui, |ui| {
            // Model Stats
            ui.heading(if self.training_complete {
                "Final Model Stats"
            } else {
                "Model Stats"
            });
            ui.separator();

            let num_splats = self.num_splats;
            let sh_degree = self.sh_degree;
            let frames = self.frames;
            stats_grid(ui, "model_stats_grid", |ui, v| {
                stat_row(ui, "Splats", format!("{num_splats}"), v);
                stat_row(ui, "SH Degree", format!("{sh_degree}"), v);
                if frames > 0 {
                    stat_row(ui, "Frames", format!("{frames}"), v);
                }
            });

            if process.is_training() {
                ui.add_space(10.0);
                ui.heading("Training Stats");
                ui.separator();

                let last_eval = self.last_eval.clone().unwrap_or_else(|| "--".to_owned());
                let training_time = format!(
                    "{}",
                    humantime::format_duration(Duration::from_secs(
                        self.last_train_step.0.as_secs()
                    ))
                );
                let train_step = self.last_train_step.1;
                let (train_views, eval_views) = self.train_eval_views;

                stats_grid(ui, "training_stats_grid", |ui, v| {
                    stat_row(ui, "Train step", format!("{train_step}"), v);
                    stat_row(ui, "Last eval", last_eval, v);
                    stat_row(ui, "Training time", training_time, v);
                    stat_row(ui, "Dataset views", format!("{train_views}"), v);
                    stat_row(ui, "Dataset eval views", format!("{eval_views}"), v);
                });
            }

            if let Some(device) = &self.device {
                ui.add_space(10.0);
                ui.heading("GPU");
                ui.separator();

                let client = WgpuRuntime::client(device);
                let memory = client.memory_usage();

                stats_grid(ui, "memory_stats_grid", |ui, v| {
                    stat_row(ui, "Bytes in use", bytes_format(memory.bytes_in_use), v);
                    stat_row(ui, "Bytes reserved", bytes_format(memory.bytes_reserved), v);
                    stat_row(
                        ui,
                        "Active allocations",
                        format!("{}", memory.number_allocs),
                        v,
                    );
                });

                // On WASM, adapter info is mostly private, not worth showing.
                if !cfg!(target_family = "wasm")
                    && let Some(adapter_info) = &self.adapter_info
                {
                    stats_grid(ui, "gpu_info_grid", |ui, v| {
                        stat_row(ui, "Name", &adapter_info.name, v);
                        stat_row(ui, "Type", format!("{:?}", adapter_info.device_type), v);
                        stat_row(
                            ui,
                            "Driver",
                            format!("{}, {}", adapter_info.driver, adapter_info.driver_info),
                            v,
                        );
                    });
                }
            }
        });
    }
}
