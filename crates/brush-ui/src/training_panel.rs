use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use anyhow::Error;
use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_render::{MainBackend, gaussian_splats::Splats};
use egui::RichText;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use web_time::Duration;

pub struct TrainingPanel {
    train_progress: Option<(u32, u32)>,
    last_train_step: Option<(Duration, u32)>,
    train_iter_per_s: f32,
    iter_per_s_samples: u32,
    train_config: Option<TrainStreamConfig>,
    manual_export_iters: Vec<u32>,
    current_splats: Option<Splats<MainBackend>>,
    export_channel: (UnboundedSender<Error>, UnboundedReceiver<Error>),
}

impl Default for TrainingPanel {
    fn default() -> Self {
        Self {
            train_progress: None,
            last_train_step: None,
            train_iter_per_s: 0.0,
            iter_per_s_samples: 0,
            train_config: None,
            manual_export_iters: Vec::new(),
            current_splats: None,
            export_channel: tokio::sync::mpsc::unbounded_channel(),
        }
    }
}

impl TrainingPanel {
    fn reset(&mut self) {
        self.train_progress = None;
        self.last_train_step = None;
        self.train_iter_per_s = 0.0;
        self.iter_per_s_samples = 0;
        self.train_config = None;
        self.manual_export_iters.clear();
        self.current_splats = None;
    }

    fn on_train_message(&mut self, message: &TrainMessage) {
        match message {
            TrainMessage::TrainConfig { config } => {
                self.train_config = Some(*config.clone());
            }
            TrainMessage::TrainStep {
                iter,
                total_steps,
                total_elapsed,
                splats,
                ..
            } => {
                self.train_progress = Some((*iter, *total_steps));
                self.current_splats = Some(splats.as_ref().clone());

                if let Some((last_elapsed, last_iter)) = self.last_train_step
                    && let Some(elapsed_diff) = total_elapsed.checked_sub(last_elapsed)
                {
                    let iter_diff = iter - last_iter;
                    if iter_diff > 0 && elapsed_diff.as_secs_f32() > 0.0 {
                        let current_iter_per_s = iter_diff as f32 / elapsed_diff.as_secs_f32();
                        let smoothing = (self.iter_per_s_samples as f32 / 20.0).min(1.0) * 0.95;
                        self.train_iter_per_s = smoothing * self.train_iter_per_s
                            + (1.0 - smoothing) * current_iter_per_s;
                        self.iter_per_s_samples += 1;
                    }
                }
                self.last_train_step = Some((*total_elapsed, *iter));
            }
            TrainMessage::DoneTraining => {
                if let Some((_, total)) = self.train_progress {
                    self.train_progress = Some((total, total));
                }
            }
            _ => {}
        }
    }
}

async fn export(splat: Splats<MainBackend>) -> Result<(), Error> {
    let data = brush_serde::splat_to_ply(splat).await?;
    rrfd::save_file("export.ply", data).await?;
    Ok(())
}

const PIN_STEM: f32 = 5.0;
const PIN_RADIUS: f32 = 3.5;
const PIN_HOVER_RADIUS: f32 = 4.5;

fn draw_pin(
    ui: &egui::Ui,
    x: f32,
    row_top: f32,
    color: egui::Color32,
    filled: bool,
    tooltip: &str,
) {
    let pin_total_height = PIN_STEM + PIN_RADIUS * 2.0;
    let hit_rect = egui::Rect::from_min_max(
        egui::pos2(x - 6.0, row_top),
        egui::pos2(x + 6.0, row_top + pin_total_height + 2.0),
    );
    let response = ui.interact(hit_rect, ui.id().with(tooltip), egui::Sense::hover());
    let radius = if response.hovered() {
        PIN_HOVER_RADIUS
    } else {
        PIN_RADIUS
    };

    let stem_bottom = row_top + PIN_STEM;
    ui.painter().line_segment(
        [egui::pos2(x, row_top), egui::pos2(x, stem_bottom)],
        egui::Stroke::new(1.5, color),
    );

    let circle_center = egui::pos2(x, stem_bottom + radius);
    ui.painter()
        .circle_stroke(circle_center, radius, egui::Stroke::new(1.5, color));
    if filled {
        ui.painter()
            .circle_filled(circle_center, radius * 0.5, color);
    }

    response.on_hover_text(tooltip);
}

impl AppPane for TrainingPanel {
    fn title(&self) -> egui::WidgetText {
        "Training".into()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default && process.is_training()
    }

    fn on_message(&mut self, message: &ProcessMessage, _process: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                self.reset();
            }
            ProcessMessage::TrainMessage(msg) => {
                self.on_train_message(msg);
            }
            _ => {}
        }
    }

    fn inner_margin(&self) -> f32 {
        6.0
    }

    fn top_bar_right_ui(&mut self, ui: &mut egui::Ui, _process: &UiProcess) {
        let text_color = ui.visuals().strong_text_color();

        // Show iter/s and ETA
        if self.train_iter_per_s > 0.0
            && let Some((iter, total)) = self.train_progress
        {
            let remaining_iters = total.saturating_sub(iter);
            let remaining_secs = (remaining_iters as f32 / self.train_iter_per_s) as u64;
            let remaining = Duration::from_secs(remaining_secs);

            ui.label(
                RichText::new(format!(
                    "{:.0} it/s  ETA {}",
                    self.train_iter_per_s,
                    humantime::format_duration(remaining)
                ))
                .size(12.0)
                .color(text_color),
            );
            ui.add_space(8.0);
        }

        // Show training elapsed time
        if let Some((elapsed, _)) = self.last_train_step {
            // Truncate to seconds for human-friendly display
            let elapsed_secs = Duration::from_secs(elapsed.as_secs());
            ui.label(
                RichText::new(format!(
                    "{} elapsed",
                    humantime::format_duration(elapsed_secs)
                ))
                .size(12.0)
                .color(text_color),
            );
            ui.add_space(4.0);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        // Show progress bar as soon as settings are available, even before first train step
        let (iter, total) = if let Some((iter, total)) = self.train_progress {
            (iter, total)
        } else if let Some(config) = &self.train_config {
            // We have settings but no train steps yet - show 0% progress
            (0, config.train_config.total_steps)
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("Waiting for training to start")
                        .size(14.0)
                        .color(egui::Color32::from_rgb(140, 140, 140))
                        .italics(),
                );
            });
            return;
        };

        let progress = iter as f32 / total as f32;
        let is_complete = iter == total;
        let padding = 8.0;

        let bar_rect = ui
            .horizontal(|ui| {
                if !is_complete {
                    let paused = process.is_train_paused();
                    let icon = if paused { "⏵" } else { "⏸" };
                    let btn_color = if paused {
                        egui::Color32::from_rgb(70, 70, 75)
                    } else {
                        egui::Color32::from_rgb(70, 130, 180)
                    };

                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(icon).size(14.0).color(egui::Color32::WHITE),
                            )
                            .min_size(egui::vec2(28.0, 20.0))
                            .corner_radius(6.0)
                            .fill(btn_color),
                        )
                        .on_hover_text(if paused {
                            "Resume training"
                        } else {
                            "Pause training"
                        })
                        .clicked()
                    {
                        process.set_train_paused(!paused);
                    }

                    ui.add_space(6.0);
                }

                let export_button_width = if self.current_splats.is_some() {
                    65.0
                } else {
                    0.0
                };
                let progress_width = ui.available_width()
                    - export_button_width
                    - if export_button_width > 0.0 { 6.0 } else { 0.0 };

                let bar_response = ui.add(
                    egui::ProgressBar::new(progress)
                        .desired_width(progress_width)
                        .desired_height(20.0)
                        .fill(if is_complete {
                            egui::Color32::from_rgb(100, 200, 100)
                        } else {
                            ui.visuals().selection.bg_fill
                        }),
                );

                let bar_rect = bar_response.rect;

                if let Some(splats) = self.current_splats.clone() {
                    ui.add_space(6.0);
                    // Make export button more prominent when training is complete
                    let (button_text, button_color) = if is_complete {
                        ("Export", egui::Color32::from_rgb(60, 160, 60))
                    } else {
                        ("Export", egui::Color32::from_rgb(80, 140, 80))
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(button_text)
                                    .size(12.0)
                                    .color(egui::Color32::WHITE),
                            )
                            .min_size(egui::vec2(55.0, 20.0))
                            .corner_radius(6.0)
                            .fill(button_color),
                        )
                        .on_hover_text(if is_complete {
                            "Export trained model"
                        } else {
                            "Export current model"
                        })
                        .clicked()
                    {
                        if !is_complete {
                            self.manual_export_iters.push(iter);
                        }
                        let sender = self.export_channel.0.clone();
                        let ctx = ui.ctx().clone();
                        tokio_with_wasm::alias::task::spawn(async move {
                            if let Err(e) = export(splats).await {
                                let _ = sender.send(e);
                                ctx.request_repaint();
                            }
                        });
                    }
                }

                bar_rect
            })
            .inner;

        // Draw export pins on the progress bar
        if let Some(config) = &self.train_config {
            let export_every = config.process_config.export_every;
            let export_color = egui::Color32::from_rgb(100, 150, 255);
            let manual_export_color = egui::Color32::from_rgb(100, 200, 100);
            let next_export = ((iter / export_every) + 1) * export_every;
            let row_top = bar_rect.bottom() - 3.0;

            let mut export_iter = export_every;
            while export_iter <= total {
                let x = bar_rect.left() + (export_iter as f32 / total as f32) * bar_rect.width();
                let completed = iter >= export_iter;
                let is_next = export_iter == next_export;
                let alpha = if completed || is_next { 1.0 } else { 0.4 };

                draw_pin(
                    ui,
                    x,
                    row_top,
                    export_color.gamma_multiply(alpha),
                    completed,
                    &format!("Auto-save at iteration {export_iter}"),
                );
                export_iter += export_every;
            }

            for &manual_iter in &self.manual_export_iters {
                let x = bar_rect.left() + (manual_iter as f32 / total as f32) * bar_rect.width();
                draw_pin(
                    ui,
                    x,
                    row_top,
                    manual_export_color,
                    true,
                    &format!("Manual save at iteration {manual_iter}"),
                );
            }
        }

        // Progress text overlay - right aligned
        let text_color = egui::Color32::WHITE;

        if is_complete {
            ui.painter().text(
                egui::pos2(bar_rect.right() - padding, bar_rect.center().y),
                egui::Align2::RIGHT_CENTER,
                "Complete!",
                egui::FontId::new(13.0, egui::FontFamily::Proportional),
                egui::Color32::WHITE,
            );
        } else {
            ui.painter().text(
                egui::pos2(bar_rect.right() - padding, bar_rect.center().y),
                egui::Align2::RIGHT_CENTER,
                format!("{iter}/{total}"),
                egui::FontId::new(12.0, egui::FontFamily::Proportional),
                text_color,
            );
        }
    }
}
