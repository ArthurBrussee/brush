#[cfg(feature = "training")]
use crate::settings_popup::SettingsPopup;
use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use brush_process::message::ProcessMessage;
#[cfg(feature = "training")]
use brush_process::message::TrainMessage;
use brush_vfs::DataSource;
use egui::Align2;
#[cfg(feature = "training")]
use web_time::Duration;

pub struct SettingsPanel {
    url: String,
    show_url_dialog: bool,
    pending_source_type: Option<String>,
    current_source: Option<(String, String)>, // (name, source_type)
    #[cfg(feature = "training")]
    train_progress: Option<(u32, u32, Duration)>, // (current_iter, total_steps, elapsed)
    #[cfg(feature = "training")]
    last_train_step: (Duration, u32), // (elapsed, iter) for calculating iter/s
    #[cfg(feature = "training")]
    train_iter_per_s: f32,
    #[cfg(feature = "training")]
    popup: Option<SettingsPopup>,
}

impl SettingsPanel {
    pub(crate) fn new() -> Self {
        Self {
            url: "splat.com/example.ply".to_owned(),
            show_url_dialog: false,
            pending_source_type: None,
            current_source: None,
            #[cfg(feature = "training")]
            train_progress: None,
            #[cfg(feature = "training")]
            last_train_step: (Duration::from_secs(0), 0),
            #[cfg(feature = "training")]
            train_iter_per_s: 0.0,
            #[cfg(feature = "training")]
            popup: None,
        }
    }
}

impl AppPane for SettingsPanel {
    fn title(&self) -> String {
        "Status".to_owned()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default
    }

    fn on_message(&mut self, message: &ProcessMessage, _process: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                self.current_source = None;
                #[cfg(feature = "training")]
                {
                    self.train_progress = None;
                }
            }
            ProcessMessage::NewSource { name } => {
                let source_type = self
                    .pending_source_type
                    .take()
                    .unwrap_or_else(|| "File".to_owned());
                self.current_source = Some((name.clone(), source_type));
            }
            #[cfg(feature = "training")]
            ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                iter,
                total_steps,
                total_elapsed,
                ..
            }) => {
                self.train_progress = Some((*iter, *total_steps, *total_elapsed));

                // Calculate smoothed iter/s
                if let Some(elapsed_diff) = total_elapsed.checked_sub(self.last_train_step.0) {
                    let iter_diff = iter - self.last_train_step.1;
                    if iter_diff > 0 && elapsed_diff.as_secs_f32() > 0.0 {
                        let current_iter_per_s = iter_diff as f32 / elapsed_diff.as_secs_f32();
                        self.train_iter_per_s = if *iter < 16 {
                            current_iter_per_s
                        } else {
                            0.95 * self.train_iter_per_s + 0.05 * current_iter_per_s
                        };
                    }
                }
                self.last_train_step = (*total_elapsed, *iter);
            }
            #[cfg(feature = "training")]
            ProcessMessage::TrainMessage(TrainMessage::DoneTraining) => {
                self.train_progress = None;
            }
            _ => {}
        }
    }

    fn inner_margin(&self) -> f32 {
        6.0
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        ui.horizontal(|ui| {
            ui.set_height(32.0);
            ui.spacing_mut().item_spacing.x = 2.0;

            let button_height = 26.0;
            let button_color = egui::Color32::from_rgb(70, 130, 180);

            let mut load_option = None;

            if ui
                .add(
                    egui::Button::new(egui::RichText::new("File").size(13.0))
                        .min_size(egui::vec2(50.0, button_height))
                        .fill(button_color)
                        .stroke(egui::Stroke::NONE),
                )
                .clicked()
            {
                load_option = Some(DataSource::PickFile);
            }

            let can_pick_dir = !cfg!(target_os = "android");
            if can_pick_dir
                && ui
                    .add(
                        egui::Button::new(egui::RichText::new("Directory").size(13.0))
                            .min_size(egui::vec2(70.0, button_height))
                            .fill(button_color)
                            .stroke(egui::Stroke::NONE),
                    )
                    .clicked()
            {
                load_option = Some(DataSource::PickDirectory);
            }

            let can_url = !cfg!(target_os = "android");
            if can_url
                && ui
                    .add(
                        egui::Button::new(egui::RichText::new("URL").size(13.0))
                            .min_size(egui::vec2(45.0, button_height))
                            .fill(button_color)
                            .stroke(egui::Stroke::NONE),
                    )
                    .clicked()
            {
                self.show_url_dialog = true;
            }

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(12.0);

            // Status section - show source info or prompt
            if let Some((name, source_type)) = &self.current_source {
                ui.label(
                    egui::RichText::new(source_type)
                        .size(14.0)
                        .color(egui::Color32::from_rgb(140, 140, 140)),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(name)
                        .size(15.0)
                        .strong()
                        .color(egui::Color32::from_rgb(220, 220, 220)),
                );
            } else {
                ui.label(
                    egui::RichText::new("Load a .ply file or dataset to get started")
                        .size(14.0)
                        .color(egui::Color32::from_rgb(140, 140, 140))
                        .italics(),
                );
            }

            #[cfg(feature = "training")]
            if let Some((iter, total, elapsed)) = self.train_progress {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let progress = iter as f32 / total as f32;
                    let percent = (progress * 100.0) as u32;

                    let eta_text = if iter > 0 {
                        let elapsed_secs = elapsed.as_secs_f32();
                        let secs_per_iter = elapsed_secs / iter as f32;
                        let remaining_iters = total.saturating_sub(iter);
                        let remaining_secs = (secs_per_iter * remaining_iters as f32) as u64;
                        let remaining = Duration::from_secs(remaining_secs);
                        format!("ETA {}", humantime::format_duration(remaining))
                    } else {
                        "ETA --".to_owned()
                    };

                    let bar_response = ui.add(
                        egui::ProgressBar::new(progress)
                            .desired_width(450.0)
                            .desired_height(22.0),
                    );

                    let bar_rect = bar_response.rect;
                    let padding = 10.0;

                    ui.painter().text(
                        egui::pos2(bar_rect.left() + padding, bar_rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("{percent}%"),
                        egui::FontId::proportional(13.0),
                        egui::Color32::WHITE,
                    );

                    let iter_text = format!("{:.1} it/s", self.train_iter_per_s);
                    let dim_color = egui::Color32::from_rgb(200, 200, 200);
                    let bright_color = egui::Color32::WHITE;

                    let galley_eta = ui.painter().layout_no_wrap(
                        eta_text.clone(),
                        egui::FontId::proportional(12.0),
                        bright_color,
                    );
                    let galley_iter = ui.painter().layout_no_wrap(
                        iter_text.clone(),
                        egui::FontId::proportional(11.0),
                        dim_color,
                    );

                    let eta_width = galley_eta.size().x;
                    let iter_width = galley_iter.size().x;
                    let separator_width = 24.0;

                    ui.painter().text(
                        egui::pos2(bar_rect.right() - padding, bar_rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        eta_text,
                        egui::FontId::proportional(12.0),
                        bright_color,
                    );

                    ui.painter().text(
                        egui::pos2(
                            bar_rect.right() - padding - eta_width - separator_width / 2.0,
                            bar_rect.center().y,
                        ),
                        egui::Align2::CENTER_CENTER,
                        "-",
                        egui::FontId::proportional(11.0),
                        dim_color,
                    );

                    ui.painter().text(
                        egui::pos2(
                            bar_rect.right() - padding - eta_width - separator_width - iter_width,
                            bar_rect.center().y,
                        ),
                        egui::Align2::LEFT_CENTER,
                        iter_text,
                        egui::FontId::proportional(11.0),
                        dim_color,
                    );

                    ui.add_space(16.0);
                    ui.label(egui::RichText::new("Training").size(16.0).strong());
                });
            }

            if self.show_url_dialog {
                egui::Window::new("Load from URL")
                    .resizable(false)
                    .collapsible(false)
                    .default_pos(ui.ctx().screen_rect().center())
                    .pivot(Align2::CENTER_CENTER)
                    .show(ui.ctx(), |ui| {
                        ui.vertical(|ui| {
                            ui.label("Enter URL:");
                            ui.add_space(5.0);

                            let url_response = ui.add(
                                egui::TextEdit::singleline(&mut self.url)
                                    .desired_width(300.0)
                                    .hint_text("e.g., splat.com/example.ply"),
                            );

                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                if ui.button("Load").clicked() && !self.url.trim().is_empty() {
                                    load_option = Some(DataSource::Url(self.url.clone()));
                                    self.show_url_dialog = false;
                                }
                                if ui.button("Cancel").clicked() {
                                    self.show_url_dialog = false;
                                }
                            });

                            if url_response.lost_focus()
                                && ui.input(|i| i.key_pressed(egui::Key::Enter))
                                && !self.url.trim().is_empty()
                            {
                                load_option = Some(DataSource::Url(self.url.clone()));
                                self.show_url_dialog = false;
                            }
                        });
                    });
            }

            if let Some(source) = load_option {
                // Track the source type for display
                self.pending_source_type = Some(match &source {
                    DataSource::PickFile => "File".to_owned(),
                    DataSource::PickDirectory => "Directory".to_owned(),
                    DataSource::Url(_) => "URL".to_owned(),
                    DataSource::Path(_) => "Path".to_owned(),
                });

                let (_sender, receiver) = tokio::sync::oneshot::channel();
                #[cfg(feature = "training")]
                {
                    self.popup = Some(SettingsPopup::new(_sender));
                }

                process.start_new_process(source, receiver);
            }
        });

        // Draw settings window if we're loading something (if loading a ply
        // this wont' do anything, only if process args are needed).
        #[cfg(feature = "training")]
        if let Some(popup) = &mut self.popup
            && process.is_loading()
        {
            popup.ui(ui);

            if popup.is_done() {
                self.popup = None;
            }
        }
    }
}
