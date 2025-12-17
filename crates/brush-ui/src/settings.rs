#[cfg(feature = "training")]
use crate::training_ui::TrainingState;
use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use brush_process::message::ProcessMessage;
use brush_vfs::DataSource;
use egui::Align2;

pub struct SettingsPanel {
    url: String,
    show_url_dialog: bool,
    current_source: Option<(String, DataSource)>,
    #[cfg(feature = "training")]
    training: TrainingState,
}

impl Default for SettingsPanel {
    fn default() -> Self {
        Self {
            url: "splat.com/example.ply".to_owned(),
            show_url_dialog: false,
            current_source: None,
            #[cfg(feature = "training")]
            training: TrainingState::new(),
        }
    }
}

impl AppPane for SettingsPanel {
    fn title(&self) -> egui::WidgetText {
        "Status".into()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default
    }

    fn on_message(&mut self, message: &ProcessMessage, _process: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                self.current_source = None;
                #[cfg(feature = "training")]
                self.training.reset();
            }
            ProcessMessage::NewSource { name, source } => {
                self.current_source = Some((name.clone(), source.clone()));
            }
            #[cfg(feature = "training")]
            ProcessMessage::TrainMessage(msg) => {
                self.training.on_train_message(msg);
            }
            _ => {}
        }
    }

    fn inner_margin(&self) -> f32 {
        3.0
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

            // Status section - show prompt when nothing loaded
            if self.current_source.is_none() {
                ui.label(
                    egui::RichText::new("Load a .ply file or dataset to get started")
                        .size(14.0)
                        .color(egui::Color32::from_rgb(140, 140, 140))
                        .italics(),
                );
            }

            #[cfg(feature = "training")]
            crate::training_ui::draw_training_progress(ui, &mut self.training, process);

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
                let (_sender, receiver) = tokio::sync::oneshot::channel();
                #[cfg(feature = "training")]
                {
                    self.training.popup = Some(crate::training_ui::SettingsPopup::new(_sender));
                }
                process.start_new_process(source, receiver);
            }
        });

        #[cfg(feature = "training")]
        crate::training_ui::draw_settings_popup(ui, &mut self.training, process);
    }
}
