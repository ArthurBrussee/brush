#[cfg(feature = "training")]
use crate::settings_popup::SettingsPopup;
use crate::{UiMode, panels::AppPane, ui_process::UiProcess};
use brush_vfs::DataSource;
use egui::Align2;

pub struct SettingsPanel {
    url: String,
    show_url_dialog: bool,
    #[cfg(feature = "training")]
    popup: Option<SettingsPopup>,
}

impl SettingsPanel {
    pub(crate) fn new() -> Self {
        Self {
            url: "splat.com/example.ply".to_owned(),
            show_url_dialog: false,
            #[cfg(feature = "training")]
            popup: None,
        }
    }
}

impl AppPane for SettingsPanel {
    fn title(&self) -> String {
        "Settings".to_owned()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(20.0);

            // Create a nice loading options UI
            let mut load_option = None;

            ui.label(
                egui::RichText::new("Load Data:")
                    .heading()
                    .color(egui::Color32::from_rgb(70, 130, 180)),
            );
            ui.add_space(5.0);

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                if ui
                    .add(
                        egui::Button::new("File")
                            .min_size(egui::vec2(50.0, 32.0))
                            .fill(egui::Color32::from_rgb(70, 130, 180))
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
                            egui::Button::new("Directory")
                                .min_size(egui::vec2(70.0, 32.0))
                                .fill(egui::Color32::from_rgb(70, 130, 180))
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
                            egui::Button::new("URL")
                                .min_size(egui::vec2(50.0, 32.0))
                                .fill(egui::Color32::from_rgb(70, 130, 180))
                                .stroke(egui::Stroke::NONE),
                        )
                        .clicked()
                {
                    self.show_url_dialog = true;
                }
            });

            ui.add_space(15.0);

            // URL dialog window
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
