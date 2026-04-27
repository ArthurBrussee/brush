use std::path::PathBuf;

use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};

use crate::ui::UiMode;
use crate::ui::panels::AppPane;
use crate::ui::settings_popup::draw_settings;
use crate::ui::ui_process::UiProcess;

#[derive(Default)]
pub struct SettingsPanel {
    config: Option<TrainStreamConfig>,
    base_path: Option<PathBuf>,
    save_status: Option<(String, web_time::Instant)>,
}

impl AppPane for SettingsPanel {
    fn title(&self) -> egui::WidgetText {
        "Settings".into()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default && process.is_training()
    }

    fn on_message(&mut self, message: &ProcessMessage, _process: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                self.config = None;
                self.base_path = None;
                self.save_status = None;
            }
            ProcessMessage::StartLoading { base_path, .. } => {
                self.base_path = base_path.clone();
            }
            ProcessMessage::TrainMessage(TrainMessage::TrainConfig { config }) => {
                self.config = Some(*config.clone());
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _process: &UiProcess) {
        // Show save confirmation popup
        if let Some((msg, time)) = &self.save_status
            && time.elapsed().as_secs() < 2
        {
            ui.label(egui::RichText::new(msg).size(12.0).italics());
            ui.add_space(4.0);
        }

        let Some(config) = &mut self.config else {
            ui.label("Waiting for training to start...");
            return;
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Fill available panel width instead of using the default 100px slider width.
            ui.spacing_mut().slider_width = (ui.available_width() - 60.0).max(60.0);
            draw_settings(ui, config, false);
        });
    }

    fn top_bar_right_ui(&mut self, ui: &mut egui::Ui, _process: &UiProcess) {
        if cfg!(target_family = "wasm") {
            return;
        }

        if self.config.is_none() {
            return;
        }

        if let Some(save_dir) = &self.base_path
            && let Some(config) = &self.config
            && ui.small_button("Save").clicked()
        {
            let args_path = save_dir.join("args.txt");
            let args = brush_process::args_file::config_to_args(config);

            match std::fs::write(&args_path, args.join(" ")) {
                Ok(()) => {
                    self.save_status = Some((
                        format!("Saved to {}", args_path.display()),
                        web_time::Instant::now(),
                    ));
                }
                Err(e) => {
                    self.save_status = Some((format!("Failed: {e}"), web_time::Instant::now()));
                }
            }
        }
    }
}
