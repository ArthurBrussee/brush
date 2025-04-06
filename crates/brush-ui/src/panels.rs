use brush_msg::ProcessMessage;

use crate::BrushUiProcess;

pub type PaneType = Box<dyn AppPanel>;

pub(crate) trait AppPanel {
    fn title(&self) -> String;

    /// Draw the pane's UI's content/
    fn ui(&mut self, ui: &mut egui::Ui, process: &mut dyn BrushUiProcess);

    /// Handle an incoming message from the UI.
    fn on_message(&mut self, message: &ProcessMessage, process: &mut dyn BrushUiProcess) {
        let _ = message;
        let _ = process;
    }

    fn on_error(&mut self, error: &anyhow::Error, process: &mut dyn BrushUiProcess) {
        let _ = error;
        let _ = process;
    }

    /// Override the inner margin for this panel.
    fn inner_margin(&self) -> f32 {
        12.0
    }
}
