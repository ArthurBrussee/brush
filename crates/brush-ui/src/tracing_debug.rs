use crate::{BrushUiProcess, panels::AppPanel};

#[derive(Default)]
pub struct TracingPanel {
    constant_redraw: bool,
}

impl AppPanel for TracingPanel {
    fn title(&self) -> String {
        "Load data".to_owned()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &dyn BrushUiProcess) {
        ui.checkbox(&mut self.constant_redraw, "Constant redraw");

        // Nb: this redraws the whole context so this will include the splat views.
        if self.constant_redraw {
            ui.ctx().request_repaint();
        }
    }
}
