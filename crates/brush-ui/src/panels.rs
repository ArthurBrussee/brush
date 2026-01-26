use brush_process::message::ProcessMessage;
use burn_wgpu::WgpuDevice;
use eframe::egui_wgpu::RenderState;

use crate::ui_process::UiProcess;

pub(crate) trait AppPane {
    fn title(&self) -> egui::WidgetText;

    /// Initialize runtime state after creation or deserialization.
    #[allow(unused_variables)]
    fn init(&mut self, state: &RenderState, burn_device: WgpuDevice) {}

    /// Draw the pane's UI's content.
    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess);

    /// Handle an incoming message from the UI.
    fn on_message(&mut self, message: &ProcessMessage, process: &UiProcess) {
        let _ = message;
        let _ = process;
    }

    /// Handle an incoming error from the UI.
    fn on_error(&mut self, error: &anyhow::Error, process: &UiProcess) {
        let _ = error;
        let _ = process;
    }

    /// Whether this pane is visible.
    fn is_visible(&self, process: &UiProcess) -> bool {
        let _ = process;
        true
    }

    /// Override the inner margin for this panel.
    fn inner_margin(&self) -> f32 {
        12.0
    }

    /// Optional UI to add to the right side of the tab bar.
    fn top_bar_right_ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        let _ = ui;
        let _ = process;
    }
}
