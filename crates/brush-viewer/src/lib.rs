#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]

use egui_tiles::SimplificationOptions;
use viewer::{ProcessMessage, ViewerContext};

mod orbit_controls;

mod panels;
mod train_loop;

pub mod viewer;

trait ViewerPanel {
    fn title(&self) -> String;
    fn ui(&mut self, ui: &mut egui::Ui, controls: &mut ViewerContext);
    fn on_message(&mut self, message: &ProcessMessage, context: &mut ViewerContext) {
        let _ = message;
        let _ = context;
    }
}

struct ViewerTree {
    zen: bool,
    context: ViewerContext,
}

type PaneType = Box<dyn ViewerPanel>;

impl egui_tiles::Behavior<PaneType> for ViewerTree {
    fn tab_title_for_pane(&mut self, pane: &PaneType) -> egui::WidgetText {
        pane.title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut PaneType,
    ) -> egui_tiles::UiResponse {
        pane.ui(ui, &mut self.context);
        egui_tiles::UiResponse::None
    }

    /// What are the rules for simplifying the tree?
    fn simplification_options(&self) -> SimplificationOptions {
        SimplificationOptions {
            all_panes_must_have_tabs: !self.zen,
            ..Default::default()
        }
    }
}
