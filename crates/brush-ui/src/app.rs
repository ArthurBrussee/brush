use crate::UiMode;
use crate::panels::AppPane;
use crate::ui_process::UiProcess;
use crate::{
    camera_controls::CameraClamping, datasets::DatasetPanel, scene::ScenePanel,
    settings::SettingsPanel, stats::StatsPanel,
};
use eframe::egui;
use egui::ThemePreference;
use egui_tiles::{SimplificationOptions, Tile, TileId, Tiles};
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::trace_span;

pub(crate) struct AppTree {
    process: Arc<UiProcess>,
}

type PaneType = Box<dyn AppPane>;

impl egui_tiles::Behavior<PaneType> for AppTree {
    fn tab_title_for_pane(&mut self, pane: &PaneType) -> egui::WidgetText {
        pane.title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: TileId,
        pane: &mut PaneType,
    ) -> egui_tiles::UiResponse {
        egui::Frame::new()
            .inner_margin(pane.inner_margin())
            .show(ui, |ui| {
                pane.ui(ui, self.process.as_ref());
            });
        egui_tiles::UiResponse::None
    }

    /// What are the rules for simplifying the tree?
    fn simplification_options(&self) -> SimplificationOptions {
        let all_panes_must_have_tabs = match self.process.ui_mode() {
            UiMode::Default => true,
            UiMode::FullScreenSplat | UiMode::EmbeddedViewer => false,
        };

        SimplificationOptions {
            all_panes_must_have_tabs,
            ..Default::default()
        }
    }

    /// Width of the gap between tiles in a horizontal or vertical layout,
    /// and between rows/columns in a grid layout.
    fn gap_width(&self, _style: &egui::Style) -> f32 {
        match self.process.ui_mode() {
            UiMode::Default => 0.5,
            UiMode::FullScreenSplat => 0.0,
            UiMode::EmbeddedViewer => 0.0,
        }
    }
}

#[derive(Clone, PartialEq, Default)]
pub struct CameraSettings {
    pub speed_scale: Option<f32>,
    pub splat_scale: Option<f32>,
    pub background: Option<Vec3>,
    pub grid_enabled: Option<bool>,
    pub clamping: CameraClamping,
}

pub struct App {
    tree: egui_tiles::Tree<PaneType>,
    tree_ctx: AppTree,
}

impl App {
    pub fn new(cc: &eframe::CreationContext, context: Arc<UiProcess>) -> Self {
        // For now just assume we're running on the default
        let state = cc
            .wgpu_render_state
            .as_ref()
            .expect("Must use wgpu to render UI.");

        // Initialize Burn on the existing device.
        let device = brush_render::burn_init_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        // Inform the context of the connection.
        log::info!("Connecting context to Burn device & GUI context.");
        context.connect_device(device.clone(), cc.egui_ctx.clone());

        // Brush is always in dark mode for now, as it looks better and I don't care much to
        // put in the work to support both light and dark mode!
        cc.egui_ctx
            .options_mut(|opt| opt.theme_preference = ThemePreference::Dark);

        let mut tiles: Tiles<PaneType> = Tiles::default();
        let settings_pane = tiles.insert_pane(Box::new(SettingsPanel::new()));
        let stats_pane =
            tiles.insert_pane(Box::new(StatsPanel::new(device, state.adapter.get_info())));
        let side_pane = tiles.insert_vertical_tile(vec![settings_pane, stats_pane]);

        let scene_pane = tiles.insert_pane(Box::new(ScenePanel::new(
            state.device.clone(),
            state.queue.clone(),
            state.renderer.clone(),
        )));
        let dataset_pane = tiles.insert_pane(Box::new(DatasetPanel::new()));

        let right_side = tiles.insert_container(egui_tiles::Linear::new(
            egui_tiles::LinearDir::Vertical,
            vec![scene_pane, dataset_pane],
        ));

        let mut lin = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![side_pane, right_side],
        );
        lin.shares.set_share(side_pane, 0.35);
        let root_container = tiles.insert_container(lin);

        let tree = egui_tiles::Tree::new("brush_tree", root_container, tiles);
        let tree_ctx = AppTree { process: context };

        Self { tree, tree_ctx }
    }

    fn receive_messages(&mut self) {
        let _span = trace_span!("Receive Messages").entered();

        let mut messages = vec![];
        while let Some(message) = self.tree_ctx.process.try_recv_message() {
            messages.push(message);
        }

        for message in messages {
            match message {
                Ok(message) => {
                    for (_, pane) in self.tree.tiles.iter_mut() {
                        match pane {
                            Tile::Pane(pane) => {
                                pane.on_message(&message, self.tree_ctx.process.as_ref());
                            }
                            Tile::Container(_) => {}
                        }
                    }
                }
                Err(e) => {
                    for (_, pane) in self.tree.tiles.iter_mut() {
                        match pane {
                            Tile::Pane(pane) => {
                                pane.on_error(&e, self.tree_ctx.process.as_ref());
                            }
                            Tile::Container(_) => {}
                        }
                    }
                }
            };
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let _span = trace_span!("Update UI").entered();

        self.receive_messages();

        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            let current_mode = self.tree_ctx.process.ui_mode();
            let new_mode = match current_mode {
                UiMode::Default => UiMode::FullScreenSplat,
                UiMode::FullScreenSplat => UiMode::Default,
                // Don't allow exiting this mode.
                UiMode::EmbeddedViewer => UiMode::EmbeddedViewer,
            };
            self.tree_ctx.process.set_ui_mode(new_mode);
        }

        let process = self.tree_ctx.process.clone();

        // Recursive function to compute visibility bottom-up
        let mut tile_visibility = HashMap::new();

        {
            let _span = trace_span!("Compute UI visibility").entered();

            fn compute_visibility(
                tile_id: TileId,
                tiles: &Tiles<PaneType>,
                process: &UiProcess,
                memo: &mut HashMap<TileId, bool>,
            ) -> bool {
                if let Some(&cached) = memo.get(&tile_id) {
                    return cached;
                }

                let c = tiles.get(tile_id).expect("Must come from valid parent");
                let visible = match c {
                    Tile::Pane(pane) => pane.is_visible(process),
                    Tile::Container(container) => {
                        // Container is visible if any child is visible
                        container
                            .active_children()
                            .any(|&child_id| compute_visibility(child_id, tiles, process, memo))
                    }
                };
                memo.insert(tile_id, visible);
                visible
            }

            // Compute visibility for all tiles
            for tile_id in self.tree.tiles.tile_ids().collect::<Vec<_>>() {
                self.tree.set_visible(
                    tile_id,
                    compute_visibility(tile_id, &self.tree.tiles, &process, &mut tile_visibility),
                );
            }
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(ctx.style().as_ref()).inner_margin(0.0))
            .show(ctx, |ui| {
                let _span = trace_span!("Render UI").entered();
                self.tree.ui(&mut self.tree_ctx, ui);
            });
    }
}
