use crate::UiMode;
#[cfg(feature = "training")]
use crate::datasets::DatasetPanel;
use crate::panels::AppPane;
use crate::settings::SettingsPanel;
#[cfg(feature = "training")]
use crate::stats::StatsPanel;
use crate::ui_process::UiProcess;
use crate::{camera_controls::CameraClamping, scene::ScenePanel};
use eframe::egui;
use egui::ThemePreference;
use egui_tiles::{SimplificationOptions, TileId, Tiles};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::trace_span;

/// Pane enum that wraps all panel types for serialization.
#[derive(Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Pane {
    Settings(#[serde(skip)] SettingsPanel),
    Scene(#[serde(skip)] Box<ScenePanel>),
    #[cfg(feature = "training")]
    Stats(#[serde(skip)] StatsPanel),
    #[cfg(feature = "training")]
    Dataset(#[serde(skip)] DatasetPanel),
}

impl Pane {
    fn as_pane(&self) -> &dyn AppPane {
        match self {
            Self::Settings(p) => p,
            Self::Scene(p) => p.as_ref(),
            #[cfg(feature = "training")]
            Self::Stats(p) => p,
            #[cfg(feature = "training")]
            Self::Dataset(p) => p,
        }
    }

    fn as_pane_mut(&mut self) -> &mut dyn AppPane {
        match self {
            Self::Settings(p) => p,
            Self::Scene(p) => p.as_mut(),
            #[cfg(feature = "training")]
            Self::Stats(p) => p,
            #[cfg(feature = "training")]
            Self::Dataset(p) => p,
        }
    }
}

pub(crate) struct AppTree {
    process: Arc<UiProcess>,
}

impl egui_tiles::Behavior<Pane> for AppTree {
    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        pane.as_pane().title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        let p = pane.as_pane_mut();
        egui::Frame::new()
            .inner_margin(p.inner_margin())
            .show(ui, |ui| p.ui(ui, self.process.as_ref()));
        egui_tiles::UiResponse::None
    }

    fn simplification_options(&self) -> SimplificationOptions {
        SimplificationOptions {
            all_panes_must_have_tabs: self.process.ui_mode() == UiMode::Default,
            ..Default::default()
        }
    }

    fn gap_width(&self, _style: &egui::Style) -> f32 {
        if self.process.ui_mode() == UiMode::Default {
            1.0
        } else {
            0.0
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

const TREE_STORAGE_KEY: &str = "brush_tile_tree";

pub struct App {
    tree: egui_tiles::Tree<Pane>,
    tree_ctx: AppTree,
}

impl App {
    pub fn new(cc: &eframe::CreationContext, context: Arc<UiProcess>) -> Self {
        let state = cc
            .wgpu_render_state
            .as_ref()
            .expect("Must use wgpu to render UI.");

        let burn_device = brush_render::burn_init_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        log::info!("Connecting context to Burn device & GUI context.");
        context.connect_device(burn_device.clone(), cc.egui_ctx.clone());

        cc.egui_ctx
            .options_mut(|opt| opt.theme_preference = ThemePreference::Dark);

        // Try to restore saved tree, or create default
        let mut tree = cc
            .storage
            .and_then(|s| eframe::get_value::<egui_tiles::Tree<Pane>>(s, TREE_STORAGE_KEY))
            .unwrap_or_else(Self::create_default_tree);

        // Initialize all panels with runtime state
        for (_, tile) in tree.tiles.iter_mut() {
            if let egui_tiles::Tile::Pane(pane) = tile {
                pane.as_pane_mut().init(
                    state.device.clone(),
                    state.queue.clone(),
                    state.renderer.clone(),
                    burn_device.clone(),
                    state.adapter.get_info(),
                );
            }
        }

        Self {
            tree,
            tree_ctx: AppTree { process: context },
        }
    }

    fn create_default_tree() -> egui_tiles::Tree<Pane> {
        let mut tiles: Tiles<Pane> = Tiles::default();

        let status_pane = tiles.insert_pane(Pane::Settings(SettingsPanel::default()));
        let scene_pane = tiles.insert_pane(Pane::Scene(Box::default()));

        #[cfg(feature = "training")]
        let main_content = {
            let stats_pane = tiles.insert_pane(Pane::Stats(StatsPanel::default()));
            let dataset_pane = tiles.insert_pane(Pane::Dataset(DatasetPanel::default()));

            let mut sidebar = egui_tiles::Linear::new(
                egui_tiles::LinearDir::Vertical,
                vec![dataset_pane, stats_pane],
            );
            sidebar.shares.set_share(dataset_pane, 0.50);
            let sidebar_id = tiles.insert_container(sidebar);

            let mut content = egui_tiles::Linear::new(
                egui_tiles::LinearDir::Horizontal,
                vec![scene_pane, sidebar_id],
            );
            content.shares.set_share(sidebar_id, 0.30);
            tiles.insert_container(content)
        };

        #[cfg(not(feature = "training"))]
        let main_content = scene_pane;

        let mut root = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Vertical,
            vec![status_pane, main_content],
        );
        root.shares.set_share(status_pane, 0.06);
        let root_id = tiles.insert_container(root);

        egui_tiles::Tree::new("brush_tree", root_id, tiles)
    }

    fn receive_messages(&mut self) {
        let _span = trace_span!("Receive Messages").entered();
        for message in self.tree_ctx.process.message_queue() {
            for (_, tile) in self.tree.tiles.iter_mut() {
                if let egui_tiles::Tile::Pane(pane) = tile {
                    let p = pane.as_pane_mut();
                    match &message {
                        Ok(msg) => p.on_message(msg, self.tree_ctx.process.as_ref()),
                        Err(e) => p.on_error(e, self.tree_ctx.process.as_ref()),
                    }
                }
            }
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, TREE_STORAGE_KEY, &self.tree);
    }

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let _span = trace_span!("Update UI").entered();
        self.receive_messages();

        let process = self.tree_ctx.process.clone();

        // Compute visibility
        fn is_visible(
            id: TileId,
            tiles: &Tiles<Pane>,
            process: &UiProcess,
            cache: &mut HashMap<TileId, bool>,
        ) -> bool {
            if let Some(&v) = cache.get(&id) {
                return v;
            }
            let v = match tiles.get(id) {
                Some(egui_tiles::Tile::Pane(p)) => p.as_pane().is_visible(process),
                Some(egui_tiles::Tile::Container(c)) => c
                    .active_children()
                    .any(|&cid| is_visible(cid, tiles, process, cache)),
                None => false,
            };
            cache.insert(id, v);
            v
        }

        let mut cache = HashMap::new();
        for id in self.tree.tiles.tile_ids().collect::<Vec<_>>() {
            self.tree
                .set_visible(id, is_visible(id, &self.tree.tiles, &process, &mut cache));
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(ctx.style().as_ref()).inner_margin(0.0))
            .show(ctx, |ui| self.tree.ui(&mut self.tree_ctx, ui));

        if ctx.input(|i| i.key_pressed(egui::Key::F)) && !ctx.wants_keyboard_input() {
            let new_mode = match self.tree_ctx.process.ui_mode() {
                UiMode::Default => UiMode::FullScreenSplat,
                UiMode::FullScreenSplat => UiMode::Default,
                UiMode::EmbeddedViewer => UiMode::EmbeddedViewer,
            };
            self.tree_ctx.process.set_ui_mode(new_mode);
        }
    }
}
