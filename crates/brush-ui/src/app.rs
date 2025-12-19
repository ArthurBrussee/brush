use crate::UiMode;
#[cfg(feature = "training")]
use crate::datasets::DatasetPanel;
use crate::panels::AppPane;
#[cfg(feature = "training")]
use crate::stats::StatsPanel;
#[cfg(feature = "training")]
use crate::training_panel::TrainingPanel;
use crate::ui_process::UiProcess;
use crate::{camera_controls::CameraClamping, scene::ScenePanel};
use brush_process::message::ProcessMessage;
use eframe::egui;
use egui::{ThemePreference, Ui};
use egui_tiles::{SimplificationOptions, Tabs, TileId, Tiles};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::trace_span;

/// Pane enum that wraps all panel types for serialization.
#[derive(Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Pane {
    Scene(#[serde(skip)] ScenePanel),
    #[cfg(feature = "training")]
    Stats(#[serde(skip)] StatsPanel),
    #[cfg(feature = "training")]
    Dataset(#[serde(skip)] DatasetPanel),
    #[cfg(feature = "training")]
    Training(#[serde(skip)] TrainingPanel),
}

impl Pane {
    fn as_pane(&self) -> &dyn AppPane {
        match self {
            Self::Scene(p) => p,
            #[cfg(feature = "training")]
            Self::Stats(p) => p,
            #[cfg(feature = "training")]
            Self::Dataset(p) => p,
            #[cfg(feature = "training")]
            Self::Training(p) => p,
        }
    }

    fn as_pane_mut(&mut self) -> &mut dyn AppPane {
        match self {
            Self::Scene(p) => p,
            #[cfg(feature = "training")]
            Self::Stats(p) => p,
            #[cfg(feature = "training")]
            Self::Dataset(p) => p,
            #[cfg(feature = "training")]
            Self::Training(p) => p,
        }
    }

    fn scene() -> RefCell<Self> {
        RefCell::new(Self::Scene(ScenePanel::default()))
    }
}

#[cfg(feature = "training")]
impl Pane {
    fn stats() -> RefCell<Self> {
        RefCell::new(Self::Stats(StatsPanel::default()))
    }

    fn dataset() -> RefCell<Self> {
        RefCell::new(Self::Dataset(DatasetPanel::default()))
    }

    fn training() -> RefCell<Self> {
        RefCell::new(Self::Training(TrainingPanel::default()))
    }
}

type PaneRef = RefCell<Pane>;

pub(crate) struct AppTree {
    process: Arc<UiProcess>,
}

impl egui_tiles::Behavior<PaneRef> for AppTree {
    fn tab_title_for_pane(&mut self, pane: &PaneRef) -> egui::WidgetText {
        pane.borrow().as_pane().title()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: TileId,
        pane: &mut PaneRef,
    ) -> egui_tiles::UiResponse {
        let process = self.process.as_ref();
        let margin = pane.borrow().as_pane().inner_margin();
        egui::Frame::new()
            .inner_margin(margin)
            .show(ui, |ui| pane.get_mut().as_pane_mut().ui(ui, process));
        egui_tiles::UiResponse::None
    }

    fn top_bar_right_ui(
        &mut self,
        tiles: &Tiles<PaneRef>,
        ui: &mut Ui,
        _tile_id: TileId,
        tabs: &Tabs,
        _scroll_offset: &mut f32,
    ) {
        if let Some(active_id) = tabs.active
            && let Some(egui_tiles::Tile::Pane(pane)) = tiles.get(active_id)
        {
            pane.borrow_mut()
                .as_pane_mut()
                .top_bar_right_ui(ui, self.process.as_ref());
        }
    }

    fn simplification_options(&self) -> SimplificationOptions {
        SimplificationOptions {
            all_panes_must_have_tabs: matches!(
                self.process.ui_mode(),
                UiMode::Default | UiMode::FullScreenSplat
            ),
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

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        26.0
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

const TREE_STORAGE_KEY: &str = "brush_tile_tree_v2";

pub struct App {
    tree: egui_tiles::Tree<PaneRef>,
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
            .and_then(|s| eframe::get_value::<egui_tiles::Tree<PaneRef>>(s, TREE_STORAGE_KEY))
            .unwrap_or_else(|| {
                let mut tiles: Tiles<PaneRef> = Tiles::default();
                let scene_pane = tiles.insert_pane(Pane::scene());

                #[cfg(feature = "training")]
                let root_id = {
                    let stats_pane = tiles.insert_pane(Pane::stats());
                    let dataset_pane = tiles.insert_pane(Pane::dataset());
                    let training_pane = tiles.insert_pane(Pane::training());
                    Self::build_default_layout(
                        &mut tiles,
                        scene_pane,
                        stats_pane,
                        dataset_pane,
                        training_pane,
                    )
                };

                #[cfg(not(feature = "training"))]
                let root_id = scene_pane;

                egui_tiles::Tree::new("brush_tree", root_id, tiles)
            });

        // Initialize all panels with runtime state
        for (_, tile) in tree.tiles.iter_mut() {
            if let egui_tiles::Tile::Pane(pane) = tile {
                pane.get_mut().as_pane_mut().init(
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

    #[cfg(feature = "training")]
    fn build_default_layout(
        tiles: &mut Tiles<PaneRef>,
        scene_pane: TileId,
        stats_pane: TileId,
        dataset_pane: TileId,
        training_pane: TileId,
    ) -> TileId {
        let mut sidebar = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Vertical,
            vec![training_pane, dataset_pane, stats_pane],
        );
        sidebar.shares.set_share(training_pane, 0.07);
        sidebar.shares.set_share(dataset_pane, 0.5);
        sidebar.shares.set_share(stats_pane, 0.43);
        let sidebar_id = tiles.insert_container(sidebar);

        let mut content = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![scene_pane, sidebar_id],
        );
        content.shares.set_share(sidebar_id, 0.35);
        tiles.insert_container(content)
    }

    fn receive_messages(&mut self) {
        let _span = trace_span!("Receive Messages").entered();
        for message in self.tree_ctx.process.message_queue() {
            for (_, tile) in self.tree.tiles.iter_mut() {
                if let egui_tiles::Tile::Pane(pane) = tile {
                    let p = pane.get_mut().as_pane_mut();
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

        #[cfg(feature = "training")]
        if process.take_reset_layout_request() {
            fn find_pane(tiles: &Tiles<RefCell<Pane>>, f: fn(&Pane) -> bool) -> TileId {
                tiles
                    .iter()
                    .find_map(|(id, tile)| {
                        if let egui_tiles::Tile::Pane(pane) = tile
                            && f(&pane.borrow())
                        {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                    .expect("Missing pane")
            }

            let tree: &mut egui_tiles::Tree<PaneRef> = &mut self.tree;
            let scene_pane = find_pane(&tree.tiles, |p| matches!(p, Pane::Scene(_)));
            let stats_pane = find_pane(&tree.tiles, |p| matches!(p, Pane::Stats(_)));
            let dataset_pane = find_pane(&tree.tiles, |p| matches!(p, Pane::Dataset(_)));
            let training_pane = find_pane(&tree.tiles, |p| matches!(p, Pane::Training(_)));

            // Remove all container tiles
            let container_ids: Vec<TileId> = tree
                .tiles
                .iter()
                .filter_map(|(id, tile)| {
                    matches!(tile, egui_tiles::Tile::Container(_)).then_some(*id)
                })
                .collect();
            for id in container_ids {
                tree.tiles.remove(id);
            }
            tree.root = Some(Self::build_default_layout(
                &mut tree.tiles,
                scene_pane,
                stats_pane,
                dataset_pane,
                training_pane,
            ));
        }

        // Check for session reset request - notify all panes
        if process.take_session_reset_request() {
            for (_, tile) in self.tree.tiles.iter_mut() {
                if let egui_tiles::Tile::Pane(pane) = tile {
                    pane.get_mut()
                        .as_pane_mut()
                        .on_message(&ProcessMessage::NewProcess, &process);
                }
            }
        }

        fn is_visible(
            id: TileId,
            tiles: &Tiles<PaneRef>,
            process: &UiProcess,
            cache: &mut HashMap<TileId, bool>,
        ) -> bool {
            if let Some(&v) = cache.get(&id) {
                return v;
            }
            let v = match tiles.get(id) {
                Some(egui_tiles::Tile::Pane(p)) => p.borrow().as_pane().is_visible(process),
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
