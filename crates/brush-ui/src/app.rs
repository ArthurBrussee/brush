use crate::UiMode;
use crate::ui_process::UiProcess;
use crate::{
    camera_controls::CameraClamping, datasets::DatasetPanel, panels::PaneType, scene::ScenePanel,
    settings::SettingsPanel, stats::StatsPanel,
};
use brush_process::message::ProcessMessage;
use eframe::egui;
use egui::ThemePreference;
use egui_tiles::{Container, SimplificationOptions, Tile, TileId, Tiles};
use glam::{Quat, Vec3};
use std::sync::Arc;

pub(crate) struct AppTree {
    context: Arc<UiProcess>,
}

impl egui_tiles::Behavior<PaneType> for AppTree {
    fn tab_title_for_pane(&mut self, pane: &PaneType) -> egui::WidgetText {
        pane.title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut PaneType,
    ) -> egui_tiles::UiResponse {
        egui::Frame::new()
            .inner_margin(pane.inner_margin())
            .show(ui, |ui| {
                pane.ui(ui, self.context.as_ref());
            });
        egui_tiles::UiResponse::None
    }

    /// What are the rules for simplifying the tree?
    fn simplification_options(&self) -> SimplificationOptions {
        SimplificationOptions {
            all_panes_must_have_tabs: self.context.ui_mode() == UiMode::Full,
            ..Default::default()
        }
    }

    /// Width of the gap between tiles in a horizontal or vertical layout,
    /// and between rows/columns in a grid layout.
    fn gap_width(&self, _style: &egui::Style) -> f32 {
        if self.context.ui_mode() == UiMode::Zen {
            0.0
        } else {
            0.5
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct CameraSettings {
    pub fov_y: f64,
    pub position: Vec3,
    pub rotation: Quat,
    pub speed_scale: Option<f32>,
    pub splat_scale: Option<f32>,
    pub clamping: CameraClamping,
    pub background: Vec3,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            fov_y: 0.8,
            position: -Vec3::Z * 2.5,
            rotation: Quat::IDENTITY,
            speed_scale: None,
            splat_scale: None,
            clamping: CameraClamping::default(),
            background: Vec3::ZERO,
        }
    }
}

pub struct App {
    tree: egui_tiles::Tree<PaneType>,
    datasets: Option<TileId>,
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
        let scene_pane = ScenePanel::new(
            state.device.clone(),
            state.queue.clone(),
            state.renderer.clone(),
            context.ui_mode(),
        );

        let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        let root_container = if context.ui_mode() == UiMode::Full {
            let loading_subs = vec![tiles.insert_pane(Box::new(SettingsPanel::new()))];
            let loading_pane = tiles.insert_tab_tile(loading_subs);

            #[allow(unused_mut)]
            let mut sides = vec![
                loading_pane,
                tiles.insert_pane(Box::new(StatsPanel::new(device, state.adapter.get_info()))),
            ];

            let side_panel = tiles.insert_vertical_tile(sides);

            let mut lin = egui_tiles::Linear::new(
                egui_tiles::LinearDir::Horizontal,
                vec![side_panel, scene_pane_id],
            );
            lin.shares.set_share(side_panel, 0.4);
            tiles.insert_container(lin)
        } else {
            scene_pane_id
        };

        let tree = egui_tiles::Tree::new("brush_tree", root_container, tiles);

        let tree_ctx = AppTree { context };

        Self {
            tree,
            tree_ctx,
            datasets: None,
        }
    }

    fn receive_messages(&mut self) {
        let mut messages = vec![];

        while let Some(message) = self.tree_ctx.context.try_recv_message() {
            messages.push(message);
        }

        for message in messages {
            match message {
                Ok(message) => {
                    if let ProcessMessage::Dataset { dataset: _ } = message {
                        // Show the dataset panel if we've loaded one.
                        if self.datasets.is_none() {
                            let pane_id =
                                self.tree.tiles.insert_pane(Box::new(DatasetPanel::new()));
                            self.datasets = Some(pane_id);
                            if let Some(Tile::Container(Container::Linear(lin))) = self
                                .tree
                                .tiles
                                .get_mut(self.tree.root().expect("UI must have a root"))
                            {
                                lin.add_child(pane_id);
                            }
                        }
                    }

                    for (_, pane) in self.tree.tiles.iter_mut() {
                        match pane {
                            Tile::Pane(pane) => {
                                pane.on_message(&message, self.tree_ctx.context.as_ref());
                            }
                            Tile::Container(_) => {}
                        }
                    }
                }
                Err(e) => {
                    for (_, pane) in self.tree.tiles.iter_mut() {
                        match pane {
                            Tile::Pane(pane) => {
                                pane.on_error(&e, self.tree_ctx.context.as_ref());
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
        self.receive_messages();

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(ctx.style().as_ref()).inner_margin(0.0))
            .show(ctx, |ui| {
                self.tree.ui(&mut self.tree_ctx, ui);
            });
    }
}
