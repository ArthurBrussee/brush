use crate::{
    BrushUiProcess, camera_controls::CameraClamping, datasets::DatasetPanel, panels::PaneType,
    scene::ScenePanel, settings::SettingsPanel, stats::StatsPanel, tracing_debug::TracingPanel,
};
use brush_process::config::ProcessArgs;
use brush_process::message::ProcessMessage;
use brush_vfs::DataSource;
use eframe::egui;
use egui::ThemePreference;
use egui_tiles::{Container, SimplificationOptions, Tile, TileId, Tiles};
use glam::{Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

fn parse_search(search: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    let search = search.trim_start_matches('?');

    for pair in search.split('&') {
        // Split each pair on '=' to separate key and value
        if let Some((key, value)) = pair.split_once('=') {
            // URL decode the key and value and insert into HashMap
            params.insert(
                urlencoding::decode(key).unwrap_or_default().into_owned(),
                urlencoding::decode(value).unwrap_or_default().into_owned(),
            );
        }
    }
    params
}

pub(crate) struct AppTree {
    zen: bool,
    context: Arc<dyn BrushUiProcess>,
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
            all_panes_must_have_tabs: !self.zen,
            ..Default::default()
        }
    }

    /// Width of the gap between tiles in a horizontal or vertical layout,
    /// and between rows/columns in a grid layout.
    fn gap_width(&self, _style: &egui::Style) -> f32 {
        if self.zen { 0.0 } else { 0.5 }
    }
}

#[derive(Clone)]
pub struct CameraSettings {
    pub focal: f64,
    pub init_position: Vec3,
    pub init_rotation: Quat,
    pub focus_distance: f32,
    pub speed_scale: f32,
    pub clamping: CameraClamping,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            focal: 0.8,
            init_position: -Vec3::Z * 2.5,
            init_rotation: Quat::IDENTITY,
            focus_distance: 4.0,
            speed_scale: 1.0,
            clamping: CameraClamping::default(),
        }
    }
}

pub struct App {
    tree: egui_tiles::Tree<PaneType>,
    datasets: Option<TileId>,
    tree_ctx: AppTree,
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext,
        start_uri_override: Option<String>,
        context: Arc<dyn BrushUiProcess>,
    ) -> Self {
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

        let start_uri = start_uri_override;

        #[cfg(target_family = "wasm")]
        let start_uri =
            start_uri.or_else(|| web_sys::window().and_then(|w| w.location().search().ok()));

        let search_params = parse_search(start_uri.as_deref().unwrap_or(""));

        let mut zen = false;
        if let Some(z) = search_params.get("zen") {
            zen = z.parse::<bool>().unwrap_or(false);
        }

        fn vec_from_uri(uri: &str) -> Option<Vec3> {
            let parts: Vec<&str> = uri.split(',').collect();
            if parts.len() == 3 {
                Some(Vec3::new(
                    parts[0].parse().ok()?,
                    parts[1].parse().ok()?,
                    parts[2].parse().ok()?,
                ))
            } else {
                None
            }
        }

        fn quat_from_uri(uri: &str) -> Option<Quat> {
            let parts: Vec<&str> = uri.split(',').collect();
            if parts.len() == 4 {
                Some(Quat::from_xyzw(
                    parts[0].parse().ok()?,
                    parts[1].parse().ok()?,
                    parts[2].parse().ok()?,
                    parts[3].parse().ok()?,
                ))
            } else {
                None
            }
        }

        // TODO: Integrate this with the embedded API.
        let position = search_params
            .get("position")
            .and_then(|f| vec_from_uri(f))
            .unwrap_or(-Vec3::Z * 2.5);
        let rotation = search_params
            .get("rotation")
            .and_then(|f| quat_from_uri(f))
            .unwrap_or(Quat::IDENTITY);
        let focus_distance = search_params
            .get("focus_distance")
            .and_then(|f| f.parse().ok())
            .unwrap_or(4.0);
        let focal = search_params
            .get("focal")
            .and_then(|f| f.parse().ok())
            .unwrap_or(0.8);
        let speed_scale = search_params
            .get("speed_scale")
            .and_then(|f| f.parse().ok())
            .unwrap_or(1.0);
        context.set_cam_settings(CameraSettings {
            focal,
            init_position: position,
            init_rotation: rotation,
            focus_distance,
            speed_scale,
            clamping: Default::default(),
        });

        let mut tiles: Tiles<PaneType> = Tiles::default();
        let scene_pane = ScenePanel::new(
            state.device.clone(),
            state.queue.clone(),
            state.renderer.clone(),
            zen,
        );

        let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        let root_container = if !zen {
            let loading_subs = vec![tiles.insert_pane(Box::new(SettingsPanel::new()))];
            let loading_pane = tiles.insert_tab_tile(loading_subs);

            #[allow(unused_mut)]
            let mut sides = vec![
                loading_pane,
                tiles.insert_pane(Box::new(StatsPanel::new(device, state.adapter.get_info()))),
            ];

            if cfg!(feature = "tracing") {
                sides.push(tiles.insert_pane(Box::new(TracingPanel::default())));
            }

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

        let url = search_params.get("url").cloned();
        if let Some(url) = url {
            context.start_new_process(DataSource::Url(url), ProcessArgs::default());
        }

        let tree_ctx = AppTree { zen, context };

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
