use std::sync::{Arc, RwLock};

use crate::orbit_controls::CameraController;
use crate::panels::SettingsPanel;
use crate::panels::{DatasetPanel, PresetsPanel, ScenePanel, StatsPanel, TracingPanel};
use brush_dataset::Dataset;
use brush_process::data_source::DataSource;
use brush_process::process_loop::{
    start_process, ControlMessage, ProcessArgs, ProcessMessage, RunningProcess,
};
use brush_render::camera::Camera;
use brush_ui::channel::reactive_receiver;
use burn_wgpu::WgpuDevice;
use eframe::egui;
use egui_tiles::SimplificationOptions;
use egui_tiles::{Container, Tile, TileId, Tiles};
use glam::{Affine3A, Quat, Vec3};
use std::collections::HashMap;

pub(crate) trait AppPanel {
    fn title(&self) -> String;
    fn ui(&mut self, ui: &mut egui::Ui, controls: &mut AppContext);
    fn on_message(&mut self, message: &ProcessMessage, context: &mut AppContext) {
        let _ = message;
        let _ = context;
    }
}

struct AppTree {
    zen: bool,
    context: Arc<RwLock<AppContext>>,
}

type PaneType = Box<dyn AppPanel>;

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
        pane.ui(ui, &mut self.context.write().expect("Lock poisoned"));
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

pub struct App {
    tree: egui_tiles::Tree<PaneType>,
    datasets: Option<TileId>,
    tree_ctx: AppTree,
}

// TODO: Bit too much random shared state here.
pub struct AppContext {
    pub dataset: Dataset,
    pub camera: Camera,
    pub controls: CameraController,
    pub model_local_to_world: Affine3A,
    pub device: WgpuDevice,
    ctx: egui::Context,
    running_process: Option<RunningProcess>,
}

struct CameraSettings {
    focal: f64,
    radius: f32,
}

impl AppContext {
    fn new(device: WgpuDevice, ctx: egui::Context, cam_settings: &CameraSettings) -> Self {
        let model_transform = Affine3A::IDENTITY;

        let controls = CameraController::new(cam_settings.radius);

        // Camera position will be controlled by the orbit controls.
        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            cam_settings.focal,
            cam_settings.focal,
            glam::vec2(0.5, 0.5),
        );

        Self {
            camera,
            controls,
            model_local_to_world: model_transform,
            device,
            ctx,
            dataset: Dataset::empty(),
            running_process: None,
        }
    }

    pub fn set_model_up(&mut self, up_axis: Vec3) {
        let rotation = Quat::from_rotation_arc(Vec3::Y, up_axis);

        let object_from_cam = self.model_local_to_world.inverse() * self.camera.local_to_world();

        let transform = Affine3A::from_rotation_translation(rotation, Vec3::ZERO);

        self.model_local_to_world = transform;
        let cam = self.model_local_to_world * object_from_cam;

        self.controls.position = cam.translation.into();
        self.controls.rotation = Quat::from_mat3a(&cam.matrix3);
    }

    pub fn focus_view(&mut self, view_cam: &Camera) {
        self.camera = view_cam.clone();
        let transform = self.model_local_to_world.inverse() * self.camera.local_to_world();
        self.controls.position = transform.translation.into();
        self.controls.rotation = Quat::from_mat3a(&transform.matrix3);
    }

    pub fn connect_to(&mut self, process: RunningProcess) {
        self.dataset = Dataset::empty();
        // Convert the receiver to a "reactive" receiver that wakes up the UI.
        let process = RunningProcess {
            messages: reactive_receiver(process.messages, self.ctx.clone()),
            ..process
        };
        self.running_process = Some(process);
    }

    pub(crate) fn control_message(&self, msg: ControlMessage) {
        if let Some(process) = self.running_process.as_ref() {
            let _ = process.control.send(msg);
        }
    }
}

pub struct AppCreateCb {
    pub context: Arc<RwLock<AppContext>>,
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext,
        create_callback: tokio::sync::oneshot::Sender<AppCreateCb>,
    ) -> Self {
        // For now just assume we're running on the default
        let state = cc
            .wgpu_render_state
            .as_ref()
            .expect("No wgpu renderer enabled in egui");
        let device = brush_render::burn_init_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        if cfg!(feature = "tracing") {
            // TODO: In debug only?
            #[cfg(target_family = "wasm")]
            {
                use tracing_subscriber::layer::SubscriberExt;

                tracing::subscriber::set_global_default(
                    tracing_subscriber::registry()
                        .with(tracing_wasm::WASMLayer::new(Default::default())),
                )
                .expect("Failed to set tracing subscriber");
            }

            #[cfg(all(feature = "tracy", not(target_family = "wasm")))]
            {
                use tracing_subscriber::layer::SubscriberExt;

                tracing::subscriber::set_global_default(
                    tracing_subscriber::registry()
                        .with(tracing_tracy::TracyLayer::default())
                        .with(sync_span::SyncLayer::<
                            burn_jit::JitBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>,
                        >::new(device.clone())),
                )
                .expect("Failed to set tracing subscriber");
            }
        }

        #[cfg(target_family = "wasm")]
        let start_uri = web_sys::window().and_then(|w| w.location().search().ok());
        #[cfg(not(target_family = "wasm"))]
        let start_uri: Option<String> = None;

        let search_params = parse_search(start_uri.as_deref().unwrap_or(""));

        let mut zen = false;
        if let Some(z) = search_params.get("zen") {
            zen = z.parse::<bool>().unwrap_or(false);
        }

        let focal = search_params
            .get("focal")
            .and_then(|f| f.parse().ok())
            .unwrap_or(0.8);
        let radius = search_params
            .get("radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(4.0);

        let settings = CameraSettings { focal, radius };
        let context = AppContext::new(device.clone(), cc.egui_ctx.clone(), &settings);

        let mut tiles: Tiles<PaneType> = Tiles::default();
        let scene_pane = ScenePanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
            zen,
        );

        let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        let root_container = if !zen {
            let loading_subs = vec![
                tiles.insert_pane(Box::new(SettingsPanel::new())),
                tiles.insert_pane(Box::new(PresetsPanel::new())),
            ];
            let loading_pane = tiles.insert_tab_tile(loading_subs);

            #[allow(unused_mut)]
            let mut sides = vec![
                loading_pane,
                tiles.insert_pane(Box::new(StatsPanel::new(device.clone(), &state.adapter))),
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

        let context = Arc::new(RwLock::new(context));
        let _ = create_callback.send(AppCreateCb {
            context: context.clone(),
        });

        let tree_ctx = AppTree { zen, context };

        let url = search_params.get("url");
        if let Some(url) = url {
            let running = start_process(
                DataSource::Url(url.to_owned()),
                ProcessArgs::default(),
                device,
            );
            tree_ctx
                .context
                .write()
                .expect("Lock poisoned")
                .connect_to(running);
        }

        Self {
            tree,
            tree_ctx,
            datasets: None,
        }
    }
}

impl App {
    fn receive_messages(&mut self) {
        let mut context = self.tree_ctx.context.write().expect("Lock poisoned");

        if let Some(process) = context.running_process.as_mut() {
            let mut messages = vec![];

            while let Ok(message) = process.messages.try_recv() {
                messages.push(message);
            }

            for message in messages {
                if let ProcessMessage::Dataset { data: _ } = message {
                    // Show the dataset panel if we've loaded one.
                    if self.datasets.is_none() {
                        let pane_id = self.tree.tiles.insert_pane(Box::new(DatasetPanel::new()));
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
                            pane.on_message(&message, &mut context);
                        }
                        Tile::Container(_) => {}
                    }
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.receive_messages();

        egui::CentralPanel::default().show(ctx, |ui| {
            self.tree.ui(&mut self.tree_ctx, ui);
        });
    }
}
