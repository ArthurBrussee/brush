use core::f32;
use std::ops::Range;
use std::sync::{Arc, RwLock};

use crate::panels::SettingsPanel;
use crate::{
    orbit_controls::OrbitControls,
    panels::{DatasetPanel, PresetsPanel, ScenePanel, StatsPanel, TracingPanel},
};
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
use glam::{Affine3A, Quat, Vec3, Vec3A};
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
    pub controls: OrbitControls,
    pub model_transform: Affine3A,
    pub device: WgpuDevice,
    ctx: egui::Context,
    running_process: Option<RunningProcess>,
}

struct CameraSettings {
    focal: f64,
    radius: f32,
    yaw_range: Range<f32>,
    pitch_range: Range<f32>,
    radius_range: Range<f32>,
}

impl AppContext {
    fn new(device: WgpuDevice, ctx: egui::Context, cam_settings: CameraSettings) -> Self {
        let model_transform = Affine3A::IDENTITY;

        let controls = OrbitControls::new(
            cam_settings.radius,
            cam_settings.radius_range,
            cam_settings.yaw_range,
            cam_settings.pitch_range,
        );

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
            model_transform,
            device,
            ctx,
            dataset: Dataset::empty(),
            running_process: None,
        }
    }

    fn update_control_positions(&mut self) {
        // set the controls transform.
        let cam_transform =
            Affine3A::from_rotation_translation(self.camera.rotation, self.camera.position);
        let transform = self.model_transform.inverse() * cam_transform;
        let (_, rotation, position) = transform.to_scale_rotation_translation();
        self.controls.position = position.into();
        self.controls.rotation = rotation;
        self.controls.dirty = true;
    }

    pub fn set_up_axis(&mut self, up_axis: Vec3) {
        let rotation = Quat::from_rotation_arc(Vec3::Y, up_axis);
        let model_transform = Affine3A::from_rotation_translation(rotation, Vec3::ZERO).inverse();
        self.model_transform = model_transform;
        self.update_control_positions();
    }

    pub fn focus_view(&mut self, cam: &Camera) {
        self.camera = cam.clone();
        self.update_control_positions();
        self.controls.focus = self.controls.position
            + self.controls.rotation * Vec3A::Z * self.dataset.train.bounds().extent.length() * 0.5;
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
        let device = brush_ui::create_wgpu_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        // brush_render::render::set_hard_floats_available(
        //     state
        //         .adapter
        //         .features()
        //         .contains(Features::SHADER_FLT32_ATOMIC),
        // );

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
            .unwrap_or(0.5);
        let radius = search_params
            .get("radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(4.0);

        let min_radius = search_params
            .get("min_radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(1.0);
        let max_radius = search_params
            .get("max_radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(10.0);

        let min_yaw = search_params
            .get("min_yaw")
            .and_then(|f| f.parse::<f32>().ok())
            .map_or(f32::MIN, |d| d.to_radians());
        let max_yaw = search_params
            .get("max_yaw")
            .and_then(|f| f.parse::<f32>().ok())
            .map_or(f32::MAX, |d| d.to_radians());

        let min_pitch = search_params
            .get("min_pitch")
            .and_then(|f| f.parse::<f32>().ok())
            .map_or(f32::MIN, |d| d.to_radians());
        let max_pitch = search_params
            .get("max_pitch")
            .and_then(|f| f.parse::<f32>().ok())
            .map_or(f32::MAX, |d| d.to_radians());

        let settings = CameraSettings {
            focal,
            radius,
            radius_range: min_radius..max_radius,
            yaw_range: min_yaw..max_yaw,
            pitch_range: min_pitch..max_pitch,
        };

        let context = AppContext::new(device.clone(), cc.egui_ctx.clone(), settings);

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
            // Close when pressing escape (in a native viewer anyway).
            #[cfg(not(target_family = "wasm"))]
            if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            self.tree.ui(&mut self.tree_ctx, ui);
        });
    }
}
