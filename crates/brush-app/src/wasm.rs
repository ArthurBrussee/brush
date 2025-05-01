use crate::ui_process::UiProcess;
use anyhow::Context;
use brush_process::config::ProcessArgs;
use brush_ui::BrushUiProcess;
use brush_ui::UiMode;
use brush_ui::app::App;
use brush_vfs::DataSource;
use glam::Vec3;
use glam::{EulerRot, Quat};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use tokio_with_wasm::alias as tokio_wasm;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

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

pub fn wasm_app(canvas_name: &str, start_uri: &str) -> anyhow::Result<Arc<UiProcess>> {
    let search_params = parse_search(start_uri);
    let mut zen = false;
    if let Some(z) = search_params.get("zen") {
        zen = z.parse::<bool>().unwrap_or(false);
    }

    let context = Arc::new(UiProcess::new(if zen { UiMode::Zen } else { UiMode::Full }));

    let wgpu_options = brush_ui::create_egui_options();
    let document = web_sys::window()
        .context("Failed to get winow")?
        .document()
        .context("Failed to get document")?;
    let canvas = document
        .get_element_by_id(canvas_name)
        .with_context(|| format!("Failed to find canvas with id: {canvas_name}"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap_or_else(|_| panic!("Found canvas {canvas_name} was in fact not a canvas"));

    // On wasm, run as a local task.
    let context_cl = context.clone();

    tokio_with_wasm::task::spawn(async {
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions {
                    wgpu_options,
                    ..Default::default()
                },
                Box::new(|cc| Ok(Box::new(App::new(cc, context_cl)))),
            )
            .await
            .expect("failed to start eframe");
    });

    let url = search_params.get("url").cloned();
    if let Some(url) = url {
        context.start_new_process(DataSource::Url(url), ProcessArgs::default());
    }

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
    let fov_y = search_params
        .get("fov_y")
        .and_then(|f| f.parse().ok())
        .unwrap_or(0.8);
    let speed_scale = search_params
        .get("speed_scale")
        .and_then(|f| f.parse().ok());

    context.set_cam_settings(brush_ui::app::CameraSettings {
        fov_y,
        position,
        rotation,
        focus_distance,
        speed_scale,
        clamping: Default::default(),
    });

    Ok(context)
}

enum EmbeddedCommands {
    LoadDataSource(DataSource),
    SetCamSettings(CameraSettings),
}

#[wasm_bindgen]
pub struct EmbeddedApp {
    command_channel: UnboundedSender<EmbeddedCommands>,
}

//Wrapper for interop.
#[wasm_bindgen]
pub struct CameraSettings(brush_ui::app::CameraSettings);

#[wasm_bindgen]
impl CameraSettings {
    #[wasm_bindgen(constructor)]
    pub fn new(
        fov_y: f64,
        x: f32,
        y: f32,
        z: f32,
        euler_x: f32,
        euler_y: f32,
        euler_z: f32,
        focus_distance: f32,
        speed_scale: Option<f32>,
        min_focus_distance: Option<f32>,
        max_focus_distance: Option<f32>,
        min_pitch: Option<f32>,
        max_pitch: Option<f32>,
        min_yaw: Option<f32>,
        max_yaw: Option<f32>,
    ) -> Self {
        Self(brush_ui::app::CameraSettings {
            fov_y,
            position: glam::vec3(x, y, z),
            // 'XYZ' matches the THREE.js default order.
            rotation: Quat::from_euler(EulerRot::XYZ, euler_x, euler_y, euler_z),
            focus_distance,
            speed_scale,
            // TODO: Could make this a separate JS object.
            clamping: brush_ui::camera_controls::CameraClamping {
                min_focus_distance,
                max_focus_distance,
                min_pitch,
                max_pitch,
                min_yaw,
                max_yaw,
            },
        })
    }
}

#[wasm_bindgen]
impl EmbeddedApp {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_name: &str, start_uri: &str) -> Result<Self, JsError> {
        let context = wasm_app(canvas_name, start_uri).map_err(|e| JsError::from(&*e))?;

        let (cmd_send, mut cmd_rec) = tokio::sync::mpsc::unbounded_channel();

        tokio_wasm::spawn(async move {
            while let Some(command) = cmd_rec.recv().await {
                match command {
                    EmbeddedCommands::LoadDataSource(data_source) => {
                        context.start_new_process(data_source, ProcessArgs::default());
                    }
                    EmbeddedCommands::SetCamSettings(settings) => {
                        context.set_cam_settings(settings.0);
                    }
                }
            }
        });

        Ok(Self {
            command_channel: cmd_send,
        })
    }

    #[wasm_bindgen]
    pub fn load_url(&self, url: &str) {
        self.command_channel
            .send(EmbeddedCommands::LoadDataSource(DataSource::Url(
                url.to_owned(),
            )))
            .expect("Viewer was closed?");
    }

    #[wasm_bindgen]
    pub fn set_camera_settings(&self, settings: CameraSettings) {
        self.command_channel
            .send(EmbeddedCommands::SetCamSettings(settings))
            .expect("Viewer was closed?");
    }
}
