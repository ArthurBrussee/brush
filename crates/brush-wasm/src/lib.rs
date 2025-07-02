#![cfg(target_family = "wasm")]

use crate::three::ThreeVector3;
use anyhow::Context;
use brush_process::config::ProcessArgs;
use brush_ui::UiMode;
use brush_ui::app::App;
use brush_ui::ui_process::UiProcess;
use brush_vfs::DataSource;
use glam::{EulerRot, Quat};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use tokio_with_wasm::alias as tokio_wasm;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

mod three;

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
        position: ThreeVector3,
        rotation_euler: ThreeVector3,
        background: ThreeVector3,
        speed_scale: Option<f32>,
        min_focus_distance: Option<f32>,
        max_focus_distance: Option<f32>,
        min_pitch: Option<f32>,
        max_pitch: Option<f32>,
        min_yaw: Option<f32>,
        max_yaw: Option<f32>,
        splat_scale: Option<f32>,
    ) -> Self {
        Self(brush_ui::app::CameraSettings {
            fov_y,
            position: position.to_glam(),
            // 'XYZ' matches the THREE.js default order.
            rotation: Quat::from_euler(
                EulerRot::XYZ,
                rotation_euler.x() as f32,
                rotation_euler.y() as f32,
                rotation_euler.z() as f32,
            ),
            speed_scale,
            splat_scale,
            // TODO: Could make this a separate JS object.
            clamping: brush_ui::camera_controls::CameraClamping {
                min_focus_distance,
                max_focus_distance,
                min_pitch,
                max_pitch,
                min_yaw,
                max_yaw,
            },
            background: background.to_glam(),
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
                        let (sender, receiver) = tokio::sync::oneshot::channel();
                        let _ = sender.send(ProcessArgs::default());
                        context.start_new_process(data_source, receiver);
                    }
                    EmbeddedCommands::SetCamSettings(settings) => {
                        context.set_cam_settings(&settings.0);
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
