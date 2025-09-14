#![cfg(target_family = "wasm")]

use crate::three::ThreeVector3;
use anyhow::Context;
use brush_process::config::ProcessArgs;
use brush_ui::UiMode;
use brush_ui::app::App;
use brush_ui::ui_process::UiProcess;
use brush_vfs::DataSource;
use glam::{EulerRot, Quat};
use std::sync::Arc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

mod three;

pub fn wasm_app(canvas_name: &str) -> anyhow::Result<Arc<UiProcess>> {
    // TODO: Only do once.
    wasm_logger::init(wasm_logger::Config::new(log::Level::Info));

    let context = Arc::new(UiProcess::new());

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

#[wasm_bindgen]
pub struct EmbeddedApp {
    context: Arc<UiProcess>,
}

//Wrapper for interop.
#[wasm_bindgen]
pub struct CameraSettings(brush_ui::app::CameraSettings);

#[wasm_bindgen]
impl CameraSettings {
    #[wasm_bindgen(constructor)]
    pub fn new(
        background: Option<ThreeVector3>,
        speed_scale: Option<f32>,
        min_focus_distance: Option<f32>,
        max_focus_distance: Option<f32>,
        min_pitch: Option<f32>,
        max_pitch: Option<f32>,
        min_yaw: Option<f32>,
        max_yaw: Option<f32>,
        splat_scale: Option<f32>,
        grid_enabled: Option<bool>,
    ) -> Self {
        Self(brush_ui::app::CameraSettings {
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
            background: background.map(|v| v.to_glam()),
            grid_enabled,
        })
    }
}

#[wasm_bindgen]
impl EmbeddedApp {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_name: &str) -> Result<Self, JsError> {
        let context = wasm_app(canvas_name).map_err(|e| JsError::from(&*e))?;
        Ok(Self { context })
    }

    #[wasm_bindgen]
    pub fn load_url(&self, url: &str) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let _ = sender.send(ProcessArgs::default());

        self.context
            .start_new_process(DataSource::Url(url.to_owned()), receiver);
    }

    #[wasm_bindgen]
    pub fn set_cam_settings(&self, settings: CameraSettings) {
        self.context.set_cam_settings(&settings.0);
    }

    #[wasm_bindgen]
    pub fn set_cam_focus_distance(&self, distance: f32) {
        self.context.set_cam_focus_distance(distance);
    }

    #[wasm_bindgen]
    pub fn set_cam_fov(&self, fov: f64) {
        self.context.set_cam_fov(fov);
    }

    #[wasm_bindgen]
    pub fn set_cam_transform(&self, position: ThreeVector3, rotation_euler: ThreeVector3) {
        let position = position.to_glam();
        // 'XYZ' matches the THREE.js default order.
        let rotation = Quat::from_euler(
            EulerRot::XYZ,
            rotation_euler.x() as f32,
            rotation_euler.y() as f32,
            rotation_euler.z() as f32,
        );
        self.context.set_cam_transform(position, rotation);
    }

    #[wasm_bindgen]
    pub fn set_ui_mode(&self, mode: UiMode) {
        self.context.set_ui_mode(mode);
    }
}
