use brush_process::config::TrainStreamConfig;
use brush_ui::UiMode;
use brush_ui::app::App;
use brush_vfs::DataSource;
use glam::{EulerRot, Quat, Vec3};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use crate::shared::startup;

// THREE.js Vector3 bindings.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Vector3, js_namespace = THREE)]
    pub type ThreeVector3;

    #[wasm_bindgen(method, getter)]
    fn x(this: &ThreeVector3) -> f64;

    #[wasm_bindgen(method, getter)]
    fn y(this: &ThreeVector3) -> f64;

    #[wasm_bindgen(method, getter)]
    fn z(this: &ThreeVector3) -> f64;
}

impl ThreeVector3 {
    fn to_glam(&self) -> Vec3 {
        Vec3::new(self.x() as f32, self.y() as f32, self.z() as f32)
    }
}

// Wrapper for interop.
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

#[derive(Clone)]
#[wasm_bindgen]
pub struct EmbeddedApp {
    runner: eframe::WebRunner,
}

#[wasm_bindgen]
#[allow(clippy::needless_pass_by_value)] // wasm_bindgen FFI types need pass by value
impl EmbeddedApp {
    /// Installs a panic hook, then returns.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        #[cfg(debug_assertions)]
        wasm_logger::init(wasm_logger::Config::new(log::Level::Info));

        startup();

        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    /// Call this once from JavaScript to start your app.
    #[wasm_bindgen]
    pub async fn start(&self, canvas_name: &str) -> Result<(), wasm_bindgen::JsValue> {
        let wgpu_options = brush_ui::create_egui_options();
        let document = web_sys::window()
            .ok_or_else(|| JsValue::from_str("Failed to get window"))?
            .document()
            .ok_or_else(|| JsValue::from_str("Failed to get document"))?;
        let canvas = document
            .get_element_by_id(canvas_name)
            .ok_or_else(|| {
                JsValue::from_str(&format!("Failed to find canvas with id: {canvas_name}"))
            })?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_e| {
                JsValue::from_str(&format!(
                    "Found canvas {canvas_name} was in fact not a canvas"
                ))
            })?;

        self.runner
            .start(
                canvas,
                eframe::WebOptions {
                    wgpu_options,
                    ..Default::default()
                },
                Box::new(|cc| Ok(Box::new(App::new(cc, None, None)))),
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to start eframe: {e:?}")))?;

        Ok(())
    }

    #[wasm_bindgen]
    pub fn load_url(&self, url: &str) {
        if let Some(app) = self.runner.app_mut::<App>() {
            app.context()
                .start_new_process(DataSource::Url(url.to_owned()), async {
                    TrainStreamConfig::default()
                });
        }
    }

    #[wasm_bindgen]
    pub fn set_cam_settings(&self, settings: CameraSettings) {
        if let Some(app) = self.runner.app_mut::<App>() {
            app.context().set_cam_settings(&settings.0);
        }
    }

    #[wasm_bindgen]
    pub fn set_cam_fov(&self, fov: f64) {
        if let Some(app) = self.runner.app_mut::<App>() {
            app.context().set_cam_fov(fov);
        }
    }

    #[wasm_bindgen]
    pub fn set_cam_transform(&self, position: ThreeVector3, rotation_euler: ThreeVector3) {
        if let Some(app) = self.runner.app_mut::<App>() {
            let position = position.to_glam();
            // 'XYZ' matches the THREE.js default order.
            let rotation = Quat::from_euler(
                EulerRot::XYZ,
                rotation_euler.x() as f32,
                rotation_euler.y() as f32,
                rotation_euler.z() as f32,
            );
            app.context().set_cam_transform(position, rotation);
        }
    }

    #[wasm_bindgen]
    pub fn set_focal_point(
        &self,
        focal_point: ThreeVector3,
        focus_distance: f32,
        rotation_euler: ThreeVector3,
    ) {
        if let Some(app) = self.runner.app_mut::<App>() {
            // 'XYZ' matches the THREE.js default order.
            let rotation = Quat::from_euler(
                EulerRot::XYZ,
                rotation_euler.x() as f32,
                rotation_euler.y() as f32,
                rotation_euler.z() as f32,
            );
            let focal_point = focal_point.to_glam();
            app.context()
                .set_focal_point(focal_point, focus_distance, rotation);
        }
    }

    #[wasm_bindgen]
    pub fn set_ui_mode(&self, mode: UiMode) {
        if let Some(app) = self.runner.app_mut::<App>() {
            app.context().set_ui_mode(mode);
        }
    }
}
