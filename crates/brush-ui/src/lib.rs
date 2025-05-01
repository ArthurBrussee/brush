#![recursion_limit = "256"]

pub mod app;
pub mod burn_texture;
pub mod camera_controls;

use std::sync::Arc;

use app::CameraSettings;
use brush_dataset::scene::SceneView;
use brush_process::{config::ProcessArgs, message::ProcessMessage};
use brush_render::camera::Camera;
use brush_vfs::DataSource;
use burn_wgpu::WgpuDevice;
use eframe::egui_wgpu::WgpuConfiguration;
use egui::Response;
use glam::Vec3;
use wgpu::{Adapter, Features};

mod datasets;
mod panels;
mod scene;
mod settings;
mod stats;
mod tracing_debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Full,
    Zen,
}

// Two way communication with the UI.
// These are used both from the GUI, and for the exposed JS API's.
pub trait BrushUiProcess {
    /// Whether any process is currently loading (be it a ply for viewing, or a data set for training).
    fn is_loading(&self) -> bool;
    /// Whether there is a current running training process.
    fn is_training(&self) -> bool;
    /// Get the current camera state. Nb: This might not have the exact FOV you expect.
    fn current_camera(&self) -> Camera;
    /// Update the camera controls given an egui response.
    fn tick_controls(&self, response: &Response, ui: &egui::Ui);
    fn model_local_to_world(&self) -> glam::Affine3A;
    fn selected_view(&self) -> Option<SceneView>;
    fn set_train_paused(&self, paused: bool);
    fn set_cam_settings(&self, settings: CameraSettings);
    fn focus_view(&self, view: &SceneView);
    fn set_model_up(&self, up: Vec3);
    fn start_new_process(&self, source: DataSource, args: ProcessArgs);
    fn try_recv_message(&self) -> Option<anyhow::Result<ProcessMessage>>;
    fn connect_device(&self, device: WgpuDevice, ctx: egui::Context);
    fn ui_mode(&self) -> UiMode;
}

pub fn create_egui_options() -> WgpuConfiguration {
    WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
                power_preference: wgpu::PowerPreference::HighPerformance,
                device_descriptor: Arc::new(|adapter: &Adapter| wgpu::DeviceDescriptor {
                    label: Some("egui+burn"),
                    required_features: adapter
                        .features()
                        .difference(Features::MAPPABLE_PRIMARY_BUFFERS),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                }),
                ..Default::default()
            },
        ),
        ..Default::default()
    }
}

pub fn draw_checkerboard(ui: &mut egui::Ui, rect: egui::Rect, color: egui::Color32) {
    let id = egui::Id::new("checkerboard");
    let handle = ui
        .ctx()
        .data(|data| data.get_temp::<egui::TextureHandle>(id));

    let handle = handle.unwrap_or_else(|| {
        let color_1 = [190, 190, 190, 255];
        let color_2 = [240, 240, 240, 255];

        let pixels = vec![color_1, color_2, color_2, color_1]
            .into_iter()
            .flatten()
            .collect::<Vec<u8>>();

        let texture_options = egui::TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Nearest,
            wrap_mode: egui::TextureWrapMode::Repeat,
            mipmap_mode: None,
        };

        let tex_data = egui::ColorImage::from_rgba_unmultiplied([2, 2], &pixels);

        let handle = ui.ctx().load_texture("checker", tex_data, texture_options);
        ui.ctx().data_mut(|data| {
            data.insert_temp(id, handle.clone());
        });
        handle
    });

    let uv = egui::Rect::from_min_max(
        egui::pos2(0.0, 0.0),
        egui::pos2(rect.width() / 24.0, rect.height() / 24.0),
    );

    ui.painter().image(handle.id(), rect, uv, color);
}

pub fn size_for_splat_view(ui: &mut egui::Ui, with_button: bool) -> egui::Vec2 {
    let mut size = ui.available_size();
    if with_button {
        size.y -= 25.0;
    }
    size.floor()
}
