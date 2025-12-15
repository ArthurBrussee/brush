#![recursion_limit = "256"]

pub mod app;
pub mod burn_texture;
pub mod camera_controls;

pub mod ui_process;

mod panels;
mod scene;
mod settings;
#[cfg(feature = "training")]
mod stats;
mod widget_3d;

#[cfg(feature = "training")]
mod datasets;

#[cfg(feature = "training")]
mod settings_popup;

use eframe::egui_wgpu::WgpuConfiguration;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wgpu::{Adapter, Features};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[wasm_bindgen]
pub enum UiMode {
    // Default UI with data loading, stats view etc.
    #[default]
    Default,
    // Show the splat fullscreen, but allow toggling, and control panels.
    FullScreenSplat,
    // Render as an embedded viewer which does nothing except render the splat.
    EmbeddedViewer,
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
