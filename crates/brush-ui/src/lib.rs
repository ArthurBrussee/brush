use std::sync::Arc;

use burn_wgpu::{RuntimeOptions, WgpuDevice};
use eframe::egui_wgpu::WgpuConfiguration;
use wgpu::{Adapter, Device, Queue};

pub mod burn_texture;

pub fn create_wgpu_device(
    adapter: Arc<Adapter>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> WgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        instance: Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor::default())), // unused... need to fix this in Burn.
        adapter: adapter.clone(),
        device: device.clone(),
        queue: queue.clone(),
    };

    burn_wgpu::init_device(
        setup,
        RuntimeOptions {
            tasks_max: 64,
            memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
        },
    )
}

pub fn create_egui_options() -> WgpuConfiguration {
    WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew {
            supported_backends: wgpu::Backends::all(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            device_descriptor: Arc::new(|adapter: &Adapter| wgpu::DeviceDescriptor {
                label: Some("egui+burn"),
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            }),
        },
        ..Default::default()
    }
}

pub fn draw_checkerboard(ui: &mut egui::Ui, rect: egui::Rect) {
    let id = egui::Id::new("checkerboard");
    let handle = ui
        .ctx()
        .data(|data| data.get_temp::<egui::TextureHandle>(id));

    let handle = if let Some(handle) = handle {
        handle
    } else {
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
    };

    let uv = egui::Rect::from_min_max(
        egui::pos2(0.0, 0.0),
        egui::pos2(rect.width() / 24.0, rect.height() / 24.0),
    );

    ui.painter()
        .image(handle.id(), rect, uv, egui::Color32::WHITE);
}
