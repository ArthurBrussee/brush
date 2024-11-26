use brush_viewer::viewer::Viewer;
use eframe::NativeOptions;
use tokio_with_wasm::alias as tokio;
use brush_train::create_wgpu_setup;
use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup};

pub fn create_viewer() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let setup = create_wgpu_setup().await;
        
        let native_options = NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([800.0, 600.0])
                .with_active(true),
            wgpu_options: WgpuConfiguration {
                wgpu_setup: WgpuSetup::Existing {
                    instance: setup.instance,
                    adapter: setup.adapter,
                    device: setup.device,
                    queue: setup.queue,
                },
                ..Default::default()
            },
            ..Default::default()
        };

        eframe::run_native(
            "Brush",
            native_options,
            Box::new(|cc| Ok(Box::new(Viewer::new(cc)))),
        ).unwrap();
    });
}