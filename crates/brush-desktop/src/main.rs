#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use tokio_with_wasm::alias as tokio;

fn main() {
    let wgpu_options = brush_ui::create_egui_options();

    #[cfg(not(target_family = "wasm"))]
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async {
            env_logger::init();

            // NB: Load carrying icon. egui at head fails when no icon is included
            // as the built-in one is git-lfs which cargo doesn't clone properly.
            let icon =
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .unwrap();

            let native_options = eframe::NativeOptions {
                // Build app display.
                viewport: egui::ViewportBuilder::default()
                    .with_inner_size(egui::Vec2::new(1450.0, 900.0))
                    .with_active(true)
                    .with_icon(std::sync::Arc::new(icon)),
                wgpu_options,
                ..Default::default()
            };

            eframe::run_native(
                "Brush",
                native_options,
                Box::new(move |cc| Ok(Box::new(brush_viewer::viewer::Viewer::new(cc)))),
            )
            .unwrap();
        });
    }

    #[cfg(target_family = "wasm")]
    {
        use wasm_bindgen::JsCast;
        eframe::WebLogger::init(log::LevelFilter::Debug).ok();

        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id("main_canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        // On wasm, run as a local task.
        tokio::spawn(async {
            let web_options = eframe::WebOptions {
                wgpu_options,
                ..Default::default()
            };

            eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|cc| Ok(Box::new(brush_viewer::viewer::Viewer::new(cc)))),
                )
                .await
                .expect("failed to start eframe");
        });
    }
}
