#[allow(unused)]
use brush_app::{App, AppCreateCb};

use brush_process::process_loop::start_process;
#[allow(unused)]
use tokio::sync::oneshot::error::RecvError;

fn main() -> Result<(), clap::Error> {
    let wgpu_options = brush_ui::create_egui_options();

    #[allow(unused)]
    let (send, rec) = tokio::sync::oneshot::channel();

    #[cfg(not(target_family = "wasm"))]
    {
        use brush_cli::Cli;
        use clap::Parser;

        let args = Cli::parse().validate()?;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to intitialize tokio runtime");

        runtime.block_on(async {
            env_logger::init();

            if args.with_viewer {
                // NB: Load carrying icon. egui at head fails when no icon is included
                // as the built-in one is git-lfs which cargo doesn't clone properly.
                let icon = eframe::icon_data::from_png_bytes(
                    &include_bytes!("../../assets/icon-256.png")[..],
                )
                .expect("Failed to load icon");

                let native_options = eframe::NativeOptions {
                    // Build app display.
                    viewport: egui::ViewportBuilder::default()
                        .with_inner_size(egui::Vec2::new(1450.0, 1200.0))
                        .with_active(true)
                        .with_icon(std::sync::Arc::new(icon)),
                    wgpu_options,
                    ..Default::default()
                };

                if let Some(source) = args.source {
                    tokio::spawn(async move {
                        let context: Result<AppCreateCb, RecvError> = rec.await;
                        if let Ok(context) = context {
                            let mut context = context.context.write().expect("Lock poisoned");
                            let process =
                                start_process(source, args.process, context.device.clone());
                            context.connect_to(process);
                        }
                    });
                }

                eframe::run_native(
                    "Brush",
                    native_options,
                    Box::new(move |cc| Ok(Box::new(App::new(cc, send)))),
                )
                .expect("Failed to run egui app");
            } else {
                let Some(source) = args.source else {
                    panic!("Validation of args failed?");
                };

                let process =
                    start_process(source, args.process, burn_wgpu::WgpuDevice::DefaultDevice);
                brush_cli::ui::process_ui(process).await;
            }
        });
    }

    #[cfg(target_family = "wasm")]
    {
        use tokio_with_wasm::alias as tokio_wasm;
        use wasm_bindgen::JsCast;

        if cfg!(debug_assertions) {
            eframe::WebLogger::init(log::LevelFilter::Debug).ok();
        }

        let document = web_sys::window()
            .expect("Failed to find web window (not running in a browser?")
            .document()
            .expect("Failed to find document body");

        if let Some(canvas) = document
            .get_element_by_id("main_canvas")
            .and_then(|x| x.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        {
            // On wasm, run as a local task.
            tokio_wasm::task::spawn(async {
                let web_options = eframe::WebOptions {
                    wgpu_options,
                    ..Default::default()
                };

                eframe::WebRunner::new()
                    .start(
                        canvas,
                        web_options,
                        Box::new(|cc| Ok(Box::new(App::new(cc, send)))),
                    )
                    .await
                    .expect("failed to start eframe");
            });
        }
    }

    Ok(())
}

#[cfg(target_family = "wasm")]
mod embedded {
    use super::*;
    use brush_app::App;
    use brush_process::{data_source::DataSource, process_loop::ProcessArgs};
    use std::future::IntoFuture;
    use tokio::sync::mpsc::UnboundedSender;
    use tokio_with_wasm::alias as tokio_wasm;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct EmbeddedApp {
        command_channel: UnboundedSender<DataSource>,
    }

    #[wasm_bindgen]
    impl EmbeddedApp {
        #[wasm_bindgen(constructor)]
        pub fn new(canvas_name: &str, url: &str) -> Self {
            let wgpu_options = brush_ui::create_egui_options();
            let document = web_sys::window().unwrap().document().unwrap();
            let canvas = document
                .get_element_by_id(canvas_name)
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();

            let url = url.to_owned();
            let (send, rec) = tokio::sync::oneshot::channel();

            let (cmd_send, mut cmd_rec) = tokio::sync::mpsc::unbounded_channel();

            // On wasm, run as a local task.
            tokio_wasm::spawn(async {
                eframe::WebRunner::new()
                    .start(
                        canvas,
                        eframe::WebOptions {
                            wgpu_options,
                            ..Default::default()
                        },
                        Box::new(|cc| Ok(Box::new(App::new(cc, send)))),
                    )
                    .await
                    .expect("failed to start eframe");
            });

            tokio_wasm::spawn(async move {
                let context = rec.into_future().await.unwrap().context;

                while let Some(source) = cmd_rec.recv().await {
                    let mut ctx = context.write().unwrap();
                    let process = start_process(source, ProcessArgs::default(), ctx.device.clone());
                    ctx.connect_to(process);
                }
            });
            // Load initial url.
            let _ = cmd_send.send(DataSource::Url(url.to_owned()));
            Self {
                command_channel: cmd_send,
            }
        }

        #[wasm_bindgen]
        pub fn load_url(&self, url: &str) {
            self.command_channel
                .send(DataSource::Url(url.to_owned()))
                .expect("Viewer was closed?");
        }
    }
}

#[cfg(target_family = "wasm")]
pub use embedded::*;
