#![recursion_limit = "256"]

mod ui_process;

use std::sync::Arc;

use brush_ui::app::App;

#[allow(clippy::unnecessary_wraps)] // Error isn't need on wasm but that's ok.
fn main() -> Result<(), anyhow::Error> {
    #[cfg(not(target_family = "wasm"))]
    {
        use brush_process::process::process_stream;
        use brush_ui::BrushUiProcess;

        let context = Arc::new(UiProcess::new());
        let wgpu_options = brush_ui::create_egui_options();

        use brush_cli::Cli;
        use clap::Parser;

        let args = Cli::parse().validate()?;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to initialize tokio runtime");

        runtime.block_on(async move {
            env_logger::builder()
                .target(env_logger::Target::Stdout)
                .init();

            if args.with_viewer {
                let icon = eframe::icon_data::from_png_bytes(
                    &include_bytes!("../assets/icon-256.png")[..],
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
                    context.start_new_process(source, args.process);
                }

                let title = if cfg!(debug_assertions) {
                    "Brush  -  Debug"
                } else {
                    "Brush"
                };

                eframe::run_native(
                    title,
                    native_options,
                    Box::new(move |cc| Ok(Box::new(App::new(cc, None, context)))),
                )?;
            } else {
                let Some(source) = args.source else {
                    panic!("Validation of args failed?");
                };
                let device = brush_render::burn_init_setup().await;
                let stream = process_stream(source, args.process.clone(), device);
                brush_cli::process_ui(stream, args.process).await?;
            }
            anyhow::Result::<(), anyhow::Error>::Ok(())
        })?;
    }

    #[cfg(target_family = "wasm")]
    {
        // Allowed to fail. When using the embedding API main canvas just won't be found.
        // Ideally it would catch only _that_ error.
        let _ = wasm_app("main_canvas", None);
    }

    Ok(())
}

#[cfg(target_family = "wasm")]
fn wasm_app(canvas_name: &str, start_uri: Option<&str>) -> anyhow::Result<Arc<UiProcess>> {
    use anyhow::Context;
    use wasm_bindgen::JsCast;

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

    let start_uri = start_uri.map(|x| x.to_owned());

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
                Box::new(|cc| Ok(Box::new(App::new(cc, start_uri, context_cl)))),
            )
            .await
            .expect("failed to start eframe");
    });

    Ok(context)
}

#[cfg(target_family = "wasm")]
mod embedded {
    use super::wasm_app;
    use brush_msg::DataSource;
    use brush_msg::config::ProcessArgs;
    use brush_ui::BrushUiProcess;
    use glam::Quat;
    use tokio::sync::mpsc::UnboundedSender;
    use tokio_with_wasm::alias as tokio_wasm;
    use wasm_bindgen::prelude::*;

    enum EmbeddedCommands {
        LoadDataSource(DataSource),
        SetCamSettings(CameraSettings),
    }

    #[wasm_bindgen]
    pub struct EmbeddedApp {
        command_channel: UnboundedSender<EmbeddedCommands>,
    }

    //Wrapper for wasm world.
    #[wasm_bindgen]
    pub struct CameraSettings(brush_ui::app::CameraSettings);

    #[wasm_bindgen]
    impl CameraSettings {
        #[wasm_bindgen(constructor)]
        pub fn new(
            focal: f64,
            x: f32,
            y: f32,
            z: f32,
            focus_distance: f32,
            speed_scale: f32,
            min_focus_distance: Option<f32>,
            max_focus_distance: Option<f32>,
            min_pitch: Option<f32>,
            max_pitch: Option<f32>,
            min_yaw: Option<f32>,
            max_yaw: Option<f32>,
        ) -> CameraSettings {
            CameraSettings(brush_ui::app::CameraSettings {
                focal,
                // TODO: Allow vecs?
                position: glam::vec3(x, y, z),
                // TODO: How to handle rotations?
                rotation: Quat::IDENTITY,
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
            let context = wasm_app(canvas_name, Some(start_uri)).map_err(|e| JsError::from(&*e))?;

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
}

#[cfg(target_family = "wasm")]
pub use embedded::*;

use ui_process::UiProcess;
