#![recursion_limit = "256"]

mod shared;

use brush_ui::app::App;

use brush_cli::Cli;
use clap::Parser;

use crate::shared::startup;

#[cfg(target_family = "windows")]
fn is_console() -> bool {
    let mut buffer = [0u32; 1];

    // SAFETY: FFI, buffer is large enough.
    unsafe {
        use winapi::um::wincon::GetConsoleProcessList;
        let count = GetConsoleProcessList(buffer.as_mut_ptr(), 1);
        count != 1
    }
}

#[allow(clippy::unnecessary_wraps)] // Error isn't need on wasm but that's ok.
fn main() -> Result<(), anyhow::Error> {
    let args = Cli::parse().validate()?;

    startup();

    #[cfg(target_family = "windows")]
    if args.with_viewer && !is_console() {
        // Hide the console window on windows when running as a GUI.
        // SAFETY: FFI.
        unsafe {
            winapi::um::wincon::FreeConsole();
        };
    }

    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;

        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
        )
        .expect("Failed to set tracing subscriber");
    }

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime")
        .block_on(async move {
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
                    wgpu_options: brush_ui::create_egui_options(),
                    persist_window: true,
                    ..Default::default()
                };

                let title = if cfg!(debug_assertions) {
                    "Brush  -  Debug"
                } else {
                    "Brush"
                };

                eframe::run_native(
                    title,
                    native_options,
                    Box::new(move |cc| {
                        Ok(Box::new(App::new(cc, Some(args.train_stream), args.source)))
                    }),
                )?;
            } else {
                let Some(source) = args.source else {
                    panic!("Validation of args failed?");
                };
                let device = brush_render::burn_init_setup().await;
                brush_cli::run_cli_ui(source, args.train_stream, device).await?;
            }

            anyhow::Result::<(), anyhow::Error>::Ok(())
        })?;

    Ok(())
}
