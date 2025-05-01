#![recursion_limit = "256"]
mod ui_process;

#[cfg(target_family = "wasm")]
mod wasm;

#[allow(clippy::unnecessary_wraps)] // Error isn't need on wasm but that's ok.
fn main() -> Result<(), anyhow::Error> {
    #[cfg(not(target_family = "wasm"))]
    {
        use brush_process::process::process_stream;
        use brush_ui::BrushUiProcess;
        use brush_ui::app::App;

        let context = std::sync::Arc::new(ui_process::UiProcess::new(brush_ui::UiMode::Full));
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
                    Box::new(move |cc| Ok(Box::new(App::new(cc, context)))),
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
        let level = if cfg!(debug_assertions) {
            // Could do 'debug' but it's way too spammy.
            log::Level::Info
        } else {
            log::Level::Warn
        };
        wasm_log::init(wasm_log::Config::new(level));

        let start_uri = web_sys::window().and_then(|w| w.location().search().ok());

        // Allowed to fail. When using the embedding API main canvas just won't be found.
        // Ideally it would catch only _that_ error.
        let _ = wasm::wasm_app("main_canvas", start_uri.as_deref().unwrap_or(""));
    }

    Ok(())
}
