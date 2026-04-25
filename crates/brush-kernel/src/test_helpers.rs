use burn::backend::wgpu::WgpuDevice;

/// Initialize and return the default GPU device for tests.
pub async fn test_device() -> WgpuDevice {
    use std::sync::Once;
    static INIT: Once = Once::new();
    let mut should_init = false;
    INIT.call_once(|| should_init = true);

    if should_init {
        #[cfg(target_family = "wasm")]
        {
            console_error_panic_hook::set_once();
            wasm_logger::init(wasm_logger::Config::new(log::Level::Warn));
        }

        let setup = burn_wgpu::init_setup_async::<burn_wgpu::graphics::AutoGraphicsApi>(
            &WgpuDevice::DefaultDevice,
            burn_wgpu::RuntimeOptions {
                tasks_max: 64,
                memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
            },
        )
        .await;
        #[cfg(target_family = "wasm")]
        setup.device.on_uncaptured_error(std::sync::Arc::new(|err| {
            // Panic so wasm-bindgen-test prints the WebGPU error in the
            // failing test's output instead of just the browser console.
            panic!("WebGPU uncaptured error: {err}");
        }));
        // Log adapter info + features so they show up in every test run's
        // captured output. When a kernel dispatch fails on CI with a "feature
        // not allowed" / "extension not allowed" message, the first thing to
        // look for is whether that feature appears in this list. Same shape
        // on native and wasm so we can compare CI runs across targets.
        let info = setup.adapter.get_info();
        let line = format!(
            "[brush test_device] adapter: {:?} backend={:?} driver={:?}\n  features: {:?}\n  limits: {:?}",
            info.name,
            info.backend,
            info.driver_info,
            setup.device.features(),
            setup.device.limits(),
        );
        #[cfg(target_family = "wasm")]
        log::warn!("{line}");
        #[cfg(not(target_family = "wasm"))]
        eprintln!("{line}");
    }
    WgpuDevice::DefaultDevice
}
