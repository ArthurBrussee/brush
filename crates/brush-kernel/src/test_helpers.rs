use burn::backend::wgpu::WgpuDevice;

/// Initialize and return the default GPU device for tests.
///
/// On native, the device initializes lazily, so this just returns it.
pub async fn test_device() -> WgpuDevice {
    #[cfg(target_family = "wasm")]
    {
        use std::cell::Cell;
        thread_local! {
            static INITIALIZED: Cell<bool> = const { Cell::new(false) };
        }
        if !INITIALIZED.with(|c| c.get()) {
            console_error_panic_hook::set_once();
            // Initialise once. Errors from a second init call mean the
            // logger is already installed by another test crate; ignore.
            let _ = wasm_logger::init(wasm_logger::Config::new(log::Level::Warn));

            let setup = burn_wgpu::init_setup_async::<burn_wgpu::graphics::AutoGraphicsApi>(
                &WgpuDevice::DefaultDevice,
                burn_wgpu::RuntimeOptions {
                    tasks_max: 64,
                    memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
                },
            )
            .await;
            setup.device.on_uncaptured_error(std::sync::Arc::new(|err| {
                // Panic so wasm-bindgen-test prints the WebGPU error in the
                // failing test's output instead of just the browser console.
                panic!("WebGPU uncaptured error: {err}");
            }));
            // Log adapter info + features at warn level so they show up in
            // every test run's captured output. When a kernel dispatch
            // fails on CI with a "feature not allowed" / "extension not
            // allowed" message, the first thing to look for is whether
            // that feature appears in this list.
            let info = setup.adapter.get_info();
            log::warn!(
                "[brush test_device] adapter: {:?} backend={:?} driver={:?}\n  features: {:?}\n  limits: {:?}",
                info.name,
                info.backend,
                info.driver_info,
                setup.device.features(),
                setup.device.limits(),
            );
            INITIALIZED.with(|c| c.set(true));
        }
    }
    WgpuDevice::DefaultDevice
}
