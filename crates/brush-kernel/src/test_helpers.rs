use burn::backend::wgpu::WgpuDevice;

/// Initialize and return the default GPU device for tests.
///
/// On WASM, this performs the required async GPU init (only once).
/// On native, the device initializes lazily, so this just returns it.
pub async fn test_device() -> WgpuDevice {
    #[cfg(target_family = "wasm")]
    {
        use std::cell::Cell;
        thread_local! {
            static INITIALIZED: Cell<bool> = const { Cell::new(false) };
        }
        if !INITIALIZED.with(|c| c.get()) {
            burn_wgpu::init_setup_async::<burn_wgpu::graphics::AutoGraphicsApi>(
                &WgpuDevice::DefaultDevice,
                burn_wgpu::RuntimeOptions {
                    tasks_max: 64,
                    memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
                },
            )
            .await;
            INITIALIZED.with(|c| c.set(true));
        }
    }
    WgpuDevice::DefaultDevice
}
