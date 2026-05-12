use burn::backend::wgpu::WgpuDevice;

#[cfg(all(
    not(target_family = "wasm"),
    target_os = "windows",
    target_arch = "aarch64"
))]
type TestGraphicsApi = burn_wgpu::graphics::Dx12;

#[cfg(not(all(
    not(target_family = "wasm"),
    target_os = "windows",
    target_arch = "aarch64"
)))]
type TestGraphicsApi = burn_wgpu::graphics::AutoGraphicsApi;

fn test_runtime_options() -> burn_wgpu::RuntimeOptions {
    burn_wgpu::RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

/// Initialize and return the default GPU device for tests.
///
/// On native, this eagerly selects the graphics API before `CubeCL` can lazily
/// initialize the default device.
pub async fn test_device() -> WgpuDevice {
    #[cfg(not(target_family = "wasm"))]
    {
        use burn_cubecl::cubecl::{
            Runtime,
            ir::{ElemType, FloatKind},
        };
        use burn_wgpu::{WgpuRuntime, graphics::GraphicsApi};

        static INIT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
        INIT.get_or_init(|| async {
            let setup = burn_wgpu::init_setup_async::<TestGraphicsApi>(
                &WgpuDevice::DefaultDevice,
                test_runtime_options(),
            )
            .await;
            let info = setup.adapter.get_info();
            let adapter_features = setup.adapter.features();
            let adapter_has_shader_f16 = format!("{adapter_features:?}").contains("SHADER_F16");
            let client = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
            let cubecl_supports_f16 = client
                .properties()
                .supports_type(ElemType::Float(FloatKind::F16));

            println!(
                "[brush test_device] backend={:?} requested_backend={:?} adapter={:?} driver={:?}",
                setup.backend,
                TestGraphicsApi::backend(),
                info.name,
                info.driver_info
            );
            println!(
                "[brush test_device] adapter_has_shader_f16={adapter_has_shader_f16} cubecl_supports_f16={cubecl_supports_f16}"
            );
        })
        .await;
    }

    #[cfg(target_family = "wasm")]
    {
        use std::sync::Once;
        static INIT: Once = Once::new();
        let mut should_init = false;
        INIT.call_once(|| should_init = true);

        if should_init {
            console_error_panic_hook::set_once();
            wasm_logger::init(wasm_logger::Config::new(log::Level::Warn));

            let setup = burn_wgpu::init_setup_async::<TestGraphicsApi>(
                &WgpuDevice::DefaultDevice,
                test_runtime_options(),
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
        }
    }
    WgpuDevice::DefaultDevice
}

#[cfg(test)]
mod tests {
    use burn_wgpu::graphics::GraphicsApi;

    #[test]
    fn selects_dx12_for_native_windows_arm64_tests() {
        #[cfg(all(
            not(target_family = "wasm"),
            target_os = "windows",
            target_arch = "aarch64"
        ))]
        assert_eq!(
            super::TestGraphicsApi::backend(),
            burn_wgpu::graphics::Dx12::backend()
        );

        #[cfg(not(all(
            not(target_family = "wasm"),
            target_os = "windows",
            target_arch = "aarch64"
        )))]
        assert_eq!(
            super::TestGraphicsApi::backend(),
            burn_wgpu::graphics::AutoGraphicsApi::backend()
        );
    }
}
