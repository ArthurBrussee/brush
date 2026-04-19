#![recursion_limit = "256"]
mod benches;

fn main() {
    // Mirror brush-process::burn_init_setup so benchmarks use the same backend
    // selection as the running app (DX12 on Windows; Vulkan + wgpu has poor OOM handling).
    #[cfg(target_os = "windows")]
    type SelectedGraphicsApi = burn_wgpu::graphics::Dx12;
    #[cfg(not(target_os = "windows"))]
    type SelectedGraphicsApi = burn_wgpu::graphics::AutoGraphicsApi;

    burn_cubecl::cubecl::future::block_on(burn_wgpu::init_setup_async::<SelectedGraphicsApi>(
        &burn_wgpu::WgpuDevice::DefaultDevice,
        burn_wgpu::RuntimeOptions {
            tasks_max: 64,
            memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
        },
    ));

    divan::main();
}
