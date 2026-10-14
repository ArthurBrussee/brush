//! Standalone diagnostic for storage-buffer address wraparound on Windows Adreno.
//! Run with `cargo run --release -p brush-process --example adreno_buffer_boundary`.
//! Uses 256 MiB of GPU memory; no dataset is needed. A correct driver prints PASS.

#[cfg(not(target_os = "windows"))]
fn main() {
    println!("This diagnostic requires the Windows DirectX 12 backend.");
}

#[cfg(target_os = "windows")]
fn main() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(probe());
}

#[cfg(target_os = "windows")]
async fn probe() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    descriptor.backends = wgpu::Backends::DX12;
    println!(
        "Compiler: {:?}",
        descriptor.backend_options.dx12.shader_compiler
    );
    let instance = wgpu::Instance::new(descriptor);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    println!("Adapter: {:?}", adapter.get_info());
    let boundary = 1u64 << 28;
    let size = boundary + 256;
    let limits = wgpu::Limits {
        max_buffer_size: size,
        max_storage_buffer_binding_size: size,
        ..Default::default()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        })
        .await
        .unwrap();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("256 MiB address boundary"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 48,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var<storage, read_write> data: array<u32>;
            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[67108863u + id.x] = 11u * (id.x + 1u);
            }
        "#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(3, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, 16);
    encoder.copy_buffer_to_buffer(&buffer, boundary - 16, &staging, 16, 32);
    queue.submit([encoder.finish()]);
    staging.map_async(wgpu::MapMode::Read, .., |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = staging.get_mapped_range(..).unwrap();
    let actual: Vec<u32> = mapped
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected = [0, 0, 0, 0, 0, 0, 0, 11, 22, 33, 0, 0];
    println!("Readback: {actual:?}");
    assert_eq!(
        actual, expected,
        "GPU shader addresses wrapped at 256 MiB; the writes must not change elements 0 and 1"
    );
    println!("PASS: GPU addresses do not wrap at 256 MiB");
}
