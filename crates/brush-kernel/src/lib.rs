use brush_wgsl::wgsl_kernel;

use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
use burn::tensor::{DType, Shape};
pub use burn_cubecl::cubecl::CompilationError;
pub use burn_cubecl::cubecl::prelude::CompiledKernel;
pub use burn_cubecl::cubecl::prelude::ExecutionMode;
pub use burn_cubecl::cubecl::{CubeCount, CubeDim, client::ComputeClient, server::ComputeServer};
pub use burn_cubecl::cubecl::{CubeTask, Runtime};
pub use burn_cubecl::cubecl::{
    prelude::KernelId,
    server::{Bindings, MetadataBinding},
};
pub use burn_cubecl::kernel::KernelMetadata;
pub use burn_cubecl::{CubeRuntime, cubecl::Compiler, tensor::CubeTensor};

use bytemuck::Pod;

// Internal kernel for creating dispatch buffers
#[wgsl_kernel(source = "src/shaders/wg.wgsl")]
struct Wg;

pub fn calc_cube_count<const D: usize>(sizes: [u32; D], workgroup_size: [u32; 3]) -> CubeCount {
    CubeCount::Static(
        sizes.first().unwrap_or(&1).div_ceil(workgroup_size[0]),
        sizes.get(1).unwrap_or(&1).div_ceil(workgroup_size[1]),
        sizes.get(2).unwrap_or(&1).div_ceil(workgroup_size[2]),
    )
}

/// Parse a pre-compiled WGSL string into a naga Module.
pub fn parse_wgsl(source: &str) -> naga::Module {
    naga::front::wgsl::parse_str(source).expect("Failed to parse pre-compiled WGSL")
}

pub fn module_to_compiled<C: Compiler>(
    debug_name: &'static str,
    module: &naga::Module,
    workgroup_size: [u32; 3],
) -> Result<CompiledKernel<C>, CompilationError> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::empty(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .expect("Failed to compile"); // Ideally this would err but seems hard but with current CubeCL.

    let shader_string =
        naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
            .expect("failed to convert naga module to source");

    // Dawn annoyingly wants some extra syntax to enable subgroups,
    // so just hack this in when running on wasm.
    #[cfg(target_family = "wasm")]
    // Ideally include all subgroup functions here but meh.
    let shader_string = if shader_string.contains("subgroupAdd")
        || shader_string.contains("subgroupAny")
        || shader_string.contains("subgroupMax")
        || shader_string.contains("subgroupBroadcast")
        || shader_string.contains("subgroupShuffle")
    {
        "enable subgroups;\n".to_owned() + &shader_string
    } else {
        shader_string
    };

    Ok(CompiledKernel {
        entrypoint_name: "main".to_owned(),
        debug_name: Some(debug_name),
        source: shader_string,
        repr: None,
        cube_dim: CubeDim::new(workgroup_size[0], workgroup_size[1], workgroup_size[2]),
        debug_info: None,
    })
}

pub fn calc_kernel_id<T: 'static>(values: &[bool]) -> KernelId {
    let kernel_id = KernelId::new::<T>();
    kernel_id.info(values.to_vec())
}

// Reserve a buffer from the client for the given shape.
pub fn create_tensor<const D: usize>(
    shape: [usize; D],
    device: &WgpuDevice,
    dtype: DType,
) -> CubeTensor<WgpuRuntime> {
    let client = WgpuRuntime::client(device);

    let shape = Shape::from(shape.to_vec());
    let bufsize = shape.num_elements() * dtype.size();
    let mut buffer = client.empty(bufsize);

    if cfg!(test) {
        use burn::tensor::ops::FloatTensorOps;
        use burn_cubecl::CubeBackend;
        // for tests - make doubly sure we're not accidentally relying on values
        // being initialized to zero by adding in some random noise.
        let f = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape.clone(),
            buffer,
            DType::F32,
        );
        let noised = CubeBackend::<WgpuRuntime, f32, i32, u32>::float_add_scalar(f, -12345.0);
        buffer = noised.handle;
    }
    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, dtype)
}

pub fn create_meta_binding<T: Pod>(val: T) -> MetadataBinding {
    // Copy data to u32. If length of T is not % 4, this will correctly
    // pad with zeros.
    let data = bytemuck::pod_collect_to_vec(&[val]);
    MetadataBinding {
        static_len: data.len(),
        data,
    }
}

/// Create a buffer to use as a shader uniform, from a structure.
pub fn create_uniform_buffer<R: CubeRuntime, T: Pod>(
    val: T,
    device: &R::Device,
    client: &ComputeClient<R>,
) -> CubeTensor<R> {
    let binding = create_meta_binding(val);
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([binding.data.len()]),
        client.create_from_slice(bytemuck::cast_slice(&binding.data)),
        DType::I32,
    )
}

pub fn create_dispatch_buffer(
    thread_nums: CubeTensor<WgpuRuntime>,
    wg_size: [u32; 3],
) -> CubeTensor<WgpuRuntime> {
    assert!(
        thread_nums.is_contiguous(),
        "Thread nums should be contiguous"
    );
    let client = thread_nums.client;
    let ret = create_tensor([3], &thread_nums.device, DType::I32);

    let data = create_meta_binding(wg::Uniforms {
        wg_size_x: wg_size[0] as i32,
        wg_size_y: wg_size[1] as i32,
        wg_size_z: wg_size[2] as i32,
    });

    // SAFETY: wgsl FFI, kernel checked to have no OOB, bounded loops.
    unsafe {
        client
            .launch_unchecked(
                Wg::task(),
                CubeCount::Static(1, 1, 1),
                Bindings::new()
                    .with_buffers(vec![
                        thread_nums.handle.binding(),
                        ret.handle.clone().binding(),
                    ])
                    .with_metadata(data),
            )
            .expect("Failed to execute");
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyKernel;

    #[test]
    fn test_kernel_id_calculation() {
        // Test kernel ID generation with different boolean flags
        let id1 = calc_kernel_id::<DummyKernel>(&[true, false]);
        let id2 = calc_kernel_id::<DummyKernel>(&[false, true]);
        let id3 = calc_kernel_id::<DummyKernel>(&[true, false]);

        // Same flags should produce same ID
        assert_eq!(id1, id3);
        // Different flags should produce different IDs
        assert_ne!(id1, id2);

        // Empty flags should work
        let id_empty = calc_kernel_id::<DummyKernel>(&[]);
        assert_ne!(id_empty, id1);
    }
}
