use brush_wgsl::wgsl_kernel;

use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
use burn::tensor::{DType, Shape};

pub use burn_cubecl::cubecl::{CubeCount, CubeDim, client::ComputeClient, server::ComputeServer};
pub use burn_cubecl::cubecl::{CubeTask, Runtime};
pub use burn_cubecl::cubecl::{
    prelude::KernelId,
    server::{Bindings, MetadataBinding},
};
pub use burn_cubecl::{CubeRuntime, tensor::CubeTensor};

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
