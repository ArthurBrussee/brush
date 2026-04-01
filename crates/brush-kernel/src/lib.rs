pub mod test_helpers;

use brush_wgsl::wgsl_kernel;

use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
use burn::tensor::{DType, Scalar, Shape};

pub use burn_cubecl::cubecl::prelude::KernelId;
use burn_cubecl::cubecl::server::MetadataBindingInfo;
pub use burn_cubecl::cubecl::{CubeCount, CubeDim, client::ComputeClient, server::ComputeServer};
pub use burn_cubecl::cubecl::{CubeTask, Runtime};
pub use burn_cubecl::{CubeRuntime, tensor::CubeTensor};

use bytemuck::NoUninit;

// Re-export bytemuck for use by brush-wgsl generated code
pub use bytemuck;

// Internal kernel for creating dispatch buffers
#[wgsl_kernel(source = "src/shaders/wg.wgsl")]
struct Wg;

/// Calculate workgroup count for a 1D dispatch, tiling into 2D if needed.
/// Use this for kernels processing a 1D array of elements that may exceed 65535 workgroups.
pub fn calc_cube_count_1d(num_elements: u32, workgroup_size: u32) -> CubeCount {
    let total_wgs = num_elements.div_ceil(workgroup_size);

    // WebGPU limit is 65535 workgroups per dimension.
    if total_wgs > 65535 {
        let wg_y = (total_wgs as f64).sqrt().ceil() as u32;
        let wg_x = total_wgs.div_ceil(wg_y);
        CubeCount::Static(wg_x, wg_y, 1)
    } else {
        CubeCount::Static(total_wgs, 1, 1)
    }
}

pub fn calc_cube_count_3d(sizes: [u32; 3], workgroup_size: [u32; 3]) -> CubeCount {
    let wg_x = sizes[0].div_ceil(workgroup_size[0]);
    let wg_y = sizes[1].div_ceil(workgroup_size[1]);
    let wg_z = sizes[2].div_ceil(workgroup_size[2]);
    CubeCount::Static(wg_x, wg_y, wg_z)
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
        let noised =
            CubeBackend::<WgpuRuntime, f32, i32, u32>::float_add_scalar(f, Scalar::Float(-12345.0));
        buffer = noised.handle;
    }
    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, dtype)
}

pub fn create_meta_binding<T: NoUninit>(val: T) -> MetadataBindingInfo {
    // Copy data to u64. If length of T is not % 8, this will correctly
    // pad with zeros.
    let data: Vec<u64> = bytemuck::pod_collect_to_vec(&[val]);
    MetadataBindingInfo::new(data, 0)
}

/// Create a buffer to use as a shader uniform, from a structure.
pub fn create_uniform_buffer<R: CubeRuntime, T: NoUninit>(
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
