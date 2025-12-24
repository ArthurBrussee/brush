//! Brush kernel infrastructure for WGSL compute shaders.
//!
//! This crate provides the infrastructure for integrating WGSL kernels with `cubecl`:
//!
//! - `#[wgsl_kernel]` proc macro for generating kernel wrappers from WGSL source files
//! - Re-exports of `bytemuck` for POD types used in uniform buffers
//! - Re-exports of cubecl types needed
//!
//! # Example
//!
//! ```ignore
//! use brush_kernel::wgsl_kernel;
//!
//! #[wgsl_kernel(source = "src/shaders/my_kernel.wgsl")]
//! pub struct MyKernel;
//! ```
use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
use burn::tensor::{DType, Shape};
use burn_cubecl::cubecl::Runtime;
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::tensor::CubeTensor;

// Re-export the proc macro
pub use brush_kernel_proc::wgsl_kernel;

// Re-export bytemuck for use by generated code and users
pub use bytemuck;

// Re-export cubecl types needed for kernel launching
pub use burn_cubecl::cubecl::CubeCount;
pub use burn_cubecl::cubecl::server::MetadataBinding;

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

/// Reserve a buffer from the client for the given shape.
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

/// Create a dynamic dispatch buffer for 1D dispatches.
/// Returns a buffer with (`wg_x`, `wg_y`, 1) that tiles into 2D if needed.
pub fn create_dispatch_buffer_1d(
    thread_count: CubeTensor<WgpuRuntime>,
    wg_size: u32,
) -> CubeTensor<WgpuRuntime> {
    assert!(
        thread_count.is_contiguous(),
        "Thread count should be contiguous"
    );
    let client = thread_count.client;
    let ret = create_tensor([3], &thread_count.device, DType::I32);

    // SAFETY: wgsl FFI, kernel checked to have no OOB, bounded loops.
    unsafe {
        client
            .launch_unchecked(
                Wg::task(),
                CubeCount::Static(1, 1, 1),
                Bindings::new()
                    .with_buffers(vec![
                        thread_count.handle.binding(),
                        ret.handle.clone().binding(),
                    ])
                    .with_metadata(wg::Uniforms { wg_size }.to_meta_binding()),
            )
            .expect("Failed to execute");
    }

    ret
}
