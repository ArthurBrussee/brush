use burn::tensor::{DType, Scalar, Shape};
use burn_wgpu::{AutoCompiler, WgpuDevice, WgpuRuntime};
use bytemuck::Pod;

pub use burn_cubecl::cubecl::prelude::KernelId;
pub use burn_cubecl::cubecl::{CubeCount, CubeDim, client::ComputeClient, server::ComputeServer};
pub use burn_cubecl::cubecl::{CubeTask, Runtime};
pub use burn_cubecl::{CubeRuntime, tensor::CubeTensor};

// Re-export bytemuck for use by generated code
pub use bytemuck;

use crate::MainBackendBase;

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
        use burn::backend::ops::FloatTensorOps;
        // for tests - make doubly sure we're not accidentally relying on values
        // being initialized to zero by adding in some random noise.
        let f = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape.clone(),
            buffer,
            DType::F32,
        );
        let noised = MainBackendBase::float_add_scalar(f, Scalar::Float(-12345.0));
        buffer = noised.handle;
    }
    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, dtype)
}

/// Upload a slice of POD data to the GPU as a 1D `CubeTensor`.
pub fn create_tensor_from_slice<T: Pod>(
    data: &[T],
    device: &WgpuDevice,
    dtype: DType,
) -> CubeTensor<WgpuRuntime<AutoCompiler>> {
    let client = WgpuRuntime::client(device);
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        Shape::new([data.len()]),
        handle,
        dtype,
    )
}

/// Pin bool tensors to u32 storage.
///
/// burn flips `WgpuDevice`'s default bool storage to u8 as soon as the `metal`
/// feature is on, but cubecl picks the shader compiler at *runtime*: a GPU that
/// can't do native MSL (a paravirtual one, say) falls back to WGSL, which has
/// no u8 type, so kernels fail to compile with `vec4<u8>`. u32 is valid under
/// both compilers, so pin it instead of depending on which one we land on.
///
/// Costs 3 bytes per element on the handful of 1-D masks we keep.
pub fn pin_bool_store(device: &WgpuDevice) {
    use burn::tensor::{BoolDType, Device};
    use std::sync::Once;

    // The setting is global, so do it exactly once: calling it repeatedly
    // while other threads are building tensors races (parallel tests hit it).
    static PIN: Once = Once::new();
    PIN.call_once(|| {
        let mut device: Device = device.clone().into();
        device
            .configure(BoolDType::U32)
            .expect("failed to pin bool storage to u32");
    });
}
