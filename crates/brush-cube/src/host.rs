use burn::tensor::{DType, Scalar, Shape};
use bytemuck::Pod;

pub use burn::cubecl::prelude::KernelId;
pub use burn::cubecl::{CubeCount, CubeDim, client::Client};
pub use burn_cubecl::{CubeDevice, tensor::CubeTensor};

// Re-export bytemuck for use by generated code
pub use bytemuck;

// Reserve a buffer from the client for the given shape.
pub fn create_tensor<const D: usize>(
    shape: [usize; D],
    device: &CubeDevice,
    dtype: DType,
) -> CubeTensor {
    let client = device.client();

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
        let noised = burn_cubecl::CubeBackend::float_add_scalar(f, Scalar::Float(-12345.0));
        buffer = noised.handle;
    }
    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, dtype)
}

/// Upload a slice of POD data to the GPU as a 1D `CubeTensor`.
pub fn create_tensor_from_slice<T: Pod>(
    data: &[T],
    device: &CubeDevice,
    dtype: DType,
) -> CubeTensor {
    let client = device.client();
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        Shape::new([data.len()]),
        handle,
        dtype,
    )
}
