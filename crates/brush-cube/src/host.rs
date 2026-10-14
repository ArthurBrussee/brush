use burn::tensor::{DType, Scalar, Shape};
use bytemuck::Pod;

pub use burn::cubecl::prelude::KernelId;
pub use burn::cubecl::{CubeCount, CubeDim, client::Client};
pub use burn_cubecl::{CubeDevice, tensor::CubeTensor};

// Re-export bytemuck for use by generated code
pub use bytemuck;

/// Largest storage buffer that can be addressed safely on this device.
///
/// The Windows Adreno driver advertises nearly 2 GiB, but dynamic shader
/// addresses wrap at 256 MiB (reproduced on X1-45, driver 31.0.148.0, with
/// both FXC and DXC). Upload/readback alone does not reveal the corruption.
pub fn max_storage_buffer_bytes(device: &CubeDevice) -> u64 {
    let client = device.client();
    let props = client.properties();
    let vendor = props.identity.physical.as_ref().and_then(|p| p.vendor);
    storage_buffer_limit(
        props.memory.max_page_size,
        vendor,
        cfg!(target_os = "windows"),
    )
}

fn storage_buffer_limit(
    advertised: u64,
    vendor: Option<burn::cubecl::ir::PciVendor>,
    windows: bool,
) -> u64 {
    use burn::cubecl::ir::PciVendor;
    // DXGI uses Microsoft's four-character Qualcomm vendor ID; Vulkan uses
    // the PCI vendor ID. Keep other vendors/platforms at their reported limit.
    if windows
        && matches!(
            vendor,
            Some(PciVendor::Qualcomm | PciVendor::Other(0x4d4f4351))
        )
    {
        advertised.min(1 << 28)
    } else {
        advertised
    }
}

fn check_buffer_size(bytes: usize, device: &CubeDevice) {
    let limit = max_storage_buffer_bytes(device);
    assert!(
        bytes as u64 <= limit,
        "GPU buffer needs {bytes} bytes, exceeding this device's safe {limit}-byte limit. Reduce splat count or image resolution."
    );
}

// Reserve a buffer from the client for the given shape.
pub fn create_tensor<const D: usize>(
    shape: [usize; D],
    device: &CubeDevice,
    dtype: DType,
) -> CubeTensor {
    let client = device.client();

    let shape = Shape::from(shape.to_vec());
    let bufsize = shape.num_elements() * dtype.size();
    check_buffer_size(bufsize, device);
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
    check_buffer_size(std::mem::size_of_val(data), device);
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        Shape::new([data.len()]),
        handle,
        dtype,
    )
}

#[cfg(test)]
mod buffer_limit_tests {
    use super::storage_buffer_limit;
    use burn::cubecl::ir::PciVendor;

    #[test]
    fn windows_qualcomm_limit_does_not_raise_smaller_limits() {
        for vendor in [PciVendor::Qualcomm, PciVendor::Other(0x4d4f4351)] {
            assert_eq!(storage_buffer_limit(2 << 30, Some(vendor), true), 1 << 28);
            assert_eq!(storage_buffer_limit(1 << 27, Some(vendor), true), 1 << 27);
            assert_eq!(storage_buffer_limit(2 << 30, Some(vendor), false), 2 << 30);
        }
        assert_eq!(
            storage_buffer_limit(2 << 30, Some(PciVendor::Nvidia), true),
            2 << 30
        );
        assert_eq!(storage_buffer_limit(2 << 30, None, true), 2 << 30);
    }
}
