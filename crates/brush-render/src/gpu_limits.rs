//! Model limits derived from the device's addressable storage buffers.

use crate::{kernels::helpers::PROJECTED_LANES_USIZE, sh::sh_coeffs_for_degree};
use burn::tensor::Device;

/// Limit the largest per-splat allocation, including the backward pass's
/// extra dummy row. Intersection buffers are checked separately at allocation.
pub fn max_splats(device: &Device, sh_degree: u32, lod: bool) -> u32 {
    let inner = device.clone().inner();
    let burn::backend::DispatchDevice::Cube(device) = inner.as_dispatch() else {
        return u32::MAX;
    };
    max_splats_for_buffer(brush_cube::max_storage_buffer_bytes(device), sh_degree, lod)
}

fn max_splats_for_buffer(bytes: u64, sh_degree: u32, lod: bool) -> u32 {
    let lanes = u64::from(sh_coeffs_for_degree(sh_degree)) * 3;
    let lanes = lanes.max(10).max(PROJECTED_LANES_USIZE as u64);
    // PUP scoring accumulates a 6x6 Hessian per splat before LOD decimation.
    let lanes = if lod { lanes.max(36) } else { lanes };
    (bytes / (lanes * 4))
        .saturating_sub(1)
        .min(u64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_backward_dummy_row_at_the_256_mib_boundary() {
        let cap = max_splats_for_buffer(1 << 28, 3, false);
        assert_eq!(cap, 1_398_100);
        assert!(
            (u64::from(cap) + 1) * 48 * 4 <= 1 << 28,
            "dummy row must fit"
        );
        assert!(
            (u64::from(cap) + 2) * 48 * 4 > 1 << 28,
            "cap must use the available capacity"
        );
    }

    #[test]
    fn all_degrees_fit_the_largest_per_splat_allocation() {
        for degree in 0..=4 {
            let cap = u64::from(max_splats_for_buffer(128 << 20, degree, false));
            for lanes in [
                10,
                PROJECTED_LANES_USIZE as u64,
                u64::from(sh_coeffs_for_degree(degree)) * 3,
            ] {
                assert!(
                    (cap + 1) * lanes * 4 <= 128 << 20,
                    "degree {degree}, lanes {lanes} must fit"
                );
            }
        }
        assert_eq!(max_splats_for_buffer(0, 3, false), 0);
        for degree in 0..=4 {
            let cap = u64::from(max_splats_for_buffer(1 << 28, degree, true));
            assert!(
                (cap + 1) * 36 * 4 <= 1 << 28,
                "degree {degree} LOD Hessian must fit"
            );
        }
    }
}
