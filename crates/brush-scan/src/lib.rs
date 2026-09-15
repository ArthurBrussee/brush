//! GPU scans shared across brush: a cube-wide block scan for kernels that
//! need a prefix sum in shared memory (the radix sort), and a device-wide
//! inclusive prefix sum built on it.

mod kernels;

pub use kernels::{
    BLOCK_SIZE, ELEMENTS_PER_THREAD, WG, block_scan, cube_exclusive_sum, cube_sum, lds_index,
};

use brush_cube::create_tensor;
use burn::backend::TensorMetadata;
use burn::cubecl::CubeDim;
use burn::cubecl::calculate_cube_count_elemwise;
use burn_wgpu::CubeTensor;
use kernels::BLOCK_SIZE_USIZE;

/// Inclusive prefix sum over a contiguous 1D `u32` tensor.
///
/// Each cube scans a block of 1024 elements and emits its total; the totals
/// are scanned recursively the same way, then each level's offsets are added
/// back down. One scan kernel per level plus one add per unwound level,
/// `log_1024(n)` levels.
pub fn prefix_sum(input: CubeTensor) -> CubeTensor {
    assert!(input.is_contiguous(), "Please ensure input is contiguous");

    let num = input.shape()[0];
    if num == 0 {
        return input;
    }

    let client = input.client.clone();
    let device = input.device.clone();
    let dtype = input.dtype;
    let cube_dim = CubeDim::new_1d(WG);

    // Level 0 scans the input; each further level scans the previous level's
    // block sums. `scanned[l]` is the inclusive scan at level `l`.
    let mut scanned: Vec<CubeTensor> = vec![];
    let mut level_input = input;
    let mut level_len = num;
    loop {
        let blocks = level_len.div_ceil(BLOCK_SIZE_USIZE);
        let out = create_tensor([level_len], &device, dtype);
        let sums = create_tensor([blocks], &device, dtype);
        kernels::scan_blocks_kernel::launch(
            &client,
            calculate_cube_count_elemwise(
                &client,
                level_len,
                CubeDim::new_1d(BLOCK_SIZE_USIZE as u32),
            ),
            cube_dim,
            level_input.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            sums.clone().into_tensor_arg(),
        );
        scanned.push(out);
        if blocks == 1 {
            break;
        }
        level_input = sums;
        level_len = blocks;
    }

    // Unwind: the scanned block sums of level l+1 are the offsets for level l.
    for l in (0..scanned.len() - 1).rev() {
        let len = scanned[l].shape()[0];
        kernels::add_block_offsets_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, len, cube_dim),
            cube_dim,
            scanned[l + 1].clone().into_tensor_arg(),
            scanned[l].clone().into_tensor_arg(),
        );
    }

    scanned.swap_remove(0)
}

#[cfg(test)]
mod tests {
    use crate::prefix_sum;
    use brush_cube::{CubeDevice, MainBackendBase, create_tensor_from_slice};
    use burn::backend::TensorMetadata;
    use burn::backend::ops::IntTensorOps;
    use burn::tensor::DType;

    use burn_wgpu::CubeTensor;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn read_i32(tensor: CubeTensor) -> Vec<i32> {
        let data = MainBackendBase::int_into_data(tensor)
            .await
            .expect("readback");
        data.as_slice::<i32>().expect("Wrong type").to_vec()
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sum_tiny() {
        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice(&[1i32, 1, 1, 1], &device, DType::I32);
        let summed = read_i32(prefix_sum(keys)).await;
        assert_eq!(summed.len(), 4);
        assert_eq!(summed, [1, 2, 3, 4]);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_workgroup_multiple() {
        const ITERS: usize = 1024;
        let data: Vec<i32> = (0..ITERS).map(|i| 90 + i as i32).collect();
        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice(&data, &device, DType::I32);
        let summed = read_i32(prefix_sum(keys)).await;
        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();
        for (summed, reff) in summed.iter().zip(prefix_sum_ref) {
            assert_eq!(*summed, reff);
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_empty() {
        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice::<i32>(&[], &device, DType::I32);
        let summed = prefix_sum(keys);
        assert_eq!(summed.shape()[0], 0);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_workgroup_boundaries() {
        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);

        for len in [255usize, 256, 257] {
            let data: Vec<i32> = (0..len).map(|i| (i % 7) as i32).collect();
            let keys = create_tensor_from_slice(&data, &device, DType::I32);
            let summed = read_i32(prefix_sum(keys)).await;
            let expected: Vec<_> = data
                .into_iter()
                .scan(0, |sum, value| {
                    *sum += value;
                    Some(*sum)
                })
                .collect();
            assert_eq!(summed, expected, "length {len}");
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sum() {
        const ITERS: usize = 512 * 16 + 123;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(2 + i as i32);
            data.push(0);
            data.push(32);
            data.push(512);
            data.push(30965);
        }

        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice(&data, &device, DType::I32);
        let summed = read_i32(prefix_sum(keys)).await;

        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();

        for (summed, reff) in summed.iter().zip(prefix_sum_ref) {
            assert_eq!(*summed, reff);
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_block_boundaries() {
        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        // Around one block, two blocks, and the first second-level block.
        for len in [
            1023usize,
            1024,
            1025,
            2048,
            2049,
            1024 * 1024,
            1024 * 1024 + 1,
        ] {
            let data: Vec<i32> = (0..len).map(|i| (i % 13) as i32).collect();
            let keys = create_tensor_from_slice(&data, &device, DType::I32);
            let summed = read_i32(prefix_sum(keys)).await;
            let mut acc = 0;
            for (i, (got, x)) in summed.iter().zip(&data).enumerate() {
                acc += x;
                assert_eq!(*got, acc, "length {len}, index {i}");
            }
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sum_large() {
        // Test with 20M elements to verify 2D dispatch works correctly.
        const NUM_ELEMENTS: usize = 30_000_000;

        // Use small values to avoid overflow in prefix sum
        let data: Vec<i32> = (0..NUM_ELEMENTS).map(|i| (i % 100) as i32).collect();

        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice(&data, &device, DType::I32);
        let summed_slice = read_i32(prefix_sum(keys)).await;

        assert_eq!(summed_slice.len(), NUM_ELEMENTS);

        // First element should equal first input
        assert_eq!(summed_slice[0], data[0]);

        // Check some specific indices
        let check_indices = [0, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 19_999_999];
        for &idx in &check_indices {
            let expected: i32 = data[..=idx].iter().sum();
            assert_eq!(
                summed_slice[idx], expected,
                "Mismatch at index {idx}: got {}, expected {expected}",
                summed_slice[idx]
            );
        }
    }
}
