mod kernels;

use brush_cube::calc_cube_count_1d;
use brush_cube::create_tensor;
use burn::backend::TensorMetadata;
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::CubeTensor;
use burn_wgpu::WgpuRuntime;

use kernels::THREADS_PER_GROUP;

pub fn prefix_sum(input: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    assert!(input.is_contiguous(), "Please ensure input is contiguous");

    let threads_per_group = THREADS_PER_GROUP as usize;
    let num = input.shape()[0];
    let client = input.client.clone();
    let outputs = create_tensor(input.shape().dims::<1>(), &input.device, input.dtype);

    let cube_dim = CubeDim::new_1d(THREADS_PER_GROUP);

    // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
    unsafe {
        kernels::prefix_sum_scan_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num as u32, THREADS_PER_GROUP),
            cube_dim,
            input.into_tensor_arg(),
            outputs.clone().into_tensor_arg(),
        );
    }

    if num <= threads_per_group {
        return outputs;
    }

    let mut group_buffer = vec![];
    let mut work_size = vec![];
    let mut work_sz = num;
    while work_sz > threads_per_group {
        work_sz = work_sz.div_ceil(threads_per_group);
        group_buffer.push(create_tensor([work_sz], &outputs.device, outputs.dtype));
        work_size.push(work_sz);
    }

    // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
    unsafe {
        kernels::prefix_sum_scan_sums_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(work_size[0] as u32, THREADS_PER_GROUP),
            cube_dim,
            outputs.clone().into_tensor_arg(),
            group_buffer[0].clone().into_tensor_arg(),
        );
    }

    for l in 0..(group_buffer.len() - 1) {
        // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
        unsafe {
            kernels::prefix_sum_scan_sums_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(work_size[l + 1] as u32, THREADS_PER_GROUP),
                cube_dim,
                group_buffer[l].clone().into_tensor_arg(),
                group_buffer[l + 1].clone().into_tensor_arg(),
            );
        }
    }

    for l in (1..group_buffer.len()).rev() {
        let work_sz = work_size[l - 1];

        // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
        unsafe {
            kernels::prefix_sum_add_scanned_sums_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(work_sz as u32, THREADS_PER_GROUP),
                cube_dim,
                group_buffer[l].clone().into_tensor_arg(),
                group_buffer[l - 1].clone().into_tensor_arg(),
            );
        }
    }

    // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
    unsafe {
        kernels::prefix_sum_add_scanned_sums_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            calc_cube_count_1d((work_size[0] * threads_per_group) as u32, THREADS_PER_GROUP),
            cube_dim,
            group_buffer[0].clone().into_tensor_arg(),
            outputs.clone().into_tensor_arg(),
        );
    }

    outputs
}

#[cfg(test)]
mod tests {
    use crate::prefix_sum;
    use brush_cube::create_tensor_from_slice;
    use burn::backend::ops::IntTensorOps;
    use burn::tensor::DType;
    use burn_cubecl::CubeBackend;
    use burn_wgpu::{CubeTensor, WgpuRuntime};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    async fn read_i32(tensor: CubeTensor<WgpuRuntime>) -> Vec<i32> {
        let data = Backend::int_into_data(tensor).await.expect("readback");
        data.as_slice::<i32>().expect("Wrong type").to_vec()
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sum_tiny() {
        let device = brush_cube::test_helpers::test_device().await;
        let keys = create_tensor_from_slice(&[1i32, 1, 1, 1], &device, DType::I32);
        let summed = read_i32(prefix_sum(keys)).await;
        assert_eq!(summed.len(), 4);
        assert_eq!(summed, [1, 2, 3, 4]);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_512_multiple() {
        const ITERS: usize = 1024;
        let data: Vec<i32> = (0..ITERS).map(|i| 90 + i as i32).collect();
        let device = brush_cube::test_helpers::test_device().await;
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

        let device = brush_cube::test_helpers::test_device().await;
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
    async fn test_sum_large() {
        // Test with 20M elements to verify 2D dispatch works correctly.
        const NUM_ELEMENTS: usize = 30_000_000;

        // Use small values to avoid overflow in prefix sum
        let data: Vec<i32> = (0..NUM_ELEMENTS).map(|i| (i % 100) as i32).collect();

        let device = brush_cube::test_helpers::test_device().await;
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
