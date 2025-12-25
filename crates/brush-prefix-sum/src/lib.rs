use brush_kernel::{
    CubeCount, calc_cube_count_1d, create_dispatch_buffer_1d, create_tensor, wgsl_kernel,
};
use burn::tensor::DType;
use burn_cubecl::cubecl::server::Bindings;
use burn_wgpu::CubeTensor;
use burn_wgpu::WgpuRuntime;

// Generate shared types and constants from helpers (no entry point)
#[wgsl_kernel(source = "src/shaders/prefix_sum_helpers.wgsl")]
pub struct PrefixSumHelpers;

// Kernel definitions using proc macro
#[wgsl_kernel(source = "src/shaders/prefix_sum_scan.wgsl")]
pub struct PrefixSumScan;

#[wgsl_kernel(source = "src/shaders/prefix_sum_scan_sums.wgsl")]
pub struct PrefixSumScanSums;

#[wgsl_kernel(source = "src/shaders/prefix_sum_add_scanned_sums.wgsl")]
pub struct PrefixSumAddScannedSums;

/// Compute prefix sum with a dynamic length stored in a GPU buffer.
///
/// This is more efficient when the actual number of elements to sum is much smaller
/// than the buffer size, as it avoids processing unused elements.
///
/// # Arguments
/// * `input` - The input buffer (must be at least `length` elements)
/// * `length` - A GPU buffer containing a single u32 with the number of elements to process
/// * `max_length` - Upper bound on the length (used for allocating intermediate buffers)
pub fn prefix_sum_with_length(
    input: CubeTensor<WgpuRuntime>,
    length: CubeTensor<WgpuRuntime>,
    max_length: usize,
) -> CubeTensor<WgpuRuntime> {
    assert!(input.is_contiguous(), "Please ensure input is contiguous");
    assert!(length.is_contiguous(), "Please ensure length is contiguous");

    let threads_per_group = prefix_sum_helpers::THREADS_PER_GROUP as usize;
    let client = &input.client;
    let outputs = create_tensor(input.shape.dims::<1>(), &input.device, DType::I32);

    // Create dynamic dispatch buffer based on length
    let dispatch_wg = create_dispatch_buffer_1d(length.clone(), PrefixSumScan::WORKGROUP_SIZE[0]);

    // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
    unsafe {
        client
            .launch_unchecked(
                PrefixSumScan::task(),
                CubeCount::Dynamic(dispatch_wg.handle.binding()),
                Bindings::new().with_buffers(vec![
                    input.handle.binding(),
                    outputs.handle.clone().binding(),
                    length.handle.clone().binding(),
                ]),
            )
            .expect("Failed to run prefix sums");
    }

    if max_length <= threads_per_group {
        return outputs;
    }

    // For the hierarchical passes, we need to compute the number of groups
    // and create appropriate length buffers for each level.
    // We allocate based on max_length to ensure buffers are large enough.
    let mut group_buffer = vec![];
    let mut work_size = vec![];
    let mut work_sz = max_length;
    while work_sz > threads_per_group {
        work_sz = work_sz.div_ceil(threads_per_group);
        group_buffer.push(create_tensor([work_sz], &input.device, DType::I32));
        work_size.push(work_sz);
    }

    // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
    unsafe {
        client
            .launch_unchecked(
                PrefixSumScanSums::task(),
                calc_cube_count_1d(work_size[0] as u32, PrefixSumScanSums::WORKGROUP_SIZE[0]),
                Bindings::new().with_buffers(vec![
                    outputs.handle.clone().binding(),
                    group_buffer[0].handle.clone().binding(),
                    length.handle.clone().binding(),
                ]),
            )
            .expect("Failed to run prefix sums");
    }

    for l in 0..(group_buffer.len() - 1) {
        // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
        unsafe {
            client
                .launch_unchecked(
                    PrefixSumScanSums::task(),
                    calc_cube_count_1d(
                        work_size[l + 1] as u32,
                        PrefixSumScanSums::WORKGROUP_SIZE[0],
                    ),
                    Bindings::new().with_buffers(vec![
                        group_buffer[l].handle.clone().binding(),
                        group_buffer[l + 1].handle.clone().binding(),
                        length.handle.clone().binding(),
                    ]),
                )
                .expect("Failed to run prefix sums");
        }
    }

    for l in (1..group_buffer.len()).rev() {
        let work_sz = work_size[l - 1];

        // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
        unsafe {
            client
                .launch_unchecked(
                    PrefixSumAddScannedSums::task(),
                    calc_cube_count_1d(work_sz as u32, PrefixSumAddScannedSums::WORKGROUP_SIZE[0]),
                    Bindings::new().with_buffers(vec![
                        group_buffer[l].handle.clone().binding(),
                        group_buffer[l - 1].handle.clone().binding(),
                        length.handle.clone().binding(),
                    ]),
                )
                .expect("Failed to run prefix sums");
        }
    }

    // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
    unsafe {
        client
            .launch_unchecked(
                PrefixSumAddScannedSums::task(),
                calc_cube_count_1d(
                    (work_size[0] * threads_per_group) as u32,
                    PrefixSumAddScannedSums::WORKGROUP_SIZE[0],
                ),
                Bindings::new().with_buffers(vec![
                    group_buffer[0].handle.clone().binding(),
                    outputs.handle.clone().binding(),
                    length.handle.clone().binding(),
                ]),
            )
            .expect("Failed to run prefix sums");
    }

    outputs
}

/// Compute prefix sum over all elements in the input buffer.
pub fn prefix_sum(input: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    // Create a length buffer with the full size using the backend
    let num = input.shape.dims[0];
    use burn::tensor::IntDType;
    use burn::tensor::ops::IntTensorOps;
    use burn_wgpu::CubeBackend;
    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    let length = Backend::int_full([1].into(), num as i32, &input.device, IntDType::U32);
    prefix_sum_with_length(input, length, num)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::{prefix_sum, prefix_sum_with_length};
    use burn::tensor::{Int, Tensor};
    use burn_wgpu::{CubeBackend, WgpuRuntime};

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    #[test]
    fn test_sum_tiny() {
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data([1, 1, 1, 1], &device).into_primitive();
        let summed = prefix_sum(keys);
        let summed = Tensor::<Backend, 1, Int>::from_primitive(summed).to_data();
        let summed = summed.as_slice::<i32>().expect("Wrong type");
        assert_eq!(summed.len(), 4);
        assert_eq!(summed, [1, 2, 3, 4]);
    }

    #[test]
    fn test_512_multiple() {
        const ITERS: usize = 1024;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(90 + i as i32);
        }
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let summed = prefix_sum(keys);
        let summed = Tensor::<Backend, 1, Int>::from_primitive(summed).to_data();
        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();
        for (summed, reff) in summed
            .as_slice::<i32>()
            .expect("Wrong type")
            .iter()
            .zip(prefix_sum_ref)
        {
            assert_eq!(*summed, reff);
        }
    }

    #[test]
    fn test_sum() {
        const ITERS: usize = 512 * 16 + 123;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(2 + i as i32);
            data.push(0);
            data.push(32);
            data.push(512);
            data.push(30965);
        }

        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let summed = prefix_sum(keys);
        let summed = Tensor::<Backend, 1, Int>::from_primitive(summed).to_data();

        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();

        for (summed, reff) in summed
            .as_slice::<i32>()
            .expect("Wrong type")
            .iter()
            .zip(prefix_sum_ref)
        {
            assert_eq!(*summed, reff);
        }
    }

    #[test]
    fn test_sum_large() {
        // Test with 20M elements to verify 2D dispatch works correctly.
        const NUM_ELEMENTS: usize = 30_000_000;

        // Use small values to avoid overflow in prefix sum
        let data: Vec<i32> = (0..NUM_ELEMENTS).map(|i| (i % 100) as i32).collect();

        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let summed = prefix_sum(keys);
        let summed = Tensor::<Backend, 1, Int>::from_primitive(summed).to_data();

        // Verify a few samples rather than all 20M elements
        let summed_slice = summed.as_slice::<i32>().expect("Wrong type");
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

    #[test]
    fn test_prefix_sum_with_length() {
        // Test that prefix_sum_with_length only processes up to `length` elements.
        // We create a larger buffer but only sum a subset.
        const BUFFER_SIZE: usize = 10000;
        const ACTIVE_LENGTH: usize = 100;

        let device = Default::default();

        // Fill buffer: first ACTIVE_LENGTH elements are 1, rest are 999 (shouldn't be summed)
        let mut data = vec![1i32; ACTIVE_LENGTH];
        data.extend(vec![999i32; BUFFER_SIZE - ACTIVE_LENGTH]);

        let input = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let length =
            Tensor::<Backend, 1, Int>::from_data([ACTIVE_LENGTH as i32], &device).into_primitive();

        let summed = prefix_sum_with_length(input, length, BUFFER_SIZE);
        let summed = Tensor::<Backend, 1, Int>::from_primitive(summed).to_data();
        let summed_slice = summed.as_slice::<i32>().expect("Wrong type");

        // Check that the first ACTIVE_LENGTH elements have correct prefix sums (1, 2, 3, ...)
        for i in 0..ACTIVE_LENGTH {
            assert_eq!(
                summed_slice[i],
                (i + 1) as i32,
                "Mismatch at index {i}: got {}, expected {}",
                summed_slice[i],
                i + 1
            );
        }
    }

    #[test]
    fn test_prefix_sum_with_length_large() {
        // Test with a larger subset that requires hierarchical passes
        const BUFFER_SIZE: usize = 100_000;
        const ACTIVE_LENGTH: usize = 10_000;

        let device = Default::default();

        // Use varying values for the active portion
        let mut data: Vec<i32> = (0..ACTIVE_LENGTH).map(|i| (i % 10) as i32).collect();
        data.extend(vec![999i32; BUFFER_SIZE - ACTIVE_LENGTH]);

        let input = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let length =
            Tensor::<Backend, 1, Int>::from_data([ACTIVE_LENGTH as i32], &device).into_primitive();

        let summed = prefix_sum_with_length(input, length, BUFFER_SIZE);
        let summed = Tensor::<Backend, 1, Int>::from_primitive(summed).to_data();
        let summed_slice = summed.as_slice::<i32>().expect("Wrong type");

        // Compute reference prefix sum for active portion
        let prefix_sum_ref: Vec<i32> = data[..ACTIVE_LENGTH]
            .iter()
            .scan(0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        // Verify the active portion
        for (i, (&summed, &expected)) in summed_slice[..ACTIVE_LENGTH]
            .iter()
            .zip(&prefix_sum_ref)
            .enumerate()
        {
            assert_eq!(
                summed, expected,
                "Mismatch at index {i}: got {summed}, expected {expected}"
            );
        }
    }
}
