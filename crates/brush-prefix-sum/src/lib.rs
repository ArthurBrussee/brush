mod shaders;

use brush_kernel::calc_cube_count;
use brush_kernel::create_tensor;
use brush_kernel::kernel_source_gen;
use burn::tensor::DType;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::CubeCount;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::cubecl::prelude::CUBE_DIM_X;
use burn_cubecl::cubecl::prelude::*;
use burn_wgpu::WgpuRuntime;
use shaders::prefix_sum_add_scanned_sums;
use shaders::prefix_sum_scan;
use shaders::prefix_sum_scan_sums;

kernel_source_gen!(PrefixSumScan {}, prefix_sum_scan);
kernel_source_gen!(PrefixSumScanSums {}, prefix_sum_scan_sums);
kernel_source_gen!(PrefixSumAddScannedSums {}, prefix_sum_add_scanned_sums);

use burn_wgpu::CubeTensor;

#[cube(launch)]
pub fn kernel_inclusive_sum(output: &mut Tensor<i32>) {
    let mut val = 0i32;
    if ABSOLUTE_POS < output.len() {
        val = output[ABSOLUTE_POS];
    }
    sync_units();
    val = plane_inclusive_sum(val);
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = val;
    }
}
#[cube(launch)]
fn prefix_sum_scan_sums_k<F: Int>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let val = input[ABSOLUTE_POS * CUBE_DIM_X - 1];
    output[ABSOLUTE_POS] = plane_inclusive_sum(val);
}
#[cube(launch)]
fn prefix_sum_add_scanned_sums_k(input: &Tensor<i32>, output: &mut Tensor<i32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] += input[CUBE_POS_X];
    }
}

/// Perform an prefix sum operation across all elements of the input tensor.
/// This sums all values to the "left" of the unit, including this unit's value.
/// Also known as "inclusive sum" or "inclusive scan".
///
/// # Example
/// `prefix_sum([1, 2, 3, 4, 5]) == [1, 3, 6, 10, 15]`
pub fn prefix_sum(input: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    const CUBE_SIZE: u32 = 256;
    assert!(CUBE_SIZE >= 2);

    let client = &input.client;
    let input_size = input.shape.dims[0] as u32;

    // Allocate temporary buffers for intermediate results
    let temp_tensors: Vec<CubeTensor<WgpuRuntime>> =
        std::iter::successors(Some(input_size), |&prev| {
            let next = prev.div_ceil(CUBE_SIZE);
            (next > 1).then_some(next)
        })
        .map(|work_size| {
            create_tensor::<1, WgpuRuntime>([work_size as usize], &input.device, client, DType::I32)
        })
        .collect();

    let mut group_buffers: Vec<&CubeTensor<WgpuRuntime>> = temp_tensors.iter().collect();
    group_buffers.insert(0, &input);

    // Calculate inclusive sum for each plane
    // E.g. for CUBE_DIM_X=3: [1, 1, 1, 1, 1, 1] -> [1, 2, 3, 1, 2, 3]
    let block_count = input_size.div_ceil(CUBE_SIZE);
    kernel_inclusive_sum::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(block_count, 1, 1),
        CubeDim::new(CUBE_SIZE, 1, 1),
        input.as_tensor_arg::<u32>(1),
    );

    // Hierarchical prefix sum computation
    // Phase 1: Compute block-level prefix sums going up the hierarchy
    let mut last_buffer = &input;
    for &buffer in group_buffers.iter().skip(1) {
        let block_count = (last_buffer.shape.num_elements() as u32).div_ceil(CUBE_SIZE);
        prefix_sum_scan_sums_k::launch::<u32, WgpuRuntime>(
            client,
            CubeCount::Static(block_count, 1, 1),
            CubeDim::new(CUBE_SIZE, 1, 1),
            last_buffer.as_tensor_arg::<u32>(1),
            buffer.as_tensor_arg::<u32>(1),
        );
        last_buffer = buffer;
    }

    // Phase 2: Propagate block-level sums down the hierarchy
    for &buffer in group_buffers.iter().rev().skip(1) {
        let block_count = (last_buffer.shape.num_elements() as u32).div_ceil(CUBE_SIZE);
        prefix_sum_add_scanned_sums_k::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(block_count, 1, 1),
            CubeDim::new(CUBE_SIZE, 1, 1),
            last_buffer.as_tensor_arg::<u32>(1),
            buffer.as_tensor_arg::<u32>(1),
        );
        last_buffer = buffer;
    }

    input
}

pub fn prefix_sum_wgsl(input: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    let threads_per_group = shaders::prefix_sum_helpers::THREADS_PER_GROUP as usize;
    let num = input.shape.dims[0];
    let client = &input.client;
    let outputs = create_tensor(input.shape.dims::<1>(), &input.device, client, DType::I32);

    // SAFETY: Kernel has to contain no OOB indexing.
    unsafe {
        client.execute_unchecked(
            PrefixSumScan::task(),
            calc_cube_count([num as u32], PrefixSumScan::WORKGROUP_SIZE),
            vec![input.handle.binding(), outputs.handle.clone().binding()],
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
        group_buffer.push(create_tensor::<1, WgpuRuntime>(
            [work_sz],
            &input.device,
            client,
            DType::I32,
        ));
        work_size.push(work_sz);
    }

    // SAFETY: Kernel has to contain no OOB indexing.
    unsafe {
        client.execute_unchecked(
            PrefixSumScanSums::task(),
            calc_cube_count([work_size[0] as u32], PrefixSumScanSums::WORKGROUP_SIZE),
            vec![
                outputs.handle.clone().binding(),
                group_buffer[0].handle.clone().binding(),
            ],
        );
    }

    for l in 0..(group_buffer.len() - 1) {
        // SAFETY: Kernel has to contain no OOB indexing.
        unsafe {
            client.execute_unchecked(
                PrefixSumScanSums::task(),
                calc_cube_count([work_size[l + 1] as u32], PrefixSumScanSums::WORKGROUP_SIZE),
                vec![
                    group_buffer[l].handle.clone().binding(),
                    group_buffer[l + 1].handle.clone().binding(),
                ],
            );
        }
    }

    for l in (1..group_buffer.len()).rev() {
        let work_sz = work_size[l - 1];

        // SAFETY: Kernel has to contain no OOB indexing.
        unsafe {
            client.execute_unchecked(
                PrefixSumAddScannedSums::task(),
                calc_cube_count([work_sz as u32], PrefixSumAddScannedSums::WORKGROUP_SIZE),
                vec![
                    group_buffer[l].handle.clone().binding(),
                    group_buffer[l - 1].handle.clone().binding(),
                ],
            );
        }
    }

    // SAFETY: Kernel has to contain no OOB indexing.
    unsafe {
        client.execute_unchecked(
            PrefixSumAddScannedSums::task(),
            calc_cube_count(
                [(work_size[0] * threads_per_group) as u32],
                PrefixSumAddScannedSums::WORKGROUP_SIZE,
            ),
            vec![
                group_buffer[0].handle.clone().binding(),
                outputs.handle.clone().binding(),
            ],
        );
    }

    outputs
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::prefix_sum;
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
}
