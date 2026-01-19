use brush_kernel::CubeCount;
use brush_kernel::calc_cube_count_1d;
use brush_kernel::create_dispatch_buffer_1d;
use brush_kernel::create_tensor;
use brush_kernel::create_uniform_buffer;
use brush_wgsl::wgsl_kernel;
use burn::tensor::DType;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorMetadata;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::server::Bindings;
use burn_wgpu::CubeTensor;
use burn_wgpu::WgpuRuntime;

// Kernel definitions using proc macro
#[wgsl_kernel(source = "src/shaders/sort_count.wgsl")]
pub struct SortCount;

#[wgsl_kernel(source = "src/shaders/sort_reduce.wgsl")]
pub struct SortReduce;

#[wgsl_kernel(source = "src/shaders/sort_scan.wgsl")]
pub struct SortScan;

#[wgsl_kernel(source = "src/shaders/sort_scan_add.wgsl")]
pub struct SortScanAdd;

#[wgsl_kernel(source = "src/shaders/sort_scatter.wgsl")]
pub struct SortScatter;

// Import types from the generated modules
use sort_count::Uniforms;

const BLOCK_SIZE: u32 = SortCount::WG * SortCount::ELEMENTS_PER_THREAD;

/// Perform a radix argsort on the input keys and values.
///
/// If `dynamic_count` is `Some(count_buffer)`, use that buffer as the actual number
/// of keys to sort (uses dynamic GPU dispatch). If `None`, use the full buffer length
/// with static CPU dispatch.
pub fn radix_argsort(
    input_keys: CubeTensor<WgpuRuntime>,
    input_values: CubeTensor<WgpuRuntime>,
    sorting_bits: u32,
    dynamic_count: Option<CubeTensor<WgpuRuntime>>,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    assert_eq!(
        input_keys.shape.dims[0], input_values.shape.dims[0],
        "Input keys and values must have the same number of elements"
    );
    assert!(sorting_bits <= 32, "Can only sort up to 32 bits");
    assert!(
        input_keys.is_contiguous(),
        "Please ensure input keys are contiguous"
    );
    assert!(
        input_values.is_contiguous(),
        "Please ensure input keys are contiguous"
    );

    let _span = tracing::trace_span!("Radix sort").entered();

    let client = &input_keys.client.clone();
    let max_n = input_keys.shape.dims[0] as u32;

    // compute buffer and dispatch sizes
    let device = &input_keys.device.clone();

    let max_needed_wgs = max_n.div_ceil(BLOCK_SIZE);

    // Handle dynamic vs static dispatch
    let (num_keys_buf, num_wgs, num_reduce_wgs) = if let Some(count_buf) = dynamic_count {
        // Dynamic dispatch: compute workgroup counts on GPU
        // num_wgs = ceil(count / BLOCK_SIZE)
        let num_wgs = create_dispatch_buffer_1d(count_buf.clone(), BLOCK_SIZE);

        // The reduce shader expects: num_reduce_wgs = BIN_COUNT * ceil(num_wgs / BLOCK_SIZE)
        // This is NOT the same as ceil(num_wgs * BIN_COUNT / BLOCK_SIZE) due to ceiling!
        // We need: first compute ceil(num_wgs_x / BLOCK_SIZE), then multiply by BIN_COUNT.
        type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
        let num_wgs_tensor: Tensor<Backend, 1, Int> = Tensor::from_primitive(num_wgs.clone());
        let num_wgs_x = num_wgs_tensor.slice([0..1]); // Get just the X component (scalar)

        // num_reduce_wg_per_bin = ceil(num_wgs_x / BLOCK_SIZE)
        let num_reduce_wg_per_bin_buf =
            create_dispatch_buffer_1d(num_wgs_x.into_primitive(), BLOCK_SIZE);
        let num_reduce_wg_per_bin: Tensor<Backend, 1, Int> =
            Tensor::from_primitive(num_reduce_wg_per_bin_buf);
        let num_reduce_wg_per_bin_x = num_reduce_wg_per_bin.slice([0..1]);

        // num_reduce_wgs_total = num_reduce_wg_per_bin * BIN_COUNT
        let num_reduce_total: Tensor<Backend, 1, Int> =
            num_reduce_wg_per_bin_x * (SortCount::BIN_COUNT as i32);

        // Create dispatch buffer for the total (uses 2D tiling if > 65535)
        let num_reduce_wgs = create_dispatch_buffer_1d(num_reduce_total.into_primitive(), 1);

        (count_buf, CubeCount::Dynamic(num_wgs.handle.binding()), CubeCount::Dynamic(num_reduce_wgs.handle.binding()))
    } else {
        // Static dispatch: use full buffer size
        let num_keys_buf = {
            type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
            Tensor::<Backend, 1, Int>::from_ints([max_n as i32], device).into_primitive()
        };
        // Calculate dispatch counts matching the original formula
        let num_wgs_count = max_n.div_ceil(BLOCK_SIZE);
        let num_reduce_wgs_count = num_wgs_count.div_ceil(BLOCK_SIZE) * SortCount::BIN_COUNT;
        let num_wgs = calc_cube_count_1d(max_n, BLOCK_SIZE);
        let num_reduce_wgs = calc_cube_count_1d(num_reduce_wgs_count, 1);
        (num_keys_buf, num_wgs, num_reduce_wgs)
    };

    let mut cur_keys = input_keys;
    let mut cur_vals = input_values;

    for pass in 0..sorting_bits.div_ceil(4) {
        let uniforms_buffer: CubeTensor<WgpuRuntime> =
            create_uniform_buffer(Uniforms { shift: pass * 4 }, device, client);

        let count_buf = create_tensor([(max_needed_wgs as usize) * 16], device, DType::I32);

        // use safe dispatch as dynamic work count isn't verified.
        client
            .launch(
                SortCount::task(),
                num_wgs.clone(),
                Bindings::new().with_buffers(vec![
                    uniforms_buffer.handle.clone().binding(),
                    num_keys_buf.handle.clone().binding(),
                    cur_keys.handle.clone().binding(),
                    count_buf.handle.clone().binding(),
                ]),
            )
            .expect("Failed to run sorting");

        {
            let reduced_buf = create_tensor([BLOCK_SIZE as usize], device, DType::I32);

            client
                .launch(
                    SortReduce::task(),
                    num_reduce_wgs.clone(),
                    Bindings::new().with_buffers(vec![
                        num_keys_buf.handle.clone().binding(),
                        count_buf.handle.clone().binding(),
                        reduced_buf.handle.clone().binding(),
                    ]),
                )
                .expect("Failed to run sorting");

            // SAFETY: No OOB or loops in kernel.
            unsafe {
                client
                    .launch_unchecked(
                        SortScan::task(),
                        CubeCount::Static(1, 1, 1),
                        Bindings::new().with_buffers(vec![
                            num_keys_buf.handle.clone().binding(),
                            reduced_buf.handle.clone().binding(),
                        ]),
                    )
                    .expect("Failed to run sorting");
            }

            client
                .launch(
                    SortScanAdd::task(),
                    num_reduce_wgs.clone(),
                    Bindings::new().with_buffers(vec![
                        num_keys_buf.handle.clone().binding(),
                        reduced_buf.handle.clone().binding(),
                        count_buf.handle.clone().binding(),
                    ]),
                )
                .expect("Failed to run sorting");
        }

        let output_keys = create_tensor([max_n as usize], device, cur_keys.dtype());
        let output_values = create_tensor([max_n as usize], device, cur_vals.dtype());

        client
            .launch(
                SortScatter::task(),
                num_wgs.clone(),
                Bindings::new().with_buffers(vec![
                    uniforms_buffer.handle.clone().binding(),
                    num_keys_buf.handle.clone().binding(),
                    cur_keys.handle.clone().binding(),
                    cur_vals.handle.clone().binding(),
                    count_buf.handle.clone().binding(),
                    output_keys.handle.clone().binding(),
                    output_values.handle.clone().binding(),
                ]),
            )
            .expect("Failed to run sorting");

        cur_keys = output_keys;
        cur_vals = output_values;
    }
    (cur_keys, cur_vals)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::radix_argsort;
    use burn::tensor::{Int, Tensor};
    use burn_wgpu::{CubeBackend, WgpuRuntime};
    use rand::Rng;

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    #[test]
    fn test_sorting() {
        for i in 0..128 {
            let keys_inp = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2i32.pow(24) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];

            let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

            let device = Default::default();
            let keys = Tensor::<Backend, 1, Int>::from_ints(keys_inp, &device).into_primitive();
            let values = Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device)
                .into_primitive();
            let (ret_keys, ret_values) = radix_argsort(keys, values, 32, None);

            let ret_keys = Tensor::<Backend, 1, Int>::from_primitive(ret_keys).into_data();

            let ret_values = Tensor::<Backend, 1, Int>::from_primitive(ret_values).into_data();

            let inds = argsort(&keys_inp);

            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i] as u32).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i] as u32).collect();

            for (((key, val), ref_key), ref_val) in ret_keys
                .as_slice::<i32>()
                .expect("Wrong type")
                .iter()
                .zip(ret_values.as_slice::<i32>().expect("Wrong type"))
                .zip(ref_keys)
                .zip(ref_values)
            {
                assert_eq!(*key, ref_key as i32);
                assert_eq!(*val, ref_val as i32);
            }
        }
    }

    #[test]
    fn test_sorting_big() {
        // Simulate some data as one might find for a bunch of gaussians.
        let mut rng = rand::rng();
        let mut keys_inp = Vec::new();
        for i in 0..10000 {
            let start = rng.random_range(i..i + 150);
            let end = rng.random_range(start..start + 250);

            for j in start..end {
                if rng.random::<f32>() < 0.5 {
                    keys_inp.push(j);
                }
            }
        }

        let values_inp: Vec<_> = keys_inp.iter().map(|&x| x * 2 + 5).collect();

        let device = Default::default();
        let keys =
            Tensor::<Backend, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive();
        let values =
            Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive();
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32, None);

        let ret_keys = Tensor::<Backend, 1, Int>::from_primitive(ret_keys).to_data();
        let ret_values = Tensor::<Backend, 1, Int>::from_primitive(ret_values).to_data();

        let inds = argsort(&keys_inp);
        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        for (((key, val), ref_key), ref_val) in ret_keys
            .as_slice::<i32>()
            .expect("Wrong type")
            .iter()
            .zip(ret_values.as_slice::<i32>().expect("Wrong type"))
            .zip(ref_keys)
            .zip(ref_values)
        {
            assert_eq!(*key, ref_key as i32);
            assert_eq!(*val, ref_val as i32);
        }
    }

    #[test]
    fn test_sorting_large() {
        // Test with a ton of elements to verify 2D dispatch works correctly.
        const NUM_ELEMENTS: usize = 30_000_000;

        let mut rng = rand::rng();

        // Generate random keys with limited range to allow verification
        let keys_inp: Vec<u32> = (0..NUM_ELEMENTS)
            .map(|_| rng.random_range(0..1_000_000))
            .collect();
        let values_inp: Vec<u32> = (0..NUM_ELEMENTS).map(|i| i as u32).collect();

        let device = Default::default();
        let keys =
            Tensor::<Backend, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive();
        let values =
            Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive();
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32, None);

        let ret_keys = Tensor::<Backend, 1, Int>::from_primitive(ret_keys).to_data();
        let ret_values = Tensor::<Backend, 1, Int>::from_primitive(ret_values).to_data();

        let ret_keys_slice = ret_keys.as_slice::<i32>().expect("Wrong type");
        let ret_values_slice = ret_values.as_slice::<i32>().expect("Wrong type");

        assert_eq!(ret_keys_slice.len(), NUM_ELEMENTS);
        assert_eq!(ret_values_slice.len(), NUM_ELEMENTS);

        // Verify the output is sorted
        for i in 1..NUM_ELEMENTS {
            assert!(
                ret_keys_slice[i - 1] <= ret_keys_slice[i],
                "Keys not sorted at index {i}: {} > {}",
                ret_keys_slice[i - 1],
                ret_keys_slice[i]
            );
        }

        // Verify that values correspond to original indices that had those keys
        // Check a sample of indices to avoid O(n^2) verification
        let check_indices = [0, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 19_999_999];
        for &idx in &check_indices {
            let sorted_key = ret_keys_slice[idx] as u32;
            let original_idx = ret_values_slice[idx] as usize;
            assert_eq!(
                keys_inp[original_idx], sorted_key,
                "Value at index {idx} points to wrong original index"
            );
        }
    }
}
