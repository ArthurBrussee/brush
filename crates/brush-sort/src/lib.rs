use brush_kernel::CubeCount;
use brush_kernel::calc_cube_count_1d;
use brush_kernel::create_tensor;
use brush_kernel::create_uniform_buffer;
use brush_wgsl::wgsl_kernel;
use burn::tensor::DType;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorMetadata;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::server::KernelArguments;
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
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    assert_eq!(
        input_keys.shape()[0],
        input_values.shape()[0],
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
    let max_n = input_keys.shape()[0] as u32;

    // compute buffer and dispatch sizes
    let device = &input_keys.device.clone();

    let max_needed_wgs = max_n.div_ceil(BLOCK_SIZE);

    // Calculate dispatch counts matching the original formula
    let num_wgs_count = max_n.div_ceil(BLOCK_SIZE);
    let num_reduce_wgs_count = num_wgs_count.div_ceil(BLOCK_SIZE) * SortCount::BIN_COUNT;

    // Handle dynamic vs static dispatch
    let (num_keys_buf, num_wgs, num_reduce_wgs) = {
        // Static dispatch: use full buffer size
        let num_keys_buf = {
            type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
            Tensor::<Backend, 1, Int>::from_ints([max_n as i32], device).into_primitive()
        };
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
        client.launch(
            SortCount::task(),
            num_wgs.clone(),
            KernelArguments::new().with_buffers(vec![
                uniforms_buffer.handle.clone().binding(),
                num_keys_buf.handle.clone().binding(),
                cur_keys.handle.clone().binding(),
                count_buf.handle.clone().binding(),
            ]),
        );

        {
            // Size `reduced_buf` to the real number of per-chunk totals. The
            // sort_scan kernel now walks the whole buffer in BLOCK_SIZE chunks,
            // so we allocate `num_reduce_wgs_count` slots (rounded up to a
            // BLOCK_SIZE boundary so the final chunk's load/store can be gated
            // by a simple `< num_reduce_wgs` check). Previously this was a
            // fixed BLOCK_SIZE, which silently broke sorts above ~67M keys
            // because sort_reduce would write past the end of `reduced_buf`
            // and sort_scan would only cover the first BLOCK_SIZE entries.
            let reduced_buf_size = num_reduce_wgs_count
                .div_ceil(BLOCK_SIZE)
                .max(1)
                * BLOCK_SIZE;
            let reduced_buf = create_tensor([reduced_buf_size as usize], device, DType::I32);

            client.launch(
                SortReduce::task(),
                num_reduce_wgs.clone(),
                KernelArguments::new().with_buffers(vec![
                    num_keys_buf.handle.clone().binding(),
                    count_buf.handle.clone().binding(),
                    reduced_buf.handle.clone().binding(),
                ]),
            );
            // SAFETY: No OOB or loops in kernel.
            unsafe {
                client.launch_unchecked(
                    SortScan::task(),
                    CubeCount::Static(1, 1, 1),
                    KernelArguments::new().with_buffers(vec![
                        num_keys_buf.handle.clone().binding(),
                        reduced_buf.handle.clone().binding(),
                    ]),
                );
            }

            client.launch(
                SortScanAdd::task(),
                num_reduce_wgs.clone(),
                KernelArguments::new().with_buffers(vec![
                    num_keys_buf.handle.clone().binding(),
                    reduced_buf.handle.clone().binding(),
                    count_buf.handle.clone().binding(),
                ]),
            );
        }

        let output_keys = create_tensor([max_n as usize], device, cur_keys.dtype());
        let output_values = create_tensor([max_n as usize], device, cur_vals.dtype());

        client.launch(
            SortScatter::task(),
            num_wgs.clone(),
            KernelArguments::new().with_buffers(vec![
                uniforms_buffer.handle.clone().binding(),
                num_keys_buf.handle.clone().binding(),
                cur_keys.handle.clone().binding(),
                cur_vals.handle.clone().binding(),
                count_buf.handle.clone().binding(),
                output_keys.handle.clone().binding(),
                output_values.handle.clone().binding(),
            ]),
        );

        cur_keys = output_keys;
        cur_vals = output_values;
    }
    (cur_keys, cur_vals)
}

#[cfg(test)]
mod tests {
    use crate::radix_argsort;
    use burn::tensor::{Int, Tensor};
    use burn_wgpu::{CubeBackend, WgpuRuntime};
    use rand::RngExt;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting() {
        let device = brush_kernel::test_helpers::test_device().await;

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

            let keys = Tensor::<Backend, 1, Int>::from_ints(keys_inp, &device).into_primitive();
            let values = Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device)
                .into_primitive();
            let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

            let ret_keys = Tensor::<Backend, 1, Int>::from_primitive(ret_keys)
                .into_data_async()
                .await
                .expect("readback");

            let ret_values = Tensor::<Backend, 1, Int>::from_primitive(ret_values)
                .into_data_async()
                .await
                .expect("readback");

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

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting_big() {
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

        let device = brush_kernel::test_helpers::test_device().await;
        let keys =
            Tensor::<Backend, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive();
        let values =
            Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive();
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

        let ret_keys = Tensor::<Backend, 1, Int>::from_primitive(ret_keys)
            .to_data_async()
            .await
            .expect("readback");
        let ret_values = Tensor::<Backend, 1, Int>::from_primitive(ret_values)
            .to_data_async()
            .await
            .expect("readback");

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

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting_large() {
        // Test with a ton of elements to verify 2D dispatch works correctly.
        const NUM_ELEMENTS: usize = 30_000_000;

        let mut rng = rand::rng();

        // Generate random keys with limited range to allow verification
        let keys_inp: Vec<u32> = (0..NUM_ELEMENTS)
            .map(|_| rng.random_range(0..1_000_000))
            .collect();
        let values_inp: Vec<u32> = (0..NUM_ELEMENTS).map(|i| i as u32).collect();

        let device = brush_kernel::test_helpers::test_device().await;
        let keys =
            Tensor::<Backend, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive();
        let values =
            Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive();
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

        let ret_keys = Tensor::<Backend, 1, Int>::from_primitive(ret_keys)
            .to_data_async()
            .await
            .expect("readback");
        let ret_values = Tensor::<Backend, 1, Int>::from_primitive(ret_values)
            .to_data_async()
            .await
            .expect("readback");

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

    // Regression test for a silent corruption in the radix sort that hits at
    // ~67M keys.
    //
    // The pipeline is: sort_count → sort_reduce → sort_scan → sort_scan_add →
    // sort_scatter. `sort_reduce` writes `num_reduce_wgs` per-chunk totals into
    // `reduced_buf`, and `sort_scan` is supposed to turn that into an exclusive
    // prefix sum that `sort_scan_add` then uses as the base offset for each
    // sort_scatter workgroup. Historically `reduced_buf` was hardcoded to
    // BLOCK_SIZE (1024) entries and `sort_scan` was a single workgroup that
    // only covered BLOCK_SIZE entries. Once `num_reduce_wgs > BLOCK_SIZE` —
    // which happens when `num_wgs > BLOCK_SIZE² / BIN_COUNT = 65536`, i.e.
    // around 67M keys — the per-chunk totals for the highest-bin chunks were
    // dropped on the floor and `sort_scan_add` read zeros for those chunks'
    // base offsets. Sort_scatter then scattered those keys into the wrong
    // slots, silently overwriting other keys.
    //
    // Weaker checks (monotonicity + spot value consistency) miss this: when a
    // bin-15 key gets overwritten by a smaller bin-0 key, the output is still
    // monotonic and the (key, value) pair at that slot still references its
    // own original index. The only way to catch it is to verify that every
    // input key still appears in the output exactly once. We do that by
    // sorting a permutation of `[0..N)` and asserting the result is the
    // identity `[0, 1, ..., N-1]`. Anything else — duplicates, drops,
    // out-of-order — fails the check.
    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting_above_scan_block_size() {
        // 70M keys: num_wgs ≈ 68360, num_reduce_wgs = 16 * ceil(68360/1024) = 1072,
        // i.e. 48 entries past the old BLOCK_SIZE = 1024 cap.
        const NUM_ELEMENTS: usize = 70_000_000;

        // Deterministic shuffle so the test is reproducible without dragging
        // in `rand`'s seedable RNG plumbing. SplitMix64-based Fisher-Yates.
        let mut keys_inp: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();
        {
            use std::num::Wrapping;
            let mut state = Wrapping(0xD15EA5Eu64);
            for i in (1..keys_inp.len()).rev() {
                state += Wrapping(0x9E3779B97F4A7C15u64);
                let mut z = state.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                let j = (z as usize) % (i + 1);
                keys_inp.swap(i, j);
            }
        }
        // Use the original index as the value, so we can also check that
        // (key, value) round-trips cleanly through the sort.
        let values_inp: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();
        // values_inp[i] == i, and keys_inp[i] is some unique key. After sort,
        // ret_keys should be [0..N) and ret_values[k] should equal the
        // original index `i` such that keys_inp[i] == k.
        let mut expected_values = vec![0u32; NUM_ELEMENTS];
        for (i, &k) in keys_inp.iter().enumerate() {
            expected_values[k as usize] = i as u32;
        }

        let device = brush_kernel::test_helpers::test_device().await;
        let keys =
            Tensor::<Backend, 1, Int>::from_ints(keys_inp.as_slice(), &device).into_primitive();
        let values =
            Tensor::<Backend, 1, Int>::from_ints(values_inp.as_slice(), &device).into_primitive();
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

        let ret_keys = Tensor::<Backend, 1, Int>::from_primitive(ret_keys)
            .to_data_async()
            .await
            .expect("readback");
        let ret_values = Tensor::<Backend, 1, Int>::from_primitive(ret_values)
            .to_data_async()
            .await
            .expect("readback");

        let ret_keys_slice = ret_keys.as_slice::<i32>().expect("Wrong type");
        let ret_values_slice = ret_values.as_slice::<i32>().expect("Wrong type");

        assert_eq!(ret_keys_slice.len(), NUM_ELEMENTS);
        assert_eq!(ret_values_slice.len(), NUM_ELEMENTS);

        // Strict identity check: every position must hold exactly the right
        // key and value. A single dropped or duplicated key fails this.
        for i in 0..NUM_ELEMENTS {
            assert_eq!(
                ret_keys_slice[i] as u32, i as u32,
                "key at sorted index {i} is {}, expected {i}",
                ret_keys_slice[i]
            );
            assert_eq!(
                ret_values_slice[i] as u32, expected_values[i],
                "value at sorted index {i} is {}, expected {}",
                ret_values_slice[i], expected_values[i]
            );
        }
    }
}
