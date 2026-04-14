// Microbenchmarks for the radix sort. Lives inside brush-sort so we can
// measure the sort kernels in isolation, separately from any rendering work.
//
// Each bench preallocates the input buffers once outside the timed block and
// only times the radix_argsort dispatch + a final readback to force the GPU to
// finish. The buffers are built directly as CubeTensors (DType::U32) instead
// of going through Burn's typed `Tensor::<_, _, Int>::from_ints`, so we can
// sort the full u32 range — Burn's i32-typed constructor would panic on any
// value with the high bit set.

use std::sync::Arc;

use brush_kernel::CubeTensor;
use brush_sort::radix_argsort;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{DType, Shape};
use burn_cubecl::cubecl::Runtime;
use burn_cubecl::cubecl::future::block_on;
use burn_wgpu::WgpuRuntime;

fn main() {
    divan::main();
}

// Sizes spanning the interesting range:
// - 1M  : "normal" frame, well below any cliff
// - 10M : medium frame
// - 30M : ~max prior to the sort_scan fix
// - 70M : just past the old `num_reduce_wgs > BLOCK_SIZE` cliff at ~67M
const SIZES: [usize; 4] = [1_000_000, 10_000_000, 30_000_000, 70_000_000];

// Number of distinct tile-id values to use for tile-id-shaped sorts.
// 1024 matches the renderer's tile budget for a 512x512 image.
const TILE_ID_RANGE: u32 = 1024;

fn device() -> WgpuDevice {
    block_on(brush_kernel::test_helpers::test_device())
}

#[derive(Copy, Clone)]
enum KeyKind {
    // Narrow keys in [0, 1024) — matches the renderer's tile-sort workload.
    // Sort runs with 10 sorting bits => 3 radix passes.
    TileIds,
    // Full 32-bit random keys — matches what the depth sort would do if depths
    // were transmuted to u32 (which is what the gaussian-splat depth sort does
    // internally for non-negative floats). 8 radix passes.
    Random32,
}

// Generate keys + values once and return an Arc'd vector for cheap cloning.
fn make_inputs(size: usize, key_kind: KeyKind) -> Arc<(Vec<u32>, Vec<u32>)> {
    use std::num::Wrapping;
    let mut state = Wrapping(0xC0FFEEu64 ^ (size as u64).rotate_left(17));
    let mut next_u32 = || {
        state += Wrapping(0x9E3779B97F4A7C15u64);
        let mut z = state.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        z as u32
    };

    let keys: Vec<u32> = match key_kind {
        KeyKind::TileIds => (0..size).map(|_| next_u32() % TILE_ID_RANGE).collect(),
        KeyKind::Random32 => (0..size).map(|_| next_u32()).collect(),
    };
    let values: Vec<u32> = (0..size as u32).collect();
    Arc::new((keys, values))
}

// Build a CubeTensor directly from a raw u32 slice. Bypasses Burn's i32-typed
// `from_ints` which would panic on values >= 2^31.
fn upload_u32(device: &WgpuDevice, data: &[u32]) -> CubeTensor<WgpuRuntime> {
    let client = WgpuRuntime::client(device);
    let handle = client.create_from_slice(bytemuck::cast_slice(data));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        Shape::new([data.len()]),
        handle,
        DType::U32,
    )
}

fn run_sort(device: &WgpuDevice, inputs: &(Vec<u32>, Vec<u32>), bits: u32) {
    let keys = upload_u32(device, &inputs.0);
    let values = upload_u32(device, &inputs.1);
    let (sorted_keys, sorted_values) = radix_argsort(keys, values, bits);
    // Force completion: read both buffers back so the GPU finishes before we
    // return from the bencher closure.
    let client = WgpuRuntime::client(device);
    let _ = block_on(client.read_async(vec![sorted_keys.handle, sorted_values.handle]));
}

#[divan::bench_group(max_time = 4)]
mod sort_bench {
    use crate::{KeyKind, SIZES, device, make_inputs, run_sort};

    #[divan::bench(args = SIZES)]
    fn radix_argsort_10bit(bencher: divan::Bencher, size: usize) {
        let dev = device();
        let inputs = make_inputs(size, KeyKind::TileIds);
        bencher.bench_local(move || run_sort(&dev, &inputs, 10));
    }

    #[divan::bench(args = SIZES)]
    fn radix_argsort_32bit(bencher: divan::Bencher, size: usize) {
        let dev = device();
        let inputs = make_inputs(size, KeyKind::Random32);
        bencher.bench_local(move || run_sort(&dev, &inputs, 32));
    }
}
