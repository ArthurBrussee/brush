// Microbenchmark for the device-wide prefix sum, timed in isolation like the
// radix sort bench: buffers are built once, only the dispatch plus a readback
// to force completion is timed.

#![cfg_attr(target_family = "wasm", allow(unused_imports, dead_code))]

use std::sync::Arc;

use brush_cube::{CubeDevice, CubeTensor};
use brush_scan::prefix_sum;
use burn::backend::TensorMetadata;
use burn::cubecl::future::block_on;

#[cfg(not(target_family = "wasm"))]
fn main() {
    divan::main();
}

#[cfg(target_family = "wasm")]
fn main() {}

const SIZES: [usize; 4] = [1_000_000, 10_000_000, 30_000_000, 70_000_000];

fn device() -> CubeDevice {
    CubeDevice::Wgpu(block_on(brush_cube::test_helpers::test_device()))
}

fn make_input(size: usize) -> Arc<Vec<u32>> {
    Arc::new((0..size as u32).map(|i| i % 7).collect())
}

fn run_scan(device: &CubeDevice, input: &CubeTensor) {
    let out = prefix_sum(input.clone());
    // Force completion with a minimal readback: the last element only.
    let client = device.client();
    let len = out.shape()[0] as u64;
    let last = out.handle.offset_start((len - 1) * 4);
    let _ = block_on(client.read_async(vec![last]));
}

#[cfg(not(target_family = "wasm"))]
#[divan::bench_group(max_time = 4)]
mod scan_bench {
    use crate::{SIZES, device, make_input, run_scan};
    use brush_cube::create_tensor_from_slice;
    use burn::tensor::DType;

    #[divan::bench(args = SIZES)]
    fn prefix_sum(bencher: divan::Bencher, size: usize) {
        let dev = device();
        let input = create_tensor_from_slice(&make_input(size), &dev, DType::U32);
        bencher.bench_local(move || run_scan(&dev, &input));
    }
}
