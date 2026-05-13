//! Microbench for `image_loss_backward`. Each iteration uploads inputs,
//! dispatches the backward, and reads back the output to force GPU completion.

#![cfg_attr(target_family = "wasm", allow(unused_imports, dead_code))]

use brush_loss::{ImageLossConfig, LossOps};
use brush_render::MainBackendBase;
use burn::{
    backend::wgpu::WgpuDevice,
    tensor::{Int, Tensor, TensorData, ops::IntTensorOps},
};
use burn_cubecl::cubecl::future::block_on;

#[cfg(not(target_family = "wasm"))]
fn main() {
    divan::main();
}

#[cfg(target_family = "wasm")]
fn main() {}

const SIZES: [(usize, usize); 4] = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)];

fn device() -> WgpuDevice {
    block_on(brush_cube::test_helpers::test_device())
}

fn pack_rgba(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|p| {
            u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16 | u32::from(p[3]) << 24
        })
        .collect()
}

fn make_pattern(h: usize, w: usize, scale: u32, offset: u32) -> Vec<u8> {
    (0..h * w * 4)
        .map(|i| ((i as u32 * scale + offset) % 251) as u8)
        .collect()
}

fn pred_chw(
    bytes: &[u8],
    h: usize,
    w: usize,
    device: &WgpuDevice,
) -> burn::tensor::ops::FloatTensor<MainBackendBase> {
    let rgb: Vec<f32> = bytes
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]].map(|b| b as f32 / 255.0))
        .collect();
    Tensor::<MainBackendBase, 1>::from_floats(rgb.as_slice(), device)
        .reshape([h, w, 3])
        .permute([2, 0, 1])
        .into_primitive()
        .tensor()
}

fn gt_packed(
    bytes: &[u8],
    h: usize,
    w: usize,
    device: &WgpuDevice,
) -> burn::tensor::ops::IntTensor<MainBackendBase> {
    Tensor::<MainBackendBase, 2, Int>::new(MainBackendBase::int_from_data(
        TensorData::new(pack_rgba(bytes), [h, w]),
        device,
    ))
    .into_primitive()
}

fn run_backward(
    dev: &WgpuDevice,
    bytes_a: &[u8],
    bytes_b: &[u8],
    h: usize,
    w: usize,
    cfg: ImageLossConfig,
) {
    let pred = pred_chw(bytes_a, h, w, dev);
    let gt = gt_packed(bytes_b, h, w, dev);
    let dl_dmap = pred_chw(bytes_b, h, w, dev);
    let out = MainBackendBase::image_loss_backward(pred, gt, dl_dmap, cfg);
    let client = out.client.clone();
    let _ = block_on(client.read_async(vec![out.handle]));
}

fn full_cfg() -> ImageLossConfig {
    ImageLossConfig {
        l1_weight: 0.2,
        ssim_weight: 0.8,
        composite_bg: None,
        mask: false,
    }
}

#[cfg(not(target_family = "wasm"))]
#[divan::bench_group(max_time = 4)]
mod loss_bench {
    use crate::{SIZES, device, full_cfg, make_pattern, run_backward};

    #[divan::bench(args = SIZES)]
    fn image_loss_backward(bencher: divan::Bencher, size: (usize, usize)) {
        let (h, w) = size;
        let dev = device();
        let bytes_a = make_pattern(h, w, 5, 1);
        let bytes_b = make_pattern(h, w, 7, 11);
        bencher.bench_local(|| run_backward(&dev, &bytes_a, &bytes_b, h, w, full_cfg()));
    }
}
