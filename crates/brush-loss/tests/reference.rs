//! Smoke + invariant tests for the loss kernels.
//!
//! GT lives as `[H, W]` u32 packing `[r g b a]` u8. We feed deterministic u8
//! data through `image_loss` and check structural properties (`SSIM(x, x) ≈ 1`,
//! output range, backward produces finite gradients). Bit-exact reference
//! matching is covered by the integration training tests in `brush-bench-test`.

use brush_loss::{ImageLossConfig, image_loss};
use burn::tensor::{Device, Int, Tensor, TensorData};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn pack_rgba(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|p| {
            u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16 | u32::from(p[3]) << 24
        })
        .collect()
}

/// Deterministic u8 pattern (avoids RNG so the test is reproducible across
/// machines). Returns `H*W*4` RGBA bytes.
fn make_pattern(h: usize, w: usize, scale: u32, offset: u32) -> Vec<u8> {
    (0..h * w * 4)
        .map(|i| ((i as u32 * scale + offset) % 251) as u8)
        .collect()
}

fn pred_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<3> {
    let rgb: Vec<f32> = bytes
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]].map(|b| b as f32 / 255.0))
        .collect();
    Tensor::<1>::from_floats(rgb.as_slice(), device).reshape([h, w, 3])
}

fn gt_packed_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<2, Int> {
    // Bit-reinterpret the u32 packing as i32 so the dispatch int_from_data
    // path doesn't reject magnitudes > i32::MAX.
    let packed: Vec<i32> = pack_rgba(bytes).into_iter().map(|x| x as i32).collect();
    Tensor::from_data(TensorData::new(packed, [h, w]), device)
}

fn ssim_only_cfg() -> ImageLossConfig {
    ImageLossConfig {
        l1_weight: 0.0,
        ssim_weight: 1.0,
        composite_bg: None,
        mask: false,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_identical_inputs_is_one() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes = make_pattern(h, w, 11, 13);
    let pred = pred_from_bytes(&bytes, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = image_loss(pred, gt, ssim_only_cfg());
    let mean: f32 = map
        .into_data_async()
        .await
        .expect("readback")
        .iter::<f32>()
        .sum::<f32>()
        / (h * w * 3) as f32;
    // Identical inputs SSIM saturates at 1; allow a sub-ULP roundoff.
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "SSIM(x, x) should be 1, got {mean}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_in_clamp_range() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes_a = make_pattern(h, w, 7, 19);
    let bytes_b = make_pattern(h, w, 13, 7);
    let pred = pred_from_bytes(&bytes_a, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let data: Vec<f32> = image_loss(pred, gt, ssim_only_cfg())
        .into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("vec");
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (-1.0..=1.0).contains(&min) && (-1.0..=1.0).contains(&max),
        "SSIM out of [-1, 1]: min={min} max={max}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn image_loss_backward_runs() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (32, 48);
    let bytes_a = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let pred = pred_from_bytes(&bytes_a, h, w, &device).require_grad();
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let map = image_loss(
        pred,
        gt,
        ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: None,
            mask: false,
        },
    );
    let _grads = map.mean().backward();
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn alpha_match_via_4ch_pred() {
    // Feeding 4-channel `pred` makes the kernel emit `|pred.a - gt.a|`
    // into the alpha channel of the loss map.
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (16, 24);
    let bytes = make_pattern(h, w, 17, 5);
    let rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
        .reshape([h, w, 4])
        .require_grad();
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = image_loss(
        pred,
        gt,
        ImageLossConfig {
            l1_weight: 1.0,
            ssim_weight: 0.0,
            composite_bg: None,
            mask: false,
        },
    );
    let _grads = map.mean().backward();
}
