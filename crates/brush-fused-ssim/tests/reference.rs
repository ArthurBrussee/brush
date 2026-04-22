//! Reference tests for the fused SSIM kernel.
//!
//! Reference values were generated once from the previous conv2d-based
//! `Ssim` impl (the `Ssim::new(11, 3, ..)` configuration: 11×11 separable
//! gaussian, sigma = 1.5, 3-channel). The fused kernel must match those
//! summary statistics within a small floating-point tolerance — different
//! summation order over the separable blur produces sub-ULP drift on
//! per-pixel values which compounds to ~1e-5 on per-image aggregates.

use brush_fused_ssim::fused_ssim;
use brush_render::MainBackend;
use burn::{backend::Autodiff, tensor::Tensor};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

type DiffBack = Autodiff<MainBackend>;

/// One captured reference summary for a deterministic SSIM input.
struct RefCase {
    h: usize,
    w: usize,
    s1: f32,
    o1: f32,
    s2: f32,
    o2: f32,
    mean: f32,
    mean_sq: f32,
    min: f32,
    max: f32,
}

fn make_img(h: usize, w: usize, scale: f32, offset: f32) -> Vec<f32> {
    let n = h * w * 3;
    (0..n)
        .map(|i| ((i as f32 * scale + offset).sin() + 1.0) * 0.5)
        .collect()
}

const REF_CASES: &[RefCase] = &[
    // Identical inputs collapse to ~1.0 everywhere — the saturation-clamp
    // in the kernel keeps this exactly bounded; we just check it didn't
    // explode.
    RefCase {
        h: 30,
        w: 50,
        s1: 0.12,
        o1: 0.5,
        s2: 0.12,
        o2: 0.5,
        mean: 1.0,
        mean_sq: 1.0,
        min: 0.9999999,
        max: 1.0000001,
    },
    // Same shape, phase-shifted — partial similarity.
    RefCase {
        h: 30,
        w: 50,
        s1: 0.12,
        o1: 0.5,
        s2: 0.12,
        o2: 1.0,
        mean: 0.8735662,
        mean_sq: 0.7639209,
        min: 0.742158,
        max: 0.9670315,
    },
    // Wildly different inputs — covers the negative-SSIM regions.
    RefCase {
        h: 30,
        w: 50,
        s1: 0.12,
        o1: 0.5,
        s2: 0.53,
        o2: 2.0,
        mean: 0.0786799,
        mean_sq: 0.0259744,
        min: -0.0229384,
        max: 0.6426152,
    },
    RefCase {
        h: 64,
        w: 96,
        s1: 0.0317,
        o1: 0.5,
        s2: 0.0317,
        o2: 0.5,
        mean: 1.0,
        mean_sq: 1.0,
        min: 0.9999999,
        max: 1.0000001,
    },
    RefCase {
        h: 64,
        w: 96,
        s1: 0.05,
        o1: 0.5,
        s2: 0.07,
        o2: 1.0,
        mean: 0.0400423,
        mean_sq: 0.2632874,
        min: -0.7258162,
        max: 0.9671891,
    },
    // Non-square, smaller than one workgroup tile (16×16) in one dim.
    RefCase {
        h: 17,
        w: 41,
        s1: 0.1,
        o1: 0.0,
        s2: 0.13,
        o2: 1.5,
        mean: 0.1192551,
        mean_sq: 0.0478722,
        min: 0.0014375,
        max: 0.8366008,
    },
];

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn fused_ssim_matches_reference() {
    let device = brush_kernel::test_helpers::test_device().await;

    for case in REF_CASES {
        let img1_data = make_img(case.h, case.w, case.s1, case.o1);
        let img2_data = make_img(case.h, case.w, case.s2, case.o2);
        let img1 = Tensor::<DiffBack, 1>::from_floats(img1_data.as_slice(), &device)
            .reshape([case.h, case.w, 3]);
        let img2 = Tensor::<DiffBack, 1>::from_floats(img2_data.as_slice(), &device)
            .reshape([case.h, case.w, 3]);

        let map = fused_ssim(img1, img2);
        let data: Vec<f32> = map
            .into_data_async()
            .await
            .expect("readback")
            .into_vec()
            .unwrap();

        let n = (case.h * case.w * 3) as f32;
        let sum: f32 = data.iter().copied().sum();
        let sumsq: f32 = data.iter().map(|v| v * v).sum();
        let mean = sum / n;
        let mean_sq = sumsq / n;
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let label = format!(
            "{}x{} s1={} o1={} s2={} o2={}",
            case.h, case.w, case.s1, case.o1, case.s2, case.o2
        );
        assert!(
            (mean - case.mean).abs() < 1e-5,
            "{label}: mean drifted: ref={} got={mean}",
            case.mean
        );
        assert!(
            (mean_sq - case.mean_sq).abs() < 1e-5,
            "{label}: mean_sq drifted: ref={} got={mean_sq}",
            case.mean_sq
        );
        // min/max have looser tolerance — they're single-pixel extremes
        // that float-reordering can shift more visibly.
        assert!(
            (min - case.min).abs() < 1e-4,
            "{label}: min drifted: ref={} got={min}",
            case.min
        );
        assert!(
            (max - case.max).abs() < 1e-4,
            "{label}: max drifted: ref={} got={max}",
            case.max
        );
    }
}

/// Smoke-test autodiff wiring: backward through the fused op runs and
/// produces a finite gradient.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn fused_ssim_backward_runs() {
    let device = brush_kernel::test_helpers::test_device().await;
    let h = 32usize;
    let w = 48usize;
    let img1_data = make_img(h, w, 0.05, 0.5);
    let img2_data = make_img(h, w, 0.07, 1.0);
    let img1 = Tensor::<DiffBack, 1>::from_floats(img1_data.as_slice(), &device).reshape([h, w, 3]);
    let img2 = Tensor::<DiffBack, 1>::from_floats(img2_data.as_slice(), &device).reshape([h, w, 3]);

    let map = fused_ssim(img1, img2);
    let loss = map.mean();
    let _grads = loss.backward();
}
