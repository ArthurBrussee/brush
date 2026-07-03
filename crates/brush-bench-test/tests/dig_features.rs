//! Tests for the `DiG` feature rasterizer: forward consistency against the
//! RGB path, gradient correctness via finite differences, and a training
//! smoke test exercising the feature loss + refine bookkeeping.

#![allow(clippy::missing_assert_message)]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_render_bwd::{render_splat_features, render_splats};
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::module::AutodiffModule;
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const TEST_SEED: u64 = 4242;
/// The SH basis constant for degree 0 (matches the render kernels).
const SH_C0: f32 = 0.282_094_79;

/// Deterministic degree-0 test splats; returns the raw SH coefficients so
/// tests can derive the expected rendered colors.
fn test_splats(device: &Device, count: usize) -> (Splats, Vec<f32>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-1.5..1.5),
                rng.random_range(-1.5..1.5),
                rng.random_range(-2.0..2.0),
            ]
        })
        .collect();
    let log_scales: Vec<f32> = (0..count)
        .flat_map(|_| {
            let s = rng.random_range(0.05..0.3_f32).ln();
            [s, s, s]
        })
        .collect();
    let rotations: Vec<f32> = (0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
            ]
        })
        .collect();
    let opacities: Vec<f32> = (0..count).map(|_| rng.random_range(0.6..0.95)).collect();
    let splats = Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh_coeffs.clone(),
        opacities,
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0);
    (splats, sh_coeffs)
}

fn test_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, -8.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    )
}

/// With 3-dim features set to the degree-0 colors, the feature rasterizer
/// must reproduce the RGB rasterizer's output (zero background, and all
/// colors positive so its `max(0)` clamp is a no-op).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn feature_render_matches_rgb_render() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (splats, sh_coeffs) = test_splats(&device, 200);
    let camera = test_camera();
    let img_size = glam::uvec2(64, 64);

    let rgb = render_splats(splats.clone(), &camera, img_size, Vec3::ZERO).await;
    assert!(rgb.num_visible > 0);
    let rgb_data = rgb
        .img
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .unwrap();

    // Degree-0 color as computed in-kernel: SH_C0 * coeff + 0.5.
    let colors: Vec<f32> = sh_coeffs.iter().map(|c| SH_C0 * c + 0.5).collect();
    let features: Tensor<2> = Tensor::from_data(
        TensorData::new(colors, [splats.num_splats() as usize, 3]),
        &device,
    );

    let feat = render_splat_features(
        splats.transforms.val(),
        splats.raw_opacities.val(),
        features,
        &camera,
        img_size,
        SplatRenderMode::Default,
    )
    .await;
    let feat_data = feat
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .unwrap();

    assert_eq!(rgb_data.len(), feat_data.len());
    let max_diff = rgb_data
        .iter()
        .zip(&feat_data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "feature render deviates from RGB render: max diff {max_diff}"
    );
}

/// Finite-difference check of the feature gradient (the only gradient the
/// feature backward produces — geometry is detached by design).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn feature_gradients_match_finite_diff() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (splats, _) = test_splats(&device, 16);
    let camera = test_camera();
    let img_size = glam::uvec2(32, 32);
    let n = splats.num_splats() as usize;
    const D: usize = 4;

    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let feat_vals: Vec<f32> = (0..n * D).map(|_| rng.random_range(-1.0..1.0)).collect();

    let render_loss = |vals: Vec<f32>| {
        let splats = splats.clone();
        let device = device.clone();
        async move {
            let features: Tensor<2> =
                Tensor::from_data(TensorData::new(vals, [n, D]), &device).require_grad();
            let out = render_splat_features(
                splats.transforms.val(),
                splats.raw_opacities.val(),
                features.clone(),
                &camera,
                img_size,
                SplatRenderMode::Default,
            )
            .await;
            // Sum of squares over the feature channels — feature-dependent
            // gradient, alpha channel excluded (it carries no feature grad).
            let feat_chans = out.slice(burn::tensor::s![.., .., 0..D]);
            let loss = (feat_chans.clone() * feat_chans).sum();
            (loss, features)
        }
    };

    let (loss, features) = render_loss(feat_vals.clone()).await;
    let grads = loss.backward();
    let analytic = features
        .grad(&grads)
        .expect("feature gradient missing")
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .unwrap();

    let eps = 5e-2f32;
    // Probe a handful of (splat, channel) entries spread across the table.
    for &idx in &[0usize, 5, D * 3 + 1, D * 7 + 2, D * 15 + 3] {
        let mut plus = feat_vals.clone();
        plus[idx] += eps;
        let mut minus = feat_vals.clone();
        minus[idx] -= eps;
        let (loss_p, _) = render_loss(plus).await;
        let (loss_m, _) = render_loss(minus).await;
        let lp: f32 = loss_p.into_scalar_async().await.expect("readback");
        let lm: f32 = loss_m.into_scalar_async().await.expect("readback");
        let fd = (lp - lm) / (2.0 * eps);
        let an = analytic[idx];
        let denom = fd.abs().max(an.abs()).max(1e-3);
        assert!(
            ((fd - an) / denom).abs() < 5e-2,
            "gradient mismatch at {idx}: finite-diff {fd} vs analytic {an}"
        );
    }
}

/// Training with feature supervision: the `DiG` loss, optimizer steps, and
/// the refine-time feature remapping must run without panicking and keep
/// the feature table aligned with the splat count.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn training_with_features_and_refine() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (mut splats, _) = test_splats(&device, 100);
    let config = TrainConfig {
        dino: true,
        ..Default::default()
    };
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    // Tiny GT: an 8×8 feature map with 8 channels.
    let (gh, gw, c) = (8usize, 8usize, 8usize);
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let gt: Vec<f32> = (0..gh * gw * c)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let img_packed = TensorData::new(vec![0x7f7f_7fffu32 as i32; 64 * 64], [64, 64]);
    let batch = SceneBatch {
        img_packed,
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        features: Some((TensorData::new(gt, [gh, gw, c]), c)),
        camera: test_camera(),
    };

    for _ in 0..3 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats).await;
        splats = new_splats;
        let loss: f32 = stats.loss.into_scalar_async().await.expect("loss readback");
        assert!(loss.is_finite());
    }

    // Refine exercises prune + split remapping of the feature table.
    let (splats_refined, _stats) = trainer.refine(3, splats.valid()).await;
    let mut splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats_refined);

    // One more step after refine — shape mismatches would panic here.
    let (new_splats, stats) = trainer.step(batch, splats).await;
    splats = new_splats;
    assert!(splats.num_splats() > 0);
    let loss: f32 = stats.loss.into_scalar_async().await.expect("loss readback");
    assert!(loss.is_finite());
}
