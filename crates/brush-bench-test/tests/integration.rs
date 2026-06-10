//! Integration tests for the benchmark functions
//!
//! These tests verify that the benchmark data generation and core operations work correctly.

#![allow(clippy::missing_assert_message)]

use brush_dataset::scene::SceneBatch;
use brush_render::gaussian_splats::RenderOptions;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_render_bwd::render_splats;
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::module::AutodiffModule;
use burn::tensor::{Device, TensorData};
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const TEST_SEED: u64 = 12345;

/// Generate small realistic splats for testing
fn generate_test_splats(device: &Device, count: usize) -> Splats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);

    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();

    let log_scales: Vec<f32> = (0..count)
        .flat_map(|_| {
            let base = rng.random_range(0.01..0.1_f32).ln();
            [base, base, base]
        })
        .collect();

    let rotations: Vec<f32> = (0..count)
        .flat_map(|_| {
            let u1 = rng.random::<f32>();
            let u2 = rng.random::<f32>();
            let u3 = rng.random::<f32>();

            let sqrt1_u1 = (1.0 - u1).sqrt();
            let sqrt_u1 = u1.sqrt();
            let theta1 = 2.0 * std::f32::consts::PI * u2;
            let theta2 = 2.0 * std::f32::consts::PI * u3;

            [
                sqrt1_u1 * theta1.sin(),
                sqrt1_u1 * theta1.cos(),
                sqrt_u1 * theta2.sin(),
                sqrt_u1 * theta2.cos(),
            ]
        })
        .collect();

    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
            ]
        })
        .collect();

    let opacities: Vec<f32> = (0..count).map(|_| rng.random_range(0.6..1.0)).collect();

    Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh_coeffs,
        opacities,
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0)
}

fn generate_test_batch(resolution: (u32, u32)) -> SceneBatch {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let (width, height) = resolution;
    let pixel_count = (width * height) as usize;

    let mut byte = |v: f32| -> u32 {
        let v = (v + (rng.random::<f32>() - 0.5) * 0.05).clamp(0.0, 1.0);
        (v * 255.0).round() as u32
    };
    let img_packed_data: Vec<i32> = (0..pixel_count)
        .map(|i| {
            let x = (i as u32) % width;
            let y = (i as u32) / width;
            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;
            let r = byte(nx * 0.5 + 0.25);
            let g = byte(ny * 0.5 + 0.25);
            let b = byte((nx + ny) * 0.25 + 0.5);
            (r | g << 8 | b << 16 | 255 << 24) as i32
        })
        .collect();

    let img_packed = TensorData::new(img_packed_data, [height as usize, width as usize]);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );

    SceneBatch {
        img_packed,
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera,
        depth: None,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_splat_generation() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 1000);

    assert_eq!(splats.num_splats(), 1000);

    // Check that means are reasonable
    let means_data = splats
        .means()
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(means_data.len(), 3000);

    for chunk in means_data.chunks(3) {
        assert!(chunk.iter().all(|&x| x.is_finite()));
        assert!(chunk[0].abs() < 10.0 && chunk[1].abs() < 10.0 && chunk[2].abs() < 20.0);
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_forward_rendering() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 1000);
    assert_eq!(splats.num_splats(), 1000);

    let camera = Camera::new(
        Vec3::new(0.0, 0.0, -8.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let img_size = glam::uvec2(64, 64);
    let result = render_splats(splats, &camera, img_size, RenderOptions::float(), 0.0).await;
    assert!(result.num_visible > 0, "no splats rendered");
    let data = result
        .img
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .expect("Wrong type");
    assert!(data.iter().all(|&v| v.is_finite()));
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_training_step() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    let splats = generate_test_splats(&device, 500);
    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );
    let (final_splats, _stats) = trainer.step(batch, splats).await;

    assert!(final_splats.num_splats() > 0);
}

// Training must actually move the parameters — guards against a broken
// gradient path (e.g. params not require_grad, or grads not registered),
// which would otherwise pass `test_training_step` (it only checks count).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn training_updates_params() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    let splats = generate_test_splats(&device, 200);
    let config = TrainConfig::default();

    let means0: Vec<f32> = splats
        .means()
        .into_data_async()
        .await
        .expect("rb")
        .into_vec::<f32>()
        .expect("vec");
    let sh0: Vec<f32> = splats
        .sh_coeffs
        .val()
        .into_data_async()
        .await
        .expect("rb")
        .into_vec::<f32>()
        .expect("vec");

    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );
    // Mirror the real loop in train_stream: lift inner splats onto the
    // autodiff graph each step, train, then `.valid()` back to inner.
    let mut splats = splats;
    for _ in 0..3 {
        let diff = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats.clone());
        let (s, _) = trainer.step(batch.clone(), diff).await;
        splats = burn::module::AutodiffModule::valid(&s);
    }

    let means1: Vec<f32> = splats
        .means()
        .into_data_async()
        .await
        .expect("rb")
        .into_vec::<f32>()
        .expect("vec");
    let sh1: Vec<f32> = splats
        .sh_coeffs
        .val()
        .into_data_async()
        .await
        .expect("rb")
        .into_vec::<f32>()
        .expect("vec");

    let max_delta = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    };
    let d_means = max_delta(&means0, &means1);
    let d_sh = max_delta(&sh0, &sh1);
    assert!(
        d_means > 1e-6,
        "means did not move after 3 steps (Δ={d_means:.2e}) — mean gradient/LR path broken"
    );
    assert!(
        d_sh > 1e-6,
        "SH color did not move after 3 steps (Δ={d_sh:.2e}) — gradient path broken"
    );
}

#[wasm_bindgen_test(unsupported = test)]
fn test_batch_generation() {
    let batch = generate_test_batch((256, 128));
    let img_dims = batch.img_packed.shape.as_slice();
    assert_eq!(img_dims, &[128, 256]);
    let img_data = batch.img_packed.into_vec::<i32>().unwrap();
    assert_eq!(img_data.len(), 128 * 256);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_multi_step_training() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    let config = TrainConfig::default();
    let mut splats = generate_test_splats(&device, 100);
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    for _ in 0..10 {
        let (new_splats, _) = trainer.step(batch.clone(), splats).await;
        splats = new_splats;
    }
    assert!(splats.num_splats() > 0);
}

// End-to-end PGSR geometry training: a non-zero `depth_normal_weight`
// auto-enables the geometry render pass, and the flattening + depth-normal
// consistency regularizers must produce a finite loss and backprop without
// crashing across several steps.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_with_geometry_losses() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    let config = TrainConfig {
        flatten_weight: 0.1,
        depth_normal_weight: 0.05,
        // Exercise the geometry path immediately.
        geo_from_iter: Some(0),
        ..Default::default()
    };
    assert!(config.needs_geometry(0));

    let mut splats = generate_test_splats(&device, 200);
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    for step in 0..5 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats).await;
        splats = new_splats;
        let loss = stats
            .loss
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");
        assert!(loss.is_finite(), "non-finite loss at step {step}: {loss}");
    }
    assert!(splats.num_splats() > 0);
}

// End-to-end depth-distortion (2DGS L_d, squared form): a non-zero
// `distortion_weight` auto-enables the geometry pass and adds the depth-moment
// variance term. Must stay finite and backprop without crashing (exercises the
// new zz render channel + its VJP).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_with_distortion_loss() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    let config = TrainConfig {
        distortion_weight: 0.1,
        geo_from_iter: Some(0),
        ..Default::default()
    };
    assert!(config.needs_geometry(0));

    let mut splats = generate_test_splats(&device, 200);
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    for step in 0..5 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats).await;
        splats = new_splats;
        let loss = stats
            .loss
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");
        assert!(loss.is_finite(), "non-finite loss at step {step}: {loss}");
    }
    assert!(splats.num_splats() > 0);
}

// End-to-end metric depth supervision: a batch carrying a (synthetic) low-res
// LiDAR depth + confidence drives the depth loss against the RaDe-GS rendered
// depth. Must stay finite and backprop without crashing (exercises the GT
// upsample + z->radial conversion + masked L1).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_with_depth_loss() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut batch = generate_test_batch((64, 64));
    let n = 16 * 16;
    batch.depth = Some(brush_dataset::scene::DepthSample {
        depth: TensorData::new(vec![2.0f32; n], [16, 16, 1]),
        conf: TensorData::new(vec![1.0f32; n], [16, 16, 1]),
    });

    let config = TrainConfig {
        depth_loss_weight: 0.5,
        geo_from_iter: Some(0),
        ..Default::default()
    };
    assert!(config.needs_geometry(0));

    let mut splats = generate_test_splats(&device, 200);
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    for step in 0..5 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats).await;
        splats = new_splats;
        let loss = stats
            .loss
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");
        assert!(loss.is_finite(), "non-finite loss at step {step}: {loss}");
    }
    assert!(splats.num_splats() > 0);
}

// Training with a camera pointing away from every splat — num_visible == 0
// every step. The training loop must not crash on this; all gradients should
// be zero (or at least finite) and the optimizer step should be a no-op.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_with_zero_visible_does_not_crash() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 200);

    // Camera pointing away from the scene (looking along +Z, scene is at ±5).
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 1000.0),
        Quat::IDENTITY, // looks along -Z in local space → away from origin
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );

    let pixel = 0x80808080u32 as i32; // mid-grey, opaque; bit-cast to i32 for the dispatch backend
    let batch = SceneBatch {
        img_packed: TensorData::new(vec![pixel; 64 * 64], [64usize, 64]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera,
        depth: None,
    };

    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );
    let (new_splats, _stats) = trainer.step(batch, splats).await;
    // Should succeed; nothing visible means num_visible ≈ 0.
    assert!(new_splats.num_splats() > 0);
}

// Training with a deliberately degenerate bounding box (NaN center) used to
// crash in `median_size`. After the fix, training must still proceed without
// panicking — the per-step `lr_mean * median_size` just ends up NaN, which is
// ultimately harmless because any NaN optimizer update is caught by
// `validate_gradient` under the debug-validation feature.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn trainer_tolerates_nan_bounds() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 100);
    let config = TrainConfig::default();

    // A degenerate bounds with one NaN axis. Before the `total_cmp` fix, this
    // panicked inside `median_size()` on the first `step` call.
    let bounds = BoundingBox {
        center: Vec3::ZERO,
        extent: Vec3::new(f32::NAN, 1.0, 1.0),
    };
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    let batch = generate_test_batch((64, 64));
    let (_splats, _stats) = trainer.step(batch, splats).await;
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_gradient_validation() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 100);

    // Create a simple loss by rendering and taking the mean
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let img_size = glam::uvec2(64, 64);

    // Clone splats since render_splats takes ownership and we need splats for gradient validation
    let result = render_splats(
        splats.clone(),
        &camera,
        img_size,
        RenderOptions::float(),
        0.0,
    )
    .await;
    splats.bwd_validate(result.img.mean()).await;
}

// One trainer + many parallel viewers.
// Test whether multithreading produces any issues.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "multi_thread")]
async fn stress_concurrent_train_and_view() {
    use brush_async::Actor;
    use brush_render::gaussian_splats::render_splats as render_splats_fwd;
    use tokio::sync::watch;

    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(64, 64);

    let viewer_count = 6;
    let train_steps = 100;
    let viewer_iters_per_task = 10;

    let initial = generate_test_splats(&device, 500);
    let (tx, rx) = watch::channel::<Splats>(initial.clone().valid());

    let trainer_actor = Actor::new("test-trainer");
    let device_c = device.clone();
    let mut splats = initial;
    let trainer_done = trainer_actor.run(move || async move {
        let batch = generate_test_batch((64, 64));
        let config = TrainConfig::default();
        let mut trainer = SplatTrainer::new(
            &config,
            &device_c,
            BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
        );
        for _ in 0..train_steps {
            let (new_splats, _) = trainer.step(batch.clone(), splats).await;
            splats = new_splats;
            let _ = tx.send(splats.valid());
        }
    });

    let mut viewer_actors = Vec::with_capacity(viewer_count);
    let mut viewer_dones = Vec::with_capacity(viewer_count);
    for v in 0..viewer_count {
        let actor = Actor::new(&format!("test-viewer-{v}"));
        let mut rx = rx.clone();
        let done = actor.run(move || async move {
            let camera = Camera::new(
                Vec3::new(0.0, 0.0, -5.0 - v as f32 * 0.3),
                Quat::IDENTITY,
                45.0,
                45.0,
                glam::vec2(0.5, 0.5),
                Pinhole,
            );
            for _ in 0..viewer_iters_per_task {
                let snap = rx.borrow_and_update().clone();
                render_splats_fwd(snap, &camera, img_size, RenderOptions::float()).await;
            }
        });
        viewer_actors.push(actor);
        viewer_dones.push(done);
    }

    trainer_done.await;
    for d in viewer_dones {
        d.await;
    }
    drop(viewer_actors);
}
