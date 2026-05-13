//! Integration tests for the benchmark functions
//!
//! These tests verify that the benchmark data generation and core operations work correctly.

#![allow(
    clippy::missing_assert_message,
    clippy::print_stderr,
    // Stress test holds FUSION_LOCK across .awaits intentionally — see
    // FUSION_LOCK doc in brush-render::burn_glue.
    clippy::await_holding_lock,
)]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, MainBackend,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
};
use brush_render_bwd::render_splats;
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::{
    backend::{Autodiff, wgpu::WgpuDevice},
    module::AutodiffModule,
    tensor::{Tensor, TensorData, TensorPrimitive},
};
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

type DiffBackend = Autodiff<MainBackend>;

const TEST_SEED: u64 = 12345;

/// Generate small realistic splats for testing
fn generate_test_splats(device: &WgpuDevice, count: usize) -> Splats<DiffBackend> {
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

    Splats::<DiffBackend>::from_raw(
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
    let img_packed_data: Vec<u32> = (0..pixel_count)
        .map(|i| {
            let x = (i as u32) % width;
            let y = (i as u32) / width;
            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;
            let r = byte(nx * 0.5 + 0.25);
            let g = byte(ny * 0.5 + 0.25);
            let b = byte((nx + ny) * 0.25 + 0.5);
            r | g << 8 | b << 16 | 255 << 24
        })
        .collect();

    let img_packed = TensorData::new(img_packed_data, [height as usize, width as usize]);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );

    SceneBatch {
        img_packed,
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_splat_generation() {
    let device = brush_cube::test_helpers::test_device().await;
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
    let device = brush_cube::test_helpers::test_device().await;
    let splats = generate_test_splats(&device, 1000);
    assert_eq!(splats.num_splats(), 1000);

    let camera = Camera::new(
        Vec3::new(0.0, 0.0, -8.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(64, 64);
    let result = render_splats(splats, &camera, img_size, Vec3::ZERO).await;
    assert!(result.num_visible > 0, "no splats rendered");
    let pixels: Tensor<DiffBackend, 3> = Tensor::from_primitive(TensorPrimitive::Float(result.img));
    let data = pixels
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .expect("Wrong type");
    assert!(data.iter().all(|&v| v.is_finite()));
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_training_step() {
    let device = brush_cube::test_helpers::test_device().await;
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

#[wasm_bindgen_test(unsupported = test)]
fn test_batch_generation() {
    let batch = generate_test_batch((256, 128));
    let img_dims = batch.img_packed.shape.as_slice();
    assert_eq!(img_dims, &[128, 256]);
    let img_data = batch.img_packed.into_vec::<u32>().unwrap();
    assert_eq!(img_data.len(), 128 * 256);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_multi_step_training() {
    let device = brush_cube::test_helpers::test_device().await;
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

// Training with a camera pointing away from every splat — num_visible == 0
// every step. The training loop must not crash on this; all gradients should
// be zero (or at least finite) and the optimizer step should be a no-op.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_with_zero_visible_does_not_crash() {
    let device = brush_cube::test_helpers::test_device().await;
    let splats = generate_test_splats(&device, 200);

    // Camera pointing away from the scene (looking along +Z, scene is at ±5).
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 1000.0),
        Quat::IDENTITY, // looks along -Z in local space → away from origin
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );

    let pixel = 0x80808080u32; // mid-grey, opaque
    let batch = SceneBatch {
        img_packed: TensorData::new(vec![pixel; 64 * 64], [64usize, 64]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera,
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
    let device = brush_cube::test_helpers::test_device().await;
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
    let device = brush_cube::test_helpers::test_device().await;
    let splats = generate_test_splats(&device, 100);

    // Create a simple loss by rendering and taking the mean
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(64, 64);

    // Clone splats since render_splats takes ownership and we need splats for gradient validation
    let result = render_splats(splats.clone(), &camera, img_size, Vec3::ZERO).await;
    let rendered: Tensor<DiffBackend, 3> =
        Tensor::from_primitive(TensorPrimitive::Float(result.img));
    splats.bwd_validate(rendered.mean()).await;
}

// One trainer + many parallel viewers, each pinned to its own
// `brush_async::Actor`. This is brush's production pattern (one
// trainer that publishes snapshots; UI / export / eval consumers
// clone and render in parallel).
//
// Categorizes failures: `Should have handle for tensor ...` panics on
// burn-fusion's `DSU-*` thread vs silent NaN values in
// `projected_splats`. Both come from the same upstream cross-stream
// race in `burn-fusion`'s `HandleContainer` coordination; we work
// around it brush-side with `brush_render::burn_glue::FUSION_LOCK`.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "multi_thread")]
async fn stress_concurrent_train_and_view() {
    use brush_async::Actor;
    use brush_render::TextureMode;
    use brush_render::gaussian_splats::render_splats as render_splats_fwd;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::watch;

    let device = brush_cube::test_helpers::test_device().await;
    let img_size = glam::uvec2(64, 64);

    // Scale up the work in *this* run rather than running the test
    // many times. More iters = more fusion register ops = more
    // chances to hit the race.
    let viewer_count = 16;
    let train_steps = 200;
    let viewer_iters_per_task = 300;

    let initial = generate_test_splats(&device, 500);
    let (tx, rx) = watch::channel::<Splats<MainBackend>>(initial.clone().valid());

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
            // Hold FUSION_LOCK across the whole iteration: step
            // (forward + backward + optimizer) AND publish. The
            // publish drops the channel's previously-held splats,
            // which otherwise schedules cross-stream tensor drops
            // concurrently with viewer renders.
            #[allow(clippy::await_holding_lock)]
            let _g = brush_render::burn_glue::FUSION_LOCK.lock();
            let (new_splats, _) = trainer.step(batch.clone(), splats).await;
            splats = new_splats;
            let _ = tx.send(splats.valid());
        }
    });

    // Per-failure-mode counters incremented by the viewer tasks (via
    // panic::catch_unwind around each render).
    let nan_panics = Arc::new(AtomicUsize::new(0));
    let handle_panics = Arc::new(AtomicUsize::new(0));
    let other_panics = Arc::new(AtomicUsize::new(0));

    let mut viewer_actors = Vec::with_capacity(viewer_count);
    let mut viewer_dones = Vec::with_capacity(viewer_count);
    for v in 0..viewer_count {
        let actor = Actor::new(&format!("test-viewer-{v}"));
        let mut rx = rx.clone();
        let nan = nan_panics.clone();
        let handle = handle_panics.clone();
        let other = other_panics.clone();
        let done = actor.run(move || async move {
            let camera = Camera::new(
                Vec3::new(0.0, 0.0, -5.0 - v as f32 * 0.3),
                Quat::IDENTITY,
                45.0,
                45.0,
                glam::vec2(0.5, 0.5),
            );
            for _ in 0..viewer_iters_per_task {
                // Hold FUSION_LOCK across the whole iteration so the
                // snap clone, the render, AND the snap's drop happen
                // serially relative to other fusion users.
                #[allow(clippy::await_holding_lock)]
                let _g = brush_render::burn_glue::FUSION_LOCK.lock();
                let snap: Splats<MainBackend> = rx.borrow_and_update().clone();
                let fut = render_splats_fwd(
                    snap,
                    &camera,
                    img_size,
                    Vec3::ZERO,
                    None,
                    TextureMode::Float,
                );
                let result =
                    futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(fut)).await;
                if let Err(payload) = result {
                    let msg = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|&s| s.to_owned()))
                        .unwrap_or_default();
                    if msg.contains("NaN") {
                        nan.fetch_add(1, Ordering::Relaxed);
                    } else if msg.contains("Should have handle for tensor")
                        || msg.contains("Result::unwrap")
                        || msg.contains("CallError")
                    {
                        handle.fetch_add(1, Ordering::Relaxed);
                    } else {
                        other.fetch_add(1, Ordering::Relaxed);
                        eprintln!("[other panic] {msg}");
                    }
                }
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

    let nan = nan_panics.load(Ordering::Relaxed);
    let handle = handle_panics.load(Ordering::Relaxed);
    let other = other_panics.load(Ordering::Relaxed);
    eprintln!(
        "stress_concurrent_train_and_view: NaN={nan} handle-race={handle} other={other} \
         (out of {} render calls)",
        viewer_count * viewer_iters_per_task,
    );
    assert_eq!(
        nan + handle + other,
        0,
        "{nan} NaN panics, {handle} handle-race panics, {other} other panics"
    );
}
