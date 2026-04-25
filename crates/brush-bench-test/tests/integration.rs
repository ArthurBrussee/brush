//! Integration tests for the benchmark functions
//!
//! These tests verify that the benchmark data generation and core operations work correctly.

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
    let pixel_count = (width * height * 3) as usize;

    let img_data: Vec<f32> = (0..pixel_count)
        .map(|i| {
            let pixel_idx = i / 3;
            let x = (pixel_idx as u32) % width;
            let y = (pixel_idx as u32) / width;
            let channel = i % 3;

            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;

            let base = match channel {
                0 => nx * 0.5 + 0.25,
                1 => ny * 0.5 + 0.25,
                2 => (nx + ny) * 0.25 + 0.5,
                _ => unreachable!(),
            };

            base + (rng.random::<f32>() - 0.5) * 0.05
        })
        .collect();

    let img_tensor = TensorData::new(img_data, [height as usize, width as usize, 3]);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );

    SceneBatch {
        img_tensor,
        alpha_mode: AlphaMode::Transparent,
        camera,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_splat_generation() {
    let device = brush_kernel::test_helpers::test_device().await;
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
    let device = brush_kernel::test_helpers::test_device().await;
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

    assert!(result.render_aux.num_visible > 0, "no splats rendered");
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
    let device = brush_kernel::test_helpers::test_device().await;
    let batch = generate_test_batch((64, 64));
    let splats = generate_test_splats(&device, 500);
    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );
    let (final_splats, stats) = trainer.step(batch, splats).await;

    assert!(final_splats.num_splats() > 0);
    let loss = stats
        .loss
        .into_data_async()
        .await
        .expect("readback")
        .as_slice::<f32>()
        .expect("Wrong type")[0];
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[wasm_bindgen_test(unsupported = test)]
fn test_batch_generation() {
    let batch = generate_test_batch((256, 128));
    let img_dims = batch.img_tensor.shape.dims();
    assert_eq!(img_dims, [128, 256, 3]);
    let img_data = batch.img_tensor.into_vec::<f32>().unwrap();
    assert!(img_data.iter().all(|&x| x.is_finite()));
    assert!(img_data.iter().all(|&x| (0.0..=1.1).contains(&x)));
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_multi_step_training() {
    let device = brush_kernel::test_helpers::test_device().await;
    let batch = generate_test_batch((64, 64));
    let config = TrainConfig::default();
    let mut splats = generate_test_splats(&device, 100);
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    for _ in 0..10 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats).await;
        splats = new_splats;

        let loss = stats
            .loss
            .into_data_async()
            .await
            .expect("readback")
            .as_slice::<f32>()
            .expect("Wrong type")[0];
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    assert!(splats.num_splats() > 0);
}

// Training with a camera pointing away from every splat — num_visible == 0
// every step. The training loop must not crash on this; all gradients should
// be zero (or at least finite) and the optimizer step should be a no-op.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_with_zero_visible_does_not_crash() {
    let device = brush_kernel::test_helpers::test_device().await;
    let splats = generate_test_splats(&device, 200);

    // Camera pointing away from the scene (looking along +Z, scene is at ±5).
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 1000.0),
        Quat::IDENTITY, // looks along -Z in local space → away from origin
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );

    let batch = SceneBatch {
        img_tensor: TensorData::new(vec![0.5f32; 64 * 64 * 3], [64usize, 64, 3]),
        alpha_mode: AlphaMode::Transparent,
        camera,
    };

    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );
    let (new_splats, stats) = trainer.step(batch, splats).await;
    // Should succeed; nothing visible means num_visible ≈ 0.
    assert!(new_splats.num_splats() > 0);
    let loss = stats
        .loss
        .into_data_async()
        .await
        .expect("readback")
        .as_slice::<f32>()
        .expect("Wrong type")[0];
    assert!(
        loss.is_finite(),
        "loss went non-finite with empty render: {loss}"
    );
}

// Training with a deliberately degenerate bounding box (NaN center) used to
// crash in `median_size`. After the fix, training must still proceed without
// panicking — the per-step `lr_mean * median_size` just ends up NaN, which is
// ultimately harmless because any NaN optimizer update is caught by
// `validate_gradient` under the debug-validation feature.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn trainer_tolerates_nan_bounds() {
    let device = brush_kernel::test_helpers::test_device().await;
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
    let (_splats, stats) = trainer.step(batch, splats).await;
    let _ = stats
        .loss
        .into_data_async()
        .await
        .expect("readback")
        .as_slice::<f32>()
        .expect("Wrong type")[0];
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_gradient_validation() {
    let device = brush_kernel::test_helpers::test_device().await;
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
