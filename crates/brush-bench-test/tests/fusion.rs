//! Guards the fusion behaviour the kernel-boundary work bought.
//!
//! The backward writes one gradient row per visible splat and the render
//! backward expands it with a gather. What that should cost is exactly one
//! fused kernel per parameter: no zero-filled dense buffer, and no separate
//! mask multiply of the kind the visibility masks used to need. burn's
//! `FusionInspector` reports what actually fused, so pin it down here.
//!
//! Note burn keeps each gather in its own block rather than folding it into
//! the optimizer's kernel, so a dense gradient per parameter is still
//! written once. That is the current behaviour, not the goal.

#![cfg(not(target_family = "wasm"))]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::backend::ir::{BaseOperationIr, OperationIr};
use burn::tensor::{Device, TensorData};
use burn_fusion::inspect::{FusionInspector, FusionReport};
use burn_fusion::stream::StreamId;
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};

const TEST_SEED: u64 = 12345;

fn test_splats(device: &Device, count: usize) -> Splats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(1.0..5.0),
            ]
        })
        .collect();
    let rots: Vec<f32> = (0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let log_scales: Vec<f32> = (0..count).flat_map(|_| [-2.0, -2.0, -2.0]).collect();
    let coeffs: Vec<f32> = (0..count).flat_map(|_| [0.5, 0.5, 0.5]).collect();
    let opacities: Vec<f32> = (0..count).map(|_| 0.5).collect();
    Splats::from_raw(
        means,
        rots,
        log_scales,
        coeffs,
        opacities,
        SplatRenderMode::Default,
        device,
    )
}

fn test_batch(width: u32, height: u32) -> SceneBatch {
    let pixels = (width * height) as usize;
    let img: Vec<i32> = (0..pixels)
        .map(|i| {
            let v = (i % 200) as u32;
            (v | v << 8 | v << 16 | 255 << 24) as i32
        })
        .collect();
    SceneBatch {
        img_packed: TensorData::new(img, [height as usize, width as usize]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera: Camera::new(
            Vec3::new(0.0, 0.0, 3.0),
            Quat::IDENTITY,
            45.0,
            45.0,
            glam::vec2(0.5, 0.5),
            Pinhole,
        ),
    }
}

/// The compact-to-dense gradient expansion.
fn is_select(op: &OperationIr) -> bool {
    matches!(op, OperationIr::BaseFloat(BaseOperationIr::Select(_)))
}

/// `transforms`, `sh_coeffs`, `raw_opacities`, the refine-weight holder, and
/// the SH second moment, which is reduced compact and gathered the same way.
const GRADIENT_EXPANSIONS: usize = 5;

#[tokio::test]
async fn gradient_gathers_fuse_into_their_consumer() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    // Warm up first: the optimizer builds its state on step one, which is not
    // the steady state we care about.
    let mut splats = test_splats(&device, 256);
    for _ in 0..2 {
        (splats, _) = trainer.step(test_batch(32, 32), splats).await;
    }

    let inspector = FusionInspector::install(StreamId::current());
    let (splats, _stats) = trainer.step(test_batch(32, 32), splats).await;
    // Force the queue to run so every plan is reported.
    let _ = splats.means().into_data_async().await.expect("readback");

    let reports = inspector.drain();
    assert!(
        !reports.is_empty(),
        "inspector saw no execution plans; is the step running on this stream?"
    );

    let mut gathers = 0;
    let mut unfused = Vec::new();
    for report in &reports {
        for block in &report.blocks {
            let selects = block.operations.iter().filter(|op| is_select(op)).count();
            gathers += selects;
            if selects > 0 && block.fuser_name().is_none() {
                unfused.push(report.format_table());
            }
        }
    }

    assert!(
        unfused.is_empty(),
        "a gradient gather ran unfused:\n{}",
        unfused.join("\n")
    );
    assert_eq!(
        gathers,
        GRADIENT_EXPANSIONS,
        "expected one gather per parameter; more means the expansion grew \
         extra ops, fewer means a gradient stopped reaching its parameter:\n{}",
        reports
            .iter()
            .map(FusionReport::format_table)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Opt-in exploration, not a pinned launch-count contract. Run in isolation:
/// `cargo test -p brush-bench-test --test fusion explore_ -- --ignored --nocapture --test-threads=1`
#[tokio::test]
#[ignore = "prints device-dependent fusion plans for investigation"]
async fn explore_tensor_fusion() {
    tensor_fusion_probes(&[
        "square_sum_dims",
        "mul_sum",
        "abs_sum",
        "square_sum_reverse",
        "flatten_before_square",
        "flatten_after_square",
        "select_consumer",
        "select_consumer_independent",
        "select_interleaved",
        "noise_slice_assign",
        "noise_cat",
        "noise_gate_original",
        "noise_gate_rank_first",
    ])
    .await;
}

#[tokio::test]
async fn rank_first_noise_matches_reference() {
    tensor_fusion_probes(&["noise_gate_original", "noise_gate_rank_first"]).await;
}

async fn tensor_fusion_probes(cases: &[&str]) {
    use burn::tensor::{Int, Tensor, s};

    let device = Device::from(brush_cube::test_helpers::test_device().await);
    let input = Tensor::<3>::from_data(
        TensorData::new(
            (0..256 * 16 * 3)
                .map(|i| (i % 97) as f32 / 97.0 - 0.5)
                .collect::<Vec<_>>(),
            [256, 16, 3],
        ),
        &device,
    );
    let indices = Tensor::<1, Int>::from_data(
        TensorData::new((0..256).rev().collect::<Vec<i32>>(), [256]),
        &device,
    );
    let opacities = Tensor::<1>::from_floats(
        (0..256)
            .map(|i| [0.0, 0.001, 0.5, 1.0][i % 4])
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );
    let visible = Tensor::<1>::from_floats(
        (0..256)
            .map(|i| if i % 3 == 2 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );
    let momentum = Tensor::<2>::from_data(TensorData::new(vec![0.1f32; 2560], [256, 10]), &device);
    let flat_input = input.clone().reshape([256, 48]);
    let transforms = input.clone().reshape([256, 48]).slice(s![.., 0..10]);
    let noise = transforms.clone().slice(s![.., 0..3]);
    // Materialize inputs before inspection so setup isn't counted.
    let _ = transforms.clone().into_data_async().await.unwrap();
    let _ = noise.clone().into_data_async().await.unwrap();

    let mut reference = None;
    for &case in cases {
        let inspector = FusionInspector::install(StreamId::current());
        let result: Tensor<2> = match case {
            "square_sum_dims" => input
                .clone()
                .powi_scalar(2)
                .sum_dims(&[1, 2])
                .reshape([256, 1]),
            "mul_sum" => (flat_input.clone() * flat_input.clone()).sum_dim(1),
            "abs_sum" => flat_input.clone().abs().sum_dim(1),
            "square_sum_reverse" => input
                .clone()
                .powi_scalar(2)
                .sum_dim(2)
                .sum_dim(1)
                .reshape([256, 1]),
            "flatten_before_square" => input.clone().reshape([256, 48]).powi_scalar(2).sum_dim(1),
            "flatten_after_square" => input.clone().powi_scalar(2).reshape([256, 48]).sum_dim(1),
            "select_consumer" => {
                transforms.clone().select(0, indices.clone()) * 0.9 + transforms.clone() * 0.1
            }
            "select_consumer_independent" => {
                transforms.clone().select(0, indices.clone()) * 0.9 + momentum.clone() * 0.1
            }
            "select_interleaved" => {
                let gathered = transforms.clone().select(0, indices.clone());
                let other = noise.clone().select(0, indices.clone());
                let result = gathered * 0.9 + momentum.clone() * 0.1;
                let other = other * 0.5;
                let _ = result.clone().into_data_async().await.unwrap();
                let _ = other.into_data_async().await.unwrap();
                result
            }
            "noise_slice_assign" => {
                let updated = transforms.clone().slice(s![.., 0..3]) + noise.clone() * 0.01;
                transforms.clone().slice_assign(s![.., 0..3], updated)
            }
            "noise_cat" => {
                let updated = transforms.clone().slice(s![.., 0..3]) + noise.clone() * 0.01;
                Tensor::cat(vec![updated, transforms.clone().slice(s![.., 3..10])], 1)
            }
            "noise_gate_original" => {
                let gate = (1.0f32 - opacities.clone())
                    .powi_scalar(150.0)
                    .clamp(0.0, 1.0)
                    * visible.clone();
                let gate = gate.unsqueeze_dim(1) * 4.0;
                let noise = (noise.clone() * gate).clamp(-0.25, 0.25);
                let updated = transforms.clone().slice(s![.., 0..3]) + noise;
                transforms.clone().slice_assign(s![.., 0..3], updated)
            }
            "noise_gate_rank_first" => {
                // Put view operations before the elementwise chain.
                let means = transforms.clone().slice(s![.., 0..3]);
                let opacity = opacities.clone().unsqueeze_dim::<2>(1);
                let visible = visible.clone().unsqueeze_dim::<2>(1);
                let gate = (1.0f32 - opacity).powi_scalar(150.0).clamp(0.0, 1.0) * visible * 4.0;
                let updated = means + (noise.clone() * gate).clamp(-0.25, 0.25);
                transforms.clone().slice_assign(s![.., 0..3], updated)
            }
            _ => unreachable!(),
        };
        let values = result
            .into_data_async()
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let reports = inspector.drain();
        let blocks: usize = reports.iter().map(|r| r.blocks.len()).sum();
        println!("\n=== {case}: {blocks} execution blocks ===");
        for report in reports {
            println!("{}", report.format_table());
        }
        if case == "square_sum_dims"
            || case == "noise_slice_assign"
            || case == "noise_gate_original"
        {
            reference = Some(values);
        } else if case.starts_with("flatten")
            || case == "mul_sum"
            || case == "square_sum_reverse"
            || case == "noise_cat"
            || case == "noise_gate_rank_first"
        {
            let reference = reference.as_ref().unwrap();
            assert_eq!(reference.len(), values.len());
            for (actual, expected) in values.iter().zip(reference) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{case}: {actual} != {expected}"
                );
            }
        }
    }
}

#[tokio::test]
#[ignore = "prints a steady-state training plan without validation readbacks"]
async fn explore_training_fusion() {
    use burn::module::Module;

    // Validation readbacks change execution boundaries. Run in isolation.
    brush_render::validation::set_enabled(false);
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut trainer = SplatTrainer::new(
        &TrainConfig::default(),
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );
    let mut splats = test_splats(&device, 256).with_sh_degree(3).train();
    for _ in 0..2 {
        (splats, _) = trainer.step(test_batch(32, 32), splats).await;
    }
    let _ = splats.transforms.val().into_data_async().await.unwrap();
    let inspector = FusionInspector::install(StreamId::current());
    let (splats, stats) = trainer.step(test_batch(32, 32), splats).await;
    let _ = splats.transforms.val().into_data_async().await.unwrap();
    let _ = splats.sh_coeffs.val().into_data_async().await.unwrap();
    let _ = splats.raw_opacities.val().into_data_async().await.unwrap();
    let _ = stats.loss.into_data_async().await.unwrap();
    let reports = inspector.drain();
    assert!(!reports.is_empty());
    let gathers = reports
        .iter()
        .flat_map(|report| &report.blocks)
        .flat_map(|block| &block.operations)
        .filter(|op| is_select(op))
        .count();
    assert_eq!(
        gathers, GRADIENT_EXPANSIONS,
        "the diagnostic must exercise every gradient expansion"
    );
    let blocks: usize = reports.iter().map(|r| r.blocks.len()).sum();
    println!("\n=== degree-3 steady-state training: {blocks} execution blocks ===");
    for report in reports {
        println!("{}", report.format_table());
    }
}

/// Synthetic end-to-end timing, including host work and GPU synchronization.
/// Run separately from the inspector tests to avoid instrumentation overhead.
#[tokio::test]
#[ignore = "manual before/after training timing"]
async fn time_training_steps() {
    use burn::module::Module;
    use std::time::Instant;

    brush_render::validation::set_enabled(false);
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = test_batch(256, 256);
    for trial in 0..3 {
        device.seed(TEST_SEED);
        let mut trainer = SplatTrainer::new(
            &TrainConfig::default(),
            &device,
            BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
        );
        let mut splats = test_splats(&device, 100_000).with_sh_degree(3).train();
        for _ in 0..10 {
            (splats, _) = trainer.step(batch.clone(), splats).await;
        }
        device.sync().expect("warmup sync");
        let start = Instant::now();
        for _ in 0..30 {
            (splats, _) = trainer.step(batch.clone(), splats).await;
        }
        device.sync().expect("timing sync");
        println!(
            "trial {trial}: {:.3} ms/step (100k splats, degree 3, 256x256)",
            start.elapsed().as_secs_f64() * 1000.0 / 30.0
        );
    }
}
