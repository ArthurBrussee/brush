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
async fn gradient_expansions_are_fused() {
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
