//! The SH second moment is reduced off the compact gradient rows and gathered,
//! rather than squaring the dense `[N, coeffs, 3]` gradient and summing it
//! away. That is only sound if it equals the dense reduction exactly, so check
//! the two against each other.

#![cfg(not(target_family = "wasm"))]

use brush_render::bwd::render_splats;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::{camera::Camera, kernels::camera_model::CameraModel::Pinhole};
use burn::module::Module;
use burn::tensor::Device;
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};

fn test_splats(device: &Device, count: usize) -> Splats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-1.5..1.5),
                rng.random_range(-1.5..1.5),
                rng.random_range(1.0..4.0),
            ]
        })
        .collect();
    let rots: Vec<f32> = (0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let log_scales: Vec<f32> = (0..count).flat_map(|_| [-2.5, -2.5, -2.5]).collect();
    // Degree 1: four coefficients, so the reduction spans 12 values per splat.
    let coeffs: Vec<f32> = (0..count)
        .flat_map(|_| (0..12).map(|i| 0.1 * (i as f32) - 0.4))
        .collect();
    let opacities: Vec<f32> = (0..count).map(|_| 0.4).collect();
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

#[tokio::test]
async fn compact_second_moment_matches_the_dense_reduction() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = test_splats(&device, 512).train();
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 0.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );

    let out = render_splats(splats.clone(), &camera, glam::uvec2(48, 48), Vec3::ZERO).await;
    // Any scalar with a gradient will do; the point is the SH path.
    let loss = out.img.clone().powi_scalar(2).sum();
    let mut grads = loss.backward();

    let reduced = out
        .coeffs_grad_sq_holder
        .grad_remove(&mut grads)
        .expect("the backward registers the SH second moment")
        .without_autodiff();
    let dense = splats
        .sh_coeffs
        .val()
        .grad_remove(&mut grads)
        .expect("SH coefficients take a gradient")
        .without_autodiff();

    let [n, coeffs, ch] = dense.dims();
    let expected = dense.clone().powi_scalar(2).sum_dims(&[1, 2]) / (coeffs * ch) as f32;
    assert_eq!(reduced.dims(), [n, 1, 1]);

    let got: Vec<f32> = reduced
        .reshape([n as i32])
        .into_data_async()
        .await
        .expect("readback")
        .try_to_vec()
        .expect("vec");
    let want: Vec<f32> = expected
        .reshape([n as i32])
        .into_data_async()
        .await
        .expect("readback")
        .try_to_vec()
        .expect("vec");

    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "no SH gradient reached the test");
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    assert!(
        worst <= 1e-6 * scale.max(1e-6),
        "compact reduction drifted from the dense one: worst {worst:e}, scale {scale:e}"
    );
}
