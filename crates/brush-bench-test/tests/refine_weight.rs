//! Characterizes the densification-weight (`refine_weight_holder`
//! gradient) path that historically caused the "Failed to sample from
//! weights" crash (issue #128).
//!
//! Two invariants are demonstrated:
//!
//! 1. `color_is_clamped` — the SH color path IS scrubbed+clamped to ±100
//!    in `project_visible` ("so the rasterize backward's gradient term
//!    can't amplify past f32 range").
//!
//! 2. `refine_weight_stays_finite_after_fix` — the refine weight, which
//!    sits *right next to* that color clamp but was passed through raw by
//!    `project_backwards`, must also be finite. With the source clamp in
//!    `project_backwards` it stays finite even for aggressive anisotropic
//!    / oblique states; this is the value `gather_stats` latches and
//!    `multinomial_sample` consumes.
//!
//! Note: the projected *conic* is bounded (~3.3 non-mip) because
//! `compensate_cov2d` adds +0.3 to the cov2d diagonal, flooring its
//! eigenvalues. The unbounded factors in `grad.refine` are the
//! `1/max(final_a,1e-5)` (≤1e5×) low-alpha amplifier, cross-tile atomic
//! accumulation, and divergent projected means — not the conic.

use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use burn::tensor::Tensor;

/// Aggressive-but-finite anisotropic scene through the real autodiff
/// backward; returns (max|refine_weight|, non_finite_count).
async fn refine_weight_stats(ls: [f32; 3], img: glam::UVec2, n: usize) -> (f32, u64) {
    let device = brush_cube::test_helpers::test_device().await;
    let device_d = burn::tensor::Device::from(device.clone()).autodiff();
    let cam = Camera::new(
        glam::vec3(0.6, -0.4, -0.3),
        glam::Quat::from_axis_angle(glam::vec3(0.3, 1.0, 0.2).normalize(), 0.9),
        0.8,
        0.8,
        glam::vec2(0.5, 0.5),
    );
    let mut means = Vec::new();
    let mut rots = Vec::new();
    let mut lsv = Vec::new();
    let mut dc = Vec::new();
    let mut opac = Vec::new();
    for i in 0..n {
        means.extend_from_slice(&[0.02 * i as f32, -0.01 * i as f32, 3.0]);
        let q = [0.4_f32, 0.5, -0.6, 0.48];
        let nrm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        rots.extend_from_slice(&[q[0] / nrm, q[1] / nrm, q[2] / nrm, q[3] / nrm]);
        lsv.extend_from_slice(&ls);
        dc.extend_from_slice(&[0.9, 0.1, 0.8]);
        opac.push(3.0);
    }
    let splats = Splats::from_raw(means, rots, lsv, dc, opac, SplatRenderMode::Default, &device_d);
    let diff = brush_render_bwd::render_splats(splats, &cam, img, glam::Vec3::ZERO).await;
    // High-contrast loss → O(1)/pixel v_output.
    let grads = diff.img.clone().abs().sum().backward();
    let rw = diff
        .refine_weight_holder
        .grad(&grads)
        .expect("refine weight gradient must exist");
    let v = rw
        .into_data_async()
        .await
        .expect("read refine weight")
        .into_vec::<f32>()
        .expect("f32");
    let mut max_abs = 0.0f32;
    let mut non_finite = 0u64;
    for x in &v {
        if !x.is_finite() {
            non_finite += 1;
        } else {
            max_abs = max_abs.max(x.abs());
        }
    }
    (max_abs, non_finite)
}

#[tokio::test]
async fn refine_weight_stays_finite_after_fix() {
    let mut worst = 0.0f32;
    let mut nf_total = 0u64;
    for &ls_hi in &[6.0_f32, 9.0, 12.0, 15.0] {
        for &img in &[glam::uvec2(256, 256), glam::uvec2(1024, 1024)] {
            let (max_abs, nf) = refine_weight_stats([ls_hi, -18.0, -18.0], img, 64).await;
            println!(
                "ls_hi={ls_hi:>5} img={:>4} -> max|refine_weight|={max_abs:.3e} non_finite={nf}",
                img.x
            );
            worst = worst.max(max_abs);
            nf_total += nf;
        }
    }
    println!("WORST max|refine_weight| = {worst:.3e}; total non_finite = {nf_total}");
    // The project_backwards source clamp must keep this finite — it is
    // latched by gather_stats and consumed by multinomial_sample.
    assert_eq!(nf_total, 0, "refine_weight went non-finite after the fix");
    assert!(
        worst <= 1.0e16,
        "refine_weight exceeded the source clamp cap: {worst:.3e}"
    );
}

/// The color path is clamped; this is the established pattern the refine
/// weight should match. Holds with or without the refine-weight fix.
#[tokio::test]
async fn color_is_clamped() {
    let device = brush_cube::test_helpers::test_device().await;
    let device_d = burn::tensor::Device::from(device.clone()).autodiff();
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let n = 16;
    let means: Vec<f32> = (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect();
    let rots: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let ls: Vec<f32> = (0..n).flat_map(|_| [0.0, 0.0, 0.0]).collect();
    let dc: Vec<f32> = (0..n).flat_map(|_| [1e6, -1e6, 1e6]).collect();
    let opac = vec![3.0f32; n];
    let splats = Splats::from_raw(means, rots, ls, dc, opac, SplatRenderMode::Default, &device_d);
    let diff =
        brush_render_bwd::render_splats(splats, &cam, glam::uvec2(128, 128), glam::Vec3::ZERO)
            .await;
    let img: Tensor<3> = diff.img.clone();
    let maxc = img
        .abs()
        .max()
        .into_scalar_async::<f32>()
        .await
        .expect("scalar");
    println!("max rendered channel with 1e6 SH DC = {maxc:.3}");
    assert!(maxc <= 101.0, "color path unexpectedly unclamped: {maxc}");
}
