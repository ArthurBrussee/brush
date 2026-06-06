//! Forward-render reference test.
//!
//! Each `<name>.safetensors` in `test_cases/` holds the splat input
//! data (means, quats, scales, coeffs, opacities) plus an `out_img`
//! tensor — the ground-truth forward render produced by an external
//! gsplat implementation.
//!
//! We don't compare backward gradients against the gsplat reference —
//! it does `dirs = dirs.detach()` before SH eval and so misses the
//! viewdir→mean path that we fixed. Backward self-consistency is
//! verified by the finite-diff suite in `tests/finite_diff.rs`.

use brush_render::{
    TextureMode,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::{Splats, render_splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use burn::tensor::Tensor;
use safetensors::SafeTensors;
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use crate::safetensor_utils::{safetensor_to_burn, splats_from_safetensors};

const CASES: &[(&str, &[u8])] = &[
    (
        "tiny_case",
        include_bytes!("../test_cases/tiny_case.safetensors"),
    ),
    (
        "basic_case",
        include_bytes!("../test_cases/basic_case.safetensors"),
    ),
    #[allow(clippy::large_include_file)] // reference data
    (
        "mix_case",
        include_bytes!("../test_cases/mix_case.safetensors"),
    ),
];

/// Per-element tolerance: 1e-5 absolute, 1% relative. The reference path
/// uses the hard alpha cutoff (the C^1 smoothstep variant is selected
/// only by the finite-diff test suite via `RasterPass::BackwardSmoothCutoff`)
/// so the rasterizer matches the gsplat reference up to f32 / dispatch
/// nondeterminism, not the ~0.7/255 ramp from smooth cutoff.
async fn assert_img_matches(name: &str, ours: Tensor<3>, reference: Tensor<3>) {
    const ATOL: f32 = 1e-5;
    const RTOL: f32 = 1e-2;
    assert_eq!(ours.dims(), reference.dims(), "{name} shape mismatch");
    let ours = ours
        .into_data_async()
        .await
        .expect("readback ours")
        .into_vec::<f32>()
        .expect("vec ours");
    let reference = reference
        .into_data_async()
        .await
        .expect("readback reference")
        .into_vec::<f32>()
        .expect("vec reference");
    for (i, (&a, &b)) in ours.iter().zip(&reference).enumerate() {
        assert!(
            !a.is_nan() && !b.is_nan(),
            "{name} NaN at idx {i}: ours={a} ref={b}"
        );
        let tol = ATOL + RTOL * b.abs();
        assert!(
            (a - b).abs() < tol,
            "{name} pixel {i}: ours={a} ref={b} |Δ|={} > tol {tol}",
            (a - b).abs(),
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_reference() -> anyhow::Result<()> {
    #[cfg(target_family = "wasm")]
    {
        console_error_panic_hook::set_once();
        wasm_logger::init(wasm_logger::Config::new(log::Level::Trace));
    }
    #[cfg(not(target_family = "wasm"))]
    {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Error)
            .is_test(false)
            .try_init();
    }

    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);

    #[cfg(not(target_family = "wasm"))]
    let rec = tokio::task::spawn_blocking(|| {
        rerun::RecordingStreamBuilder::new("render test")
            .connect_grpc()
            .ok()
    })
    .await
    .unwrap();

    for (i, &(name, data)) in CASES.iter().enumerate() {
        log::info!("Checking {name}");

        let tensors = SafeTensors::deserialize(data)?;
        let splats: Splats = splats_from_safetensors(&tensors, &device)?;
        let img_ref = safetensor_to_burn::<3>(&tensors.tensor("out_img")?, &device);
        let [h, w, _] = img_ref.dims();

        let fov = std::f64::consts::PI * 0.5;
        let focal = fov_to_focal(fov, w as u32, &Pinhole);
        let cam = Camera::new(
            glam::vec3(0.123, 0.456, -8.0),
            glam::Quat::IDENTITY,
            focal_to_fov(focal, w as u32, &Pinhole),
            focal_to_fov(focal, h as u32, &Pinhole),
            glam::vec2(0.5, 0.5),
            Pinhole,
        );

        let (ours, _aux) = render_splats(
            splats,
            &cam,
            glam::uvec2(w as u32, h as u32),
            glam::Vec3::ZERO,
            None,
            TextureMode::Float,
            false,
        )
        .await;

        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = rec.as_ref() {
            use brush_rerun::burn_to_rerun::BurnToImage;
            rec.set_time_sequence("test case", i as i64);
            rec.log("img/render", &ours.clone().into_rerun_image_blocking())?;
            rec.log("img/ref", &img_ref.clone().into_rerun_image_blocking())?;
            rec.log(
                "img/dif",
                &(img_ref.clone() - ours.clone()).into_rerun_image_blocking(),
            )?;
        }
        #[cfg(target_family = "wasm")]
        let _ = i;

        assert_img_matches(name, ours, img_ref).await;
    }
    Ok(())
}
