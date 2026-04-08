use brush_render::{
    MainBackend,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use burn::{
    Tensor,
    backend::Autodiff,
    prelude::Backend,
    tensor::{TensorPrimitive, s},
};

use anyhow::{Context, Result};
use glam::Vec3;
use safetensors::SafeTensors;
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use crate::safetensor_utils::{safetensor_to_burn, splats_from_safetensors};

type DiffBack = Autodiff<MainBackend>;

static CRAB_PNG: &[u8] = include_bytes!("../test_cases/crab.png");
static TINY_CASE: &[u8] = include_bytes!("../test_cases/tiny_case.safetensors");
static BASIC_CASE: &[u8] = include_bytes!("../test_cases/basic_case.safetensors");
#[allow(clippy::large_include_file)] // It's fine, just for a test, not the final binary.
static MIX_CASE: &[u8] = include_bytes!("../test_cases/mix_case.safetensors");

async fn compare<B: Backend, const D1: usize>(
    name: &str,
    tensor_a: Tensor<B, D1>,
    tensor_b: Tensor<B, D1>,
    atol: f32,
    rtol: f32,
) {
    assert!(
        tensor_a.dims() == tensor_b.dims(),
        "Tensor shapes for {name} must match"
    );

    let data_a = tensor_a
        .into_data_async()
        .await
        .unwrap_or_else(|_| panic!("Failed to convert tensor {name}:a"))
        .into_vec::<f32>()
        .unwrap_or_else(|_| panic!("Failed to convert tensor {name}:a"));
    let data_b = tensor_b
        .into_data_async()
        .await
        .unwrap_or_else(|_| panic!("Failed to convert tensor {name}:b"))
        .into_vec::<f32>()
        .unwrap_or_else(|_| panic!("Failed to convert tensor {name}:b"));

    for (i, (a, b)) in data_a.iter().zip(&data_b).enumerate() {
        let tol = atol + rtol * b.abs();

        assert!(
            !a.is_nan() && !b.is_nan(),
            "{name}: Found Nan values at position {i}: {a} vs {b}"
        );

        assert!(
            (a - b).abs() < tol,
            "{name} mismatch: {a} vs {b} at absolution position idx {i}, Difference is {} > {}",
            a - b,
            tol
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_reference() -> Result<()> {
    let device = brush_kernel::test_helpers::test_device().await;

    let crab_img = image::load_from_memory(CRAB_PNG)?;

    let raw_buffer = crab_img.to_rgb8().into_raw();
    let crab_tens: Tensor<DiffBack, 3> = Tensor::<_, 1>::from_floats(
        raw_buffer
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    )
    .reshape([crab_img.height() as usize, crab_img.width() as usize, 3]);

    // Concat alpha to tensor.
    let crab_tens = Tensor::cat(
        vec![
            crab_tens,
            Tensor::zeros(
                [crab_img.height() as usize, crab_img.width() as usize, 1],
                &device,
            ),
        ],
        2,
    );

    #[cfg(not(target_family = "wasm"))]
    let rec = tokio::task::spawn_blocking(|| {
        rerun::RecordingStreamBuilder::new("render test")
            .connect_grpc()
            .ok()
    })
    .await
    .unwrap();

    let test_cases: &[(&str, &[u8])] = &[
        ("tiny_case", TINY_CASE),
        ("basic_case", BASIC_CASE),
        ("mix_case", MIX_CASE),
    ];

    for (i, &(path, data)) in test_cases.iter().enumerate() {
        log::info!("Checking path {path}");

        let tensors = SafeTensors::deserialize(data)?;
        let splats: Splats<DiffBack> = splats_from_safetensors(&tensors, &device)?;

        let img_ref = safetensor_to_burn::<DiffBack, 3>(&tensors.tensor("out_img")?, &device);
        let [h, w, _] = img_ref.dims();

        let fov = std::f64::consts::PI * 0.5;

        let focal = fov_to_focal(fov, w as u32);
        let fov_x = focal_to_fov(focal, w as u32);
        let fov_y = focal_to_fov(focal, h as u32);

        let cam = Camera::new(
            glam::vec3(0.123, 0.456, -8.0),
            glam::Quat::IDENTITY,
            fov_x,
            fov_y,
            glam::vec2(0.5, 0.5),
        );

        let diff_out = brush_render_bwd::render_splats(
            splats.clone(),
            &cam,
            glam::uvec2(w as u32, h as u32),
            Vec3::ZERO,
        )
        .await;

        let out: Tensor<DiffBack, 3> = Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));

        #[cfg(not(target_family = "wasm"))]
        if let Some(rec) = rec.as_ref() {
            use brush_rerun::burn_to_rerun::{BurnToImage, BurnToRerun};
            let render_aux = &diff_out.render_aux;
            rec.set_time_sequence("test case", i as i64);
            rec.log("img/render", &out.clone().into_rerun_image_blocking())?;
            rec.log("img/ref", &img_ref.clone().into_rerun_image_blocking())?;
            rec.log(
                "img/dif",
                &(img_ref.clone() - out.clone()).into_rerun_image_blocking(),
            )?;
            rec.log(
                "images/tile_depth",
                &render_aux.calc_tile_depth().into_rerun_blocking(),
            )?;
        }
        let _ = (i, &diff_out.render_aux); // suppress unused warnings when rerun is off

        // Tolerance reflects f16 precision in the color pipeline: ProjectedSplat stores
        // colors as f16, giving ~1e-3 relative precision that accumulates through blending.
        compare("img", out.clone(), img_ref, 1e-5, 1e-2).await;

        let grads = (out.clone() - crab_tens.clone())
            .powi_scalar(2.0)
            .mean()
            .backward();

        let v_coeffs_ref =
            safetensor_to_burn::<DiffBack, 3>(&tensors.tensor("v_coeffs")?, &device).inner();
        // Compare DC gradient (first coeff) against reference slice.
        let v_dc = splats.sh_coeffs_dc.grad(&grads).context("dc grad")?;
        let [n, _, _] = v_dc.dims();
        let v_dc_ref = v_coeffs_ref.clone().slice(s![0..n, 0..1]);
        compare("v_coeffs_dc", v_dc, v_dc_ref, 1e-5, 1e-7).await;
        // Compare rest gradient if present.
        let [_, total_c, _] = v_coeffs_ref.dims();
        if total_c > 1
            && let Some(v_rest) = splats.sh_coeffs_rest.grad(&grads)
        {
            let v_rest_ref = v_coeffs_ref.slice(s![0..n, 1..total_c]);
            compare("v_coeffs_rest", v_rest, v_rest_ref, 1e-5, 1e-7).await;
        }

        let v_transforms = splats.transforms.grad(&grads).context("transforms grad")?;
        // Slice transforms gradient: means(0..3), quats(3..7), log_scales(7..10)
        let v_means = v_transforms.clone().slice(s![.., 0..3]);
        let v_means_ref =
            safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("v_means")?, &device).inner();
        compare("v_means", v_means, v_means_ref, 1e-5, 1e-7).await;

        let v_quats = v_transforms.clone().slice(s![.., 3..7]);
        let v_quats_ref =
            safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("v_quats")?, &device).inner();
        compare("v_quats", v_quats, v_quats_ref, 1e-5, 1e-7).await;

        let v_scales = v_transforms.slice(s![.., 7..10]);
        let v_scales_ref =
            safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("v_scales")?, &device).inner();
        compare("v_scales", v_scales, v_scales_ref, 1e-5, 1e-7).await;

        let v_opacities_ref =
            safetensor_to_burn::<DiffBack, 1>(&tensors.tensor("v_opacities")?, &device).inner();
        let v_opacities = splats
            .raw_opacities
            .grad(&grads)
            .context("opacities grad")?;
        compare("v_opacities", v_opacities, v_opacities_ref, 1e-5, 1e-7).await;
    }
    Ok(())
}
