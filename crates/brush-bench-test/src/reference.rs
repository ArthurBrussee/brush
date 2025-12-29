use anyhow::{Context, Result};
use brush_render::{
    MainBackend,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use brush_render_bwd::render_splats;
use brush_rerun::burn_to_rerun::{BurnToImage, BurnToRerun};
use burn::{
    backend::{Autodiff, wgpu::WgpuDevice},
    prelude::Backend,
    tensor::{Float, Int, Tensor, TensorPrimitive},
};
use glam::Vec3;
use safetensors::SafeTensors;
use std::{fs::File, io::Read};

use crate::safetensor_utils::{safetensor_to_burn, splats_from_safetensors};

type DiffBack = Autodiff<MainBackend>;

const USE_RERUN: bool = true;

fn compare<B: Backend, const D1: usize>(
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
        .into_data()
        .into_vec::<f32>()
        .unwrap_or_else(|_| panic!("Failed to convert tensor {name}:a"));
    let data_b = tensor_b
        .into_data()
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

#[tokio::test]
async fn test_reference() -> Result<()> {
    let device = WgpuDevice::DefaultDevice;

    let crab_img = image::open("./test_cases/crab.png")?;

    // Convert the image to RGB format
    // Get the raw buffer
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

    let rec = if USE_RERUN {
        rerun::RecordingStreamBuilder::new("render test")
            .connect_grpc()
            .ok()
    } else {
        None
    };

    for (i, path) in ["tiny_case", "basic_case", "mix_case"].iter().enumerate() {
        log::info!("Checking path {path}");

        let mut buffer = Vec::new();
        let _ = File::open(format!("./test_cases/{path}.safetensors"))?.read_to_end(&mut buffer)?;

        let tensors = SafeTensors::deserialize(&buffer)?;
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

        let diff_out = render_splats(&splats, &cam, glam::uvec2(w as u32, h as u32), Vec3::ZERO);

        let (out, aux) = (
            Tensor::from_primitive(TensorPrimitive::Float(diff_out.img)),
            diff_out.aux,
        );

        if let Some(rec) = rec.as_ref() {
            rec.set_time_sequence("test case", i as i64);
            rec.log("img/render", &out.clone().into_rerun_image().await)?;
            rec.log("img/ref", &img_ref.clone().into_rerun_image().await)?;
            rec.log(
                "img/dif",
                &(img_ref.clone() - out.clone()).into_rerun_image().await,
            )?;
            rec.log(
                "images/tile_depth",
                &aux.calc_tile_depth().into_rerun().await,
            )?;
        }

        splats.validate_values();
        aux.validate_values();

        let num_visible: Tensor<DiffBack, 1, Int> = aux.num_visible();
        let num_visible = num_visible.into_scalar_async().await.unwrap() as usize;
        let global_from_compact_gid: Tensor<DiffBack, 1, Int> =
            Tensor::from_primitive(aux.global_from_compact_gid.clone());
        let gs_ids = global_from_compact_gid.clone().slice([0..num_visible]);
        let projected_splats =
            Tensor::from_primitive(TensorPrimitive::Float(aux.projected_splats.clone()));
        let xys: Tensor<DiffBack, 2, Float> =
            projected_splats.clone().slice([0..num_visible, 0..2]);
        let xys_ref = safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("xys")?, &device);
        let xys_ref = xys_ref.select(0, gs_ids.clone());
        compare("xy", xys, xys_ref, 1e-5, 2e-5);
        let conics: Tensor<DiffBack, 2, Float> =
            projected_splats.clone().slice([0..num_visible, 2..5]);
        let conics_ref = safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("conics")?, &device);
        let conics_ref = conics_ref.select(0, gs_ids.clone());
        compare("conics", conics, conics_ref, 1e-6, 2e-5);

        // Check if images match.
        compare("img", out.clone(), img_ref, 1e-5, 1e-5);

        let grads = (out.clone() - crab_tens.clone())
            .powi_scalar(2.0)
            .mean()
            .backward();

        let v_coeffs_ref =
            safetensor_to_burn::<DiffBack, 3>(&tensors.tensor("v_coeffs")?, &device).inner();
        let v_coeffs = splats.sh_coeffs.grad(&grads).context("coeffs grad")?;
        compare("v_coeffs", v_coeffs, v_coeffs_ref, 1e-5, 1e-7);

        let v_means_ref =
            safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("v_means")?, &device).inner();
        let v_means = splats.means.grad(&grads).context("means grad")?;
        compare("v_means", v_means, v_means_ref, 1e-5, 1e-7);

        let v_quats = splats.rotations.grad(&grads).context("quats grad")?;
        let v_quats_ref =
            safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("v_quats")?, &device).inner();
        compare("v_quats", v_quats, v_quats_ref, 1e-5, 1e-7);

        let v_scales = splats.log_scales.grad(&grads).context("scales grad")?;
        let v_scales_ref =
            safetensor_to_burn::<DiffBack, 2>(&tensors.tensor("v_scales")?, &device).inner();
        compare("v_scales", v_scales, v_scales_ref, 1e-5, 1e-7);

        let v_opacities_ref =
            safetensor_to_burn::<DiffBack, 1>(&tensors.tensor("v_opacities")?, &device).inner();
        let v_opacities = splats
            .raw_opacities
            .grad(&grads)
            .context("opacities grad")?;
        compare("v_opacities", v_opacities, v_opacities_ref, 1e-5, 1e-7);
    }
    Ok(())
}
