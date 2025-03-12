#![allow(clippy::single_range_in_vec_init)]
#![recursion_limit = "256"]

mod safetensor_utils;

use std::collections::HashMap;
use std::path::Path;
use std::{fs::File, io::Read};

use brush_render::{
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use brush_train::burn_glue::SplatForwardDiff;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::tensor::{Tensor, TensorPrimitive};
use safetensor_utils::splats_from_safetensors;
use safetensors::SafeTensors;

fn main() {
    divan::main();
}

type DiffBack = Autodiff<Wgpu>;

const BENCH_DENSITIES: [f32; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const DENSE_MULT: f32 = 0.25;

const LOW_RES: glam::UVec2 = glam::uvec2(512, 512);
const HIGH_RES: glam::UVec2 = glam::uvec2(1024, 1024);

const TARGET_SAMPLE_COUNT: u32 = 50;
const INTERNAL_ITERS: u32 = 5;

fn generate_bench_data() -> anyhow::Result<()> {
    <DiffBack as burn::prelude::Backend>::seed(4);
    let num_points = 2usize.pow(21); //  Maximum number of splats to bench.

    let device = WgpuDevice::DefaultDevice;

    let means = Tensor::<DiffBack, 2>::random(
        [num_points, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    ) * 10000.0;
    let log_scales = Tensor::<DiffBack, 2>::random(
        [num_points, 3],
        burn::tensor::Distribution::Uniform(0.05, 15.0),
        &device,
    )
    .log();
    let coeffs = Tensor::<DiffBack, 3>::random(
        [num_points, 1, 3],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let u = Tensor::<DiffBack, 2>::random(
        [num_points, 1],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let v = Tensor::<DiffBack, 2>::random(
        [num_points, 1],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let w = Tensor::<DiffBack, 2>::random(
        [num_points, 1],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let v = v * 2.0 * std::f32::consts::PI;
    let w = w * 2.0 * std::f32::consts::PI;

    let quats = Tensor::cat(
        vec![
            Tensor::sqrt(-u.clone() + 1.0) * Tensor::sin(v.clone()),
            Tensor::sqrt(-u.clone() + 1.0) * Tensor::cos(v),
            Tensor::sqrt(u.clone()) * Tensor::sin(w.clone()),
            Tensor::sqrt(u) * Tensor::cos(w),
        ],
        1,
    );
    let opacities = Tensor::<DiffBack, 1>::random(
        [num_points],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let bytes = means.to_data().as_bytes().to_vec();
    let means =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, means.shape().dims, &bytes)?;
    let bytes = log_scales.to_data().as_bytes().to_vec();
    let log_scales = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        log_scales.shape().dims,
        &bytes,
    )?;
    let bytes = quats.to_data().as_bytes().to_vec();
    let quats =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, quats.shape().dims, &bytes)?;
    let bytes = coeffs.to_data().as_bytes().to_vec();
    let coeffs =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, coeffs.shape().dims, &bytes)?;
    let bytes = opacities.to_data().as_bytes().to_vec();
    let opacities = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        opacities.shape().dims,
        &bytes,
    )?;

    let tensors = HashMap::from([
        ("means", means),
        ("scales", log_scales),
        ("coeffs", coeffs),
        ("quats", quats),
        ("opacities", opacities),
    ]);

    safetensors::serialize_to_file(
        &tensors,
        &None,
        Path::new("./test_cases/bench_data.safetensors"),
    )?;
    Ok(())
}

fn bench_general(
    bencher: divan::Bencher,
    dens: f32,
    mean_mult: f32,
    resolution: glam::UVec2,
    grad: bool,
) {
    if !Path::new("./test_cases/bench_data.safetensors").exists() {
        generate_bench_data().expect("Failed to generate bench data");
    }

    let device = WgpuDevice::DefaultDevice;
    let mut buffer = Vec::new();
    let _ = File::open("./test_cases/bench_data.safetensors")
        .expect("Failed to open bench data")
        .read_to_end(&mut buffer)
        .expect("Failed to read bench data");
    let tensors = SafeTensors::deserialize(&buffer).expect("Failed to deserialize bench data");
    let splats: Splats<DiffBack> =
        splats_from_safetensors(&tensors, &device).expect("Failed to load bench data");
    let num_points = (splats.num_splats() as f32 * dens) as usize;
    let splats = Splats::from_tensor_data(
        (splats.means.val() * mean_mult).slice([0..num_points]),
        splats.rotation.val().slice([0..num_points]),
        splats.log_scales.val().slice([0..num_points]),
        splats.sh_coeffs.val().slice([0..num_points]),
        splats.raw_opacity.val().slice([0..num_points]),
    );
    let [w, h] = resolution.into();
    let fov = std::f64::consts::PI * 0.5;
    let focal = fov_to_focal(fov, w);
    let fov_x = focal_to_fov(focal, w);
    let fov_y = focal_to_fov(focal, h);
    let camera = Camera::new(
        glam::vec3(0.0, 0.0, -8.0),
        glam::Quat::IDENTITY,
        fov_x,
        fov_y,
        glam::vec2(0.5, 0.5),
    );

    if grad {
        bencher.bench_local(move || {
            for _ in 0..INTERNAL_ITERS {
                let diff_out = DiffBack::render_splats(
                    &camera,
                    resolution,
                    splats.means.val().into_primitive().tensor(),
                    splats.log_scales.val().into_primitive().tensor(),
                    splats.rotation.val().into_primitive().tensor(),
                    splats.sh_coeffs.val().into_primitive().tensor(),
                    splats.opacities().into_primitive().tensor(),
                );
                let img: Tensor<DiffBack, 3> =
                    Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
                let _ = img.mean().backward();
            }
            // Wait for GPU work.
            <Wgpu as burn::prelude::Backend>::sync(&device);
        });
    } else {
        // Run with no autodiff graph.
        let splats = splats.valid();

        bencher.bench_local(move || {
            for _ in 0..INTERNAL_ITERS {
                let _ = splats.render(&camera, resolution, true);
            }
            // Wait for GPU work.
            <Wgpu as burn::prelude::Backend>::sync(&device);
        });
    }
}

#[divan::bench_group(max_time = 1000, sample_count = TARGET_SAMPLE_COUNT, sample_size = 1)]
mod fwd {
    use crate::{BENCH_DENSITIES, DENSE_MULT, HIGH_RES, LOW_RES, bench_general};

    #[divan::bench(args = BENCH_DENSITIES)]
    fn base(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, LOW_RES, false);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn dense(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, DENSE_MULT, LOW_RES, false);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn hd(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, HIGH_RES, false);
    }
}

#[divan::bench_group(max_time = 20, sample_count = TARGET_SAMPLE_COUNT, sample_size = 1)]
mod bwd {
    use crate::{BENCH_DENSITIES, DENSE_MULT, HIGH_RES, LOW_RES, bench_general};

    #[divan::bench(args = BENCH_DENSITIES)]
    fn base(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, LOW_RES, true);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn dense(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, DENSE_MULT, LOW_RES, true);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn hd(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, HIGH_RES, true);
    }
}
