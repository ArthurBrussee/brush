#![recursion_limit = "256"]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, MainBackend,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    render_splats,
};
use brush_render_bwd::render_splats as render_splats_diff;
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::{
    backend::{Autodiff, wgpu::WgpuDevice},
    module::AutodiffModule,
    prelude::Backend,
    tensor::{Tensor, TensorData, TensorPrimitive},
};
use glam::{Quat, Vec3};
use rand::{Rng, SeedableRng};

fn main() {
    divan::main();
}

type DiffBackend = Autodiff<MainBackend>;

const SEED: u64 = 42;
const RESOLUTIONS: [(u32, u32); 4] = [(1024, 1024), (1920, 1080), (2560, 1440), (3200, 1800)];
const SPLAT_COUNTS: [usize; 3] = [500_000, 1_000_000, 2_500_000];
const ITERS_PER_SYNC: u32 = 10;

fn gen_splats(device: &WgpuDevice, count: usize) -> Splats<DiffBackend> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    // Generate positions in a realistic scene distribution (roughly spherical with some structure)
    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            // Create clusters with some randomness
            let cluster_center = [
                rng.random_range(-5.0..5.0),
                rng.random_range(-3.0..3.0),
                rng.random_range(-10.0..10.0),
            ];
            let offset = [
                rng.random::<f32>() - 0.5,
                rng.random::<f32>() - 0.5,
                rng.random::<f32>() - 0.5,
            ];
            [
                cluster_center[0] + offset[0] * 2.0,
                cluster_center[1] + offset[1] * 2.0,
                cluster_center[2] + offset[2] * 3.0,
            ]
        })
        .collect();

    // Realistic scale distribution (log-normal-ish)
    let log_scales: Vec<f32> = (0..count)
        .flat_map(|_| {
            let base_scale = rng.random_range(0.01..0.1_f32).ln();
            let variation = rng.random_range(0.8..1.2);
            [base_scale, base_scale * variation, base_scale * variation]
        })
        .collect();

    // Random rotations using proper quaternion generation
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

    // Realistic color distribution
    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|_| {
            let r = rng.random_range(0.1..0.9);
            let g = rng.random_range(0.1..0.9);
            let b = rng.random_range(0.1..0.9);
            [r, g, b]
        })
        .collect();

    // Realistic opacity distribution (mostly opaque with some variation)
    let opacities: Vec<f32> = (0..count).map(|_| rng.random_range(0.05..1.0)).collect();

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

fn generate_training_batch(resolution: (u32, u32), camera_pos: Vec3) -> SceneBatch {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED + camera_pos.x as u64);

    let (width, height) = resolution;
    let pixel_count = (width * height * 3) as usize;

    let img_data: Vec<f32> = (0..pixel_count)
        .map(|i| {
            let pixel_idx = i / 3;
            let x = (pixel_idx as u32) % width;
            let y = (pixel_idx as u32) / width;
            let channel = i % 3;
            // Create some structure in the image
            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;

            let base = match channel {
                0 => nx * 0.6 + 0.2,
                1 => ny * 0.6 + 0.2,
                2 => (nx + ny) * 0.3 + 0.4,
                _ => unreachable!(),
            };
            // Add some noise
            base + (rng.random::<f32>() - 0.5) * 0.1
        })
        .collect();

    let img_tensor = TensorData::new(img_data, [height as usize, width as usize, 3]);
    let camera = Camera::new(camera_pos, Quat::IDENTITY, 50.0, 50.0, glam::vec2(0.5, 0.5));

    SceneBatch {
        img_tensor,
        alpha_mode: AlphaMode::Transparent,
        camera,
    }
}

#[divan::bench_group(max_time = 1)]
mod forward_rendering {
    use crate::{
        AutodiffModule, Backend, Camera, ITERS_PER_SYNC, MainBackend, Quat, RESOLUTIONS,
        SPLAT_COUNTS, Vec3, WgpuDevice, gen_splats, render_splats,
    };

    #[divan::bench(args = SPLAT_COUNTS)]
    fn render_1080p(bencher: divan::Bencher, splat_count: usize) {
        let device = WgpuDevice::default();
        let splats = gen_splats(&device, splat_count).valid();
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            50.0,
            50.0,
            glam::vec2(0.5, 0.5),
        );

        bencher.bench_local(move || {
            for _ in 0..ITERS_PER_SYNC {
                let _ = render_splats(&splats, &camera, glam::uvec2(1920, 1080), Vec3::ZERO, None);
            }
            MainBackend::sync(&device).expect("Failed to sync");
        });
    }

    #[divan::bench(args = RESOLUTIONS)]
    fn render_2m_splats(bencher: divan::Bencher, (width, height): (u32, u32)) {
        let device = WgpuDevice::default();
        let splats = gen_splats(&device, 2_000_000).valid();
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            50.0,
            50.0,
            glam::vec2(0.5, 0.5),
        );

        bencher.bench_local(move || {
            for _ in 0..ITERS_PER_SYNC {
                let _ = render_splats(
                    &splats,
                    &camera,
                    glam::uvec2(width, height),
                    Vec3::ZERO,
                    None,
                );
            }
            MainBackend::sync(&device).expect("Failed to sync");
        });
    }
}

#[divan::bench_group(max_time = 2)]
mod backward_rendering {
    use crate::{
        Backend, Camera, DiffBackend, ITERS_PER_SYNC, MainBackend, Quat, RESOLUTIONS, Tensor,
        TensorPrimitive, Vec3, WgpuDevice, gen_splats, render_splats_diff,
    };

    #[divan::bench(args = [1_000_000, 2_000_000, 5_000_000])]
    fn render_grad_1080p(bencher: divan::Bencher, splat_count: usize) {
        let device = WgpuDevice::default();
        let splats = gen_splats(&device, splat_count);
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            50.0,
            50.0,
            glam::vec2(0.5, 0.5),
        );

        bencher.bench_local(move || {
            for _ in 0..ITERS_PER_SYNC {
                let diff_out =
                    render_splats_diff(&splats, &camera, glam::uvec2(1920, 1080), Vec3::ZERO);
                let img: Tensor<DiffBackend, 3> =
                    Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
                let _ = img.mean().backward();
            }
            MainBackend::sync(&device).expect("Failed to sync");
        });
    }

    #[divan::bench(args = RESOLUTIONS)]
    fn render_grad_2m_splats(bencher: divan::Bencher, (width, height): (u32, u32)) {
        let device = WgpuDevice::default();
        let splats = gen_splats(&device, 2_000_000);
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            50.0,
            50.0,
            glam::vec2(0.5, 0.5),
        );
        bencher.bench_local(move || {
            for _ in 0..ITERS_PER_SYNC {
                let diff_out =
                    render_splats_diff(&splats, &camera, glam::uvec2(width, height), Vec3::ZERO);
                let img: Tensor<DiffBackend, 3> =
                    Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
                let _ = img.mean().backward();
            }
            MainBackend::sync(&device).expect("Failed to sync");
        });
    }
}

#[divan::bench_group(max_time = 4)]
mod training {
    use crate::{
        Backend, MainBackend, SPLAT_COUNTS, SplatTrainer, TrainConfig, Vec3, WgpuDevice,
        gen_splats, generate_training_batch,
    };

    #[divan::bench(args = SPLAT_COUNTS)]
    fn train_steps(splat_count: usize) {
        burn_cubecl::cubecl::future::block_on(async {
            let device = WgpuDevice::default();
            let batch1 = generate_training_batch((1920, 1080), Vec3::new(0.0, 0.0, 5.0));
            let batch2 = generate_training_batch((1920, 1080), Vec3::new(2.0, 0.0, 5.0));
            let batches = [batch1, batch2];
            let config = TrainConfig::default();
            let mut splats = gen_splats(&device, splat_count);
            let mut trainer = SplatTrainer::new(&config, &device, splats.clone()).await;

            for step in 0..20 {
                let batch = batches[step % batches.len()].clone();
                let (new_splats, _) = trainer.step(batch, splats);
                splats = new_splats;
            }
            MainBackend::sync(&device).expect("Failed to sync");
        });
    }
}

// Integration tests using the same realistic data generation
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_forward_rendering_works() {
        let device = WgpuDevice::default();
        let splats = generate_realistic_splats(&device, 10_000).valid();
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            50.0,
            50.0,
            glam::vec2(0.5, 0.5),
        );

        let (img, _) = render_splats(&splats, &camera, glam::uvec2(256, 256), Vec3::ZERO, None);
        let dims = img.dims();
        assert_eq!(dims, [256, 256, 3]);

        let img_data = img.into_data().into_vec::<f32>().unwrap();
        assert!(img_data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_training_step_completes() {
        let device = WgpuDevice::default();
        let batch = generate_training_batch(&device, (128, 128), Vec3::new(0.0, 0.0, 5.0));
        let splats = generate_realistic_splats(&device, 10_000);
        let config = TrainConfig::default();
        let mut trainer = SplatTrainer::new(&config, &device);

        let (final_splats, stats) = trainer.step(10.0, 0, &batch, splats);

        assert!(final_splats.num_splats() > 0);
        let loss = stats.loss.into_scalar();
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_backward_rendering_works() {
        let device = WgpuDevice::default();
        let splats = generate_realistic_splats(&device, 1000);
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            50.0,
            50.0,
            glam::vec2(0.5, 0.5),
        );

        let diff_out = DiffBackend::render_splats(
            &camera,
            glam::uvec2(64, 64),
            splats.means.val().into_primitive().tensor(),
            splats.log_scales.val().into_primitive().tensor(),
            splats.rotation.val().into_primitive().tensor(),
            splats.sh_coeffs.val().into_primitive().tensor(),
            splats.raw_opacity.val().into_primitive().tensor(),
            Vec3::ZERO,
        );

        let img: Tensor<DiffBackend, 3> =
            Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
        let loss = img.mean();
        let grads = loss.backward();
        // Just verify gradients were computed
        assert!(grads.wrt(&splats.means.val()).is_some());
    }
}
