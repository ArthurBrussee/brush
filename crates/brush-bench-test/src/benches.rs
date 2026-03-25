use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, MainBackend, TextureMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    render_splats,
};
use brush_render_bwd::render_splats as render_splats_diff;
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::{
    backend::{Autodiff, wgpu::WgpuDevice},
    module::AutodiffModule,
    tensor::{Tensor, TensorData, TensorPrimitive},
};
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};

type DiffBackend = Autodiff<MainBackend>;

const SEED: u64 = 42;
const RESOLUTIONS: [(u32, u32); 4] = [(1024, 1024), (1920, 1080), (2560, 1440), (3200, 1800)];
const SPLAT_COUNTS: [usize; 3] = [500_000, 1_000_000, 2_500_000];
const ITERS_PER_SYNC: u32 = 4;

fn gen_splats(device: &WgpuDevice, count: usize) -> Splats<DiffBackend> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

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
            [
                rng.random_range(0.1..0.9),
                rng.random_range(0.1..0.9),
                rng.random_range(0.1..0.9),
            ]
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

fn bench_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        50.0,
        50.0,
        glam::vec2(0.5, 0.5),
    )
}

/// Run forward rendering loop. Generates splats once, then renders `iters` times.
pub async fn run_forward_render(
    device: &WgpuDevice,
    splat_count: usize,
    resolution: (u32, u32),
    iters: u32,
) {
    let splats = gen_splats(device, splat_count).valid();
    let camera = bench_camera();
    for _ in 0..iters {
        let _ = render_splats(
            splats.clone(),
            &camera,
            glam::uvec2(resolution.0, resolution.1),
            Vec3::ZERO,
            None,
            TextureMode::Float,
        )
        .await;
    }
}

/// Run backward rendering loop. Generates splats once, then renders+backward `iters` times.
pub async fn run_backward_render(
    device: &WgpuDevice,
    splat_count: usize,
    resolution: (u32, u32),
    iters: u32,
) {
    let splats = gen_splats(device, splat_count);
    let camera = bench_camera();
    for _ in 0..iters {
        let diff_out = render_splats_diff(
            splats.clone(),
            &camera,
            glam::uvec2(resolution.0, resolution.1),
            Vec3::ZERO,
        )
        .await;
        let img: Tensor<DiffBackend, 3> =
            Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
        let _ = img.mean().backward();
    }
}

/// Run training loop. Generates splats once, then trains `iters` steps.
pub async fn run_training_steps(
    device: &WgpuDevice,
    splat_count: usize,
    resolution: (u32, u32),
    iters: u32,
) {
    let batch1 = generate_training_batch(resolution, Vec3::new(0.0, 0.0, 5.0));
    let batch2 = generate_training_batch(resolution, Vec3::new(2.0, 0.0, 5.0));
    let batches = [batch1, batch2];
    let config = TrainConfig::default();
    let mut splats = gen_splats(device, splat_count);
    let mut trainer = SplatTrainer::new(
        &config,
        device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );
    for step in 0..iters {
        let batch = batches[step as usize % batches.len()].clone();
        let (new_splats, _) = trainer.step(batch, splats).await;
        splats = new_splats;
    }
    assert!(splats.num_splats() > 0, "Failed smoke test");
}

#[cfg(not(target_family = "wasm"))]
#[divan::bench_group(max_time = 1)]
mod forward_rendering {
    use brush_render::MainBackend;
    use burn::{backend::wgpu::WgpuDevice, prelude::*};
    use burn_cubecl::cubecl::future::block_on;

    use crate::benches::{ITERS_PER_SYNC, RESOLUTIONS, SPLAT_COUNTS, run_forward_render};

    #[divan::bench(args = SPLAT_COUNTS)]
    fn render_1080p(bencher: divan::Bencher, splat_count: usize) {
        let device = WgpuDevice::default();
        bencher.bench_local(move || {
            block_on(async {
                run_forward_render(&device, splat_count, (1920, 1080), ITERS_PER_SYNC).await;
                MainBackend::sync(&device).expect("Failed to sync");
            });
        });
    }

    #[divan::bench(args = RESOLUTIONS)]
    fn render_2m_splats(bencher: divan::Bencher, (width, height): (u32, u32)) {
        let device = WgpuDevice::default();
        bencher.bench_local(move || {
            block_on(async {
                run_forward_render(&device, 2_000_000, (width, height), ITERS_PER_SYNC).await;
                MainBackend::sync(&device).expect("Failed to sync");
            });
        });
    }
}

#[cfg(not(target_family = "wasm"))]
#[divan::bench_group(max_time = 2)]
mod backward_rendering {
    use brush_render::MainBackend;
    use burn::{backend::wgpu::WgpuDevice, prelude::*};
    use burn_cubecl::cubecl::future::block_on;

    use crate::benches::{ITERS_PER_SYNC, RESOLUTIONS, run_backward_render};

    #[divan::bench(args = [1_000_000, 2_000_000, 5_000_000])]
    fn render_grad_1080p(bencher: divan::Bencher, splat_count: usize) {
        let device = WgpuDevice::default();
        bencher.bench_local(move || {
            block_on(async {
                run_backward_render(&device, splat_count, (1920, 1080), ITERS_PER_SYNC).await;
                MainBackend::sync(&device).expect("Failed to sync");
            });
        });
    }

    #[divan::bench(args = RESOLUTIONS)]
    fn render_grad_2m_splats(bencher: divan::Bencher, (width, height): (u32, u32)) {
        let device = WgpuDevice::default();
        bencher.bench_local(move || {
            block_on(async {
                run_backward_render(&device, 2_000_000, (width, height), ITERS_PER_SYNC).await;
                MainBackend::sync(&device).expect("Failed to sync");
            });
        });
    }
}

#[cfg(not(target_family = "wasm"))]
#[divan::bench_group(max_time = 4)]
mod training {
    use brush_render::MainBackend;
    use burn::{backend::wgpu::WgpuDevice, prelude::*};
    use burn_cubecl::cubecl::future::block_on;

    use crate::benches::{ITERS_PER_SYNC, SPLAT_COUNTS, run_training_steps};

    #[divan::bench(args = SPLAT_COUNTS)]
    fn train_steps(splat_count: usize) {
        let device = WgpuDevice::default();
        block_on(async {
            run_training_steps(&device, splat_count, (1920, 1080), ITERS_PER_SYNC).await;
            MainBackend::sync(&device).expect("Failed to sync");
        });
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::benches::{
        ITERS_PER_SYNC, run_backward_render, run_forward_render, run_training_steps,
    };
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_fwd_render() {
        let device = brush_kernel::test_helpers::test_device().await;
        run_forward_render(&device, 500_000, (1920, 1080), ITERS_PER_SYNC).await;
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_bwd_render() {
        let device = brush_kernel::test_helpers::test_device().await;
        run_backward_render(&device, 500_000, (1920, 1080), ITERS_PER_SYNC).await;
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_training() {
        let device = brush_kernel::test_helpers::test_device().await;
        run_training_steps(&device, 500_000, (1920, 1080), ITERS_PER_SYNC).await;
    }
}
