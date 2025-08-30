use crate::{config::TrainConfig, train::SplatTrainer};
use brush_dataset::scene::SceneBatch;
use brush_render::{MainBackend, camera::Camera, gaussian_splats::Splats};
use burn::{
    backend::{Autodiff, wgpu::WgpuDevice},
    tensor::{Tensor, TensorData},
};
use glam::{Quat, Vec3};

type DiffBackend = Autodiff<MainBackend>;

/// Create a minimal mock dataset for testing
fn create_mock_batch(
    device: &WgpuDevice,
    img_size: usize,
    camera_pos: Vec3,
) -> SceneBatch<DiffBackend> {
    // Create a simple test image with gradient pattern
    let img_data: Vec<f32> = (0..img_size * img_size * 3)
        .map(|i| {
            let pixel = i / 3;
            let x = pixel % img_size;
            let y = pixel / img_size;
            let channel = i % 3;

            match channel {
                0 => (x as f32 / img_size as f32) * 0.8 + 0.1,
                1 => (y as f32 / img_size as f32) * 0.8 + 0.1,
                2 => 0.5,
                _ => unreachable!(),
            }
        })
        .collect();

    let img_tensor = Tensor::<DiffBackend, 3>::from_data(
        TensorData::new(img_data, [img_size, img_size, 3]),
        device,
    );

    let camera = Camera::new(camera_pos, Quat::IDENTITY, 60.0, 60.0, glam::vec2(0.5, 0.5));

    SceneBatch {
        img_tensor,
        alpha_is_mask: false,
        camera,
    }
}

/// Create initial splats for testing
fn create_test_splats(device: &WgpuDevice, count: usize) -> Splats<DiffBackend> {
    let means: Vec<f32> = (0..count)
        .flat_map(|i| {
            let x = (i % 4) as f32 - 1.5;
            let y = (i / 4) as f32 - 1.5;
            [x * 0.3, y * 0.3, 0.0]
        })
        .collect();

    let rotations: Vec<f32> = (0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let log_scales: Vec<f32> = (0..count).flat_map(|_| [-2.0, -2.0, -2.0]).collect();
    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|i| {
            let val = 0.4 + (i as f32 / count as f32) * 0.4;
            [val, val * 0.9, val * 0.7]
        })
        .collect();
    let opacities: Vec<f32> = (0..count).map(|_| 0.7).collect();
    Splats::<DiffBackend>::from_raw(
        means,
        Some(rotations),
        Some(log_scales),
        Some(sh_coeffs),
        Some(opacities),
        device,
    )
    .with_sh_degree(0)
}

#[test]
fn test_training_workflow() {
    // Integration test for complete training workflow
    let device = WgpuDevice::default();

    // Create test data
    let batch1 = create_mock_batch(&device, 32, Vec3::new(0.0, 0.0, -2.0));
    let batch2 = create_mock_batch(&device, 32, Vec3::new(0.5, 0.0, -2.0));
    let batches = vec![batch1, batch2];

    let mut splats = create_test_splats(&device, 128);

    // Create minimal training config
    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(&config, &device);
    let scene_extent = 1.0;
    let mut losses = Vec::new();
    let initial_count = splats.num_splats();

    // Run training steps
    for step in 0..50 {
        let batch = &batches[step as usize % batches.len()];
        let (new_splats, stats) = trainer.step(scene_extent, step, batch, splats);
        splats = new_splats;
        let loss = stats.loss.into_scalar();
        losses.push(loss);
        // Verify loss is finite
        assert!(loss.is_finite(), "Loss should be finite at step {step}");
        assert!(loss >= 0.0, "Loss should be non-negative");
    }

    let initial_loss = losses[0];
    let final_loss = *losses.last().unwrap();

    // Basic sanity checks
    assert!(splats.num_splats() > 0, "Should still have splats.");
    assert!(final_loss < initial_loss, "Loss should go down.");

    println!("Training test completed:");
    println!("  Initial loss: {initial_loss:.6}");
    println!("  Final loss: {final_loss:.6}");
    println!("  Initial splats: {initial_count}");
    println!("  Final splats: {}", splats.num_splats());
}

#[test]
fn test_rendering_pipeline_integration() {
    // Test the rendering pipeline with realistic data using MainBackend
    let device = WgpuDevice::default();

    // Create splats with MainBackend instead of DiffBackend
    let splats = {
        let means: Vec<f32> = (0..16)
            .flat_map(|i| {
                let x = (i % 4) as f32 - 1.5;
                let y = (i / 4) as f32 - 1.5;
                [x * 0.3, y * 0.3, 0.0]
            })
            .collect();
        let rotations: Vec<f32> = (0..16).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
        let log_scales: Vec<f32> = (0..16).flat_map(|_| [-2.0, -2.0, -2.0]).collect();
        let sh_coeffs: Vec<f32> = (0..16)
            .flat_map(|i| {
                let val = 0.4 + (i as f32 / 16.0) * 0.4;
                [val, val * 0.9, val * 0.7]
            })
            .collect();
        let opacities: Vec<f32> = (0..16).map(|_| 0.7).collect();
        Splats::<MainBackend>::from_raw(
            means,
            Some(rotations),
            Some(log_scales),
            Some(sh_coeffs),
            Some(opacities),
            &device,
        )
        .with_sh_degree(0)
    };

    let camera = Camera::new(
        Vec3::new(0.0, 0.0, -1.5),
        Quat::IDENTITY,
        50.0,
        50.0,
        glam::vec2(0.5, 0.5),
    );

    // Test rendering through the splats render method
    let resolutions = [(64, 64), (128, 128)];

    for (width, height) in resolutions {
        let (rendered_img, _) =
            splats.render(&camera, glam::uvec2(width, height), Vec3::ZERO, None);
        // Debug: check what dimensions we actually get
        let dims = rendered_img.dims();
        // Basic validation - just check we got something reasonable
        assert!(dims.len() >= 2, "Should have at least 2 dimensions");
        assert_eq!(dims[0], height as usize, "Height should match");
        assert_eq!(dims[1], width as usize, "Width should match");
        // Validate pixel values are in reasonable range
        let img_data = rendered_img.into_data().into_vec::<f32>().unwrap();
        let mut finite_pixels = 0;
        for &pixel in &img_data {
            if pixel.is_finite() {
                finite_pixels += 1;
            }
        }
        // All pixels should at least be finite
        assert_eq!(finite_pixels, img_data.len(), "All pixels should be finite");
    }
}
