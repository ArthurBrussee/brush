use crate::{MainBackend, SplatForward, camera::Camera};
use assert_approx_eq::assert_approx_eq;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_wgpu::WgpuDevice;
use glam::Vec3;

#[test]
fn renders_at_all() {
    // Check if rendering doesn't hard crash or anything.
    // These are some zero-sized gaussians, so we know
    // what the result should look like.
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, 0.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(32, 32);
    let device = WgpuDevice::DefaultDevice;
    let num_points = 8;
    let means = Tensor::<MainBackend, 2>::zeros([num_points, 3], &device);
    let log_scales = Tensor::<MainBackend, 2>::ones([num_points, 3], &device) * 2.0;
    let quats: Tensor<MainBackend, 2> =
        Tensor::<MainBackend, 1>::from_floats(glam::Quat::IDENTITY.to_array(), &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, num_points);
    let sh_coeffs = Tensor::<MainBackend, 3>::ones([num_points, 1, 3], &device);
    let raw_opacity = Tensor::<MainBackend, 1>::zeros([num_points], &device);
    let (output, aux) = <MainBackend as SplatForward<MainBackend>>::render_splats(
        &cam,
        img_size,
        means.into_primitive().tensor(),
        log_scales.into_primitive().tensor(),
        quats.into_primitive().tensor(),
        sh_coeffs.into_primitive().tensor(),
        raw_opacity.into_primitive().tensor(),
        Vec3::ZERO,
        true,
    );
    aux.validate_values();

    let output: Tensor<MainBackend, 3> = Tensor::from_primitive(TensorPrimitive::Float(output));
    let rgb = output.clone().slice([0..32, 0..32, 0..3]);
    let alpha = output.slice([0..32, 0..32, 3..4]);
    let rgb_mean = rgb.mean().to_data().as_slice::<f32>().expect("Wrong type")[0];
    let alpha_mean = alpha
        .mean()
        .to_data()
        .as_slice::<f32>()
        .expect("Wrong type")[0];
    assert_approx_eq!(rgb_mean, 0.0, 1e-5);
    assert_approx_eq!(alpha_mean, 0.0);
}

#[test]
fn single_visible_splat() {
    // Test that a single visible splat in front of the camera is correctly counted
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, 0.0), // Camera at origin
        glam::Quat::IDENTITY,      // Looking down -Z axis
        0.5,                       // focal_x
        0.5,                       // focal_y
        glam::vec2(0.5, 0.5),      // pixel_center
    );
    let img_size = glam::uvec2(32, 32);
    let device = WgpuDevice::DefaultDevice;

    // Single gaussian in front of the camera
    // Camera space needs positive Z, and with IDENTITY rotation + zero position,
    // world_to_local is identity, so we need positive Z in world space too
    let means = Tensor::<MainBackend, 2>::from_floats(
        [[0.0, 0.0, 2.0]], // 2 units in front with positive Z
        &device,
    );

    // Larger scale to make sure it's visible
    let log_scales = Tensor::<MainBackend, 2>::from_floats(
        [[0.0, 0.0, 0.0]], // exp(0) = 1.0 scale - fairly large
        &device,
    );

    // Identity quaternion for rotation
    let quats = Tensor::<MainBackend, 2>::from_floats(
        [[1.0, 0.0, 0.0, 0.0]], // w, x, y, z
        &device,
    );

    // White color with single SH coefficient
    let sh_coeffs = Tensor::<MainBackend, 3>::from_floats(
        [[[1.0, 1.0, 1.0]]], // RGB
        &device,
    );

    // High opacity so it's definitely visible
    let raw_opacity = Tensor::<MainBackend, 1>::from_floats([2.0], &device); // sigmoid(2.0) ≈ 0.88

    let (_output, aux) = <MainBackend as SplatForward<MainBackend>>::render_splats(
        &cam,
        img_size,
        means.into_primitive().tensor(),
        log_scales.into_primitive().tensor(),
        quats.into_primitive().tensor(),
        sh_coeffs.into_primitive().tensor(),
        raw_opacity.into_primitive().tensor(),
        Vec3::ZERO,
        true,
    );

    // Check that exactly 1 splat is visible
    let num_visible = aux.num_visible();
    let num_visible_val = num_visible.to_data().as_slice::<i32>().expect("Wrong type")[0];
    assert_eq!(
        num_visible_val, 1,
        "Expected 1 visible splat, got {num_visible_val}",
    );
}

#[test]
fn zero_visible_splats_behind_camera() {
    // Test that a splat behind the camera is not counted as visible
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, 0.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(32, 32);
    let device = WgpuDevice::DefaultDevice;

    // Single gaussian behind the camera (negative Z in camera space)
    let means = Tensor::<MainBackend, 2>::from_floats(
        [[0.0, 0.0, -2.0]], // Behind camera (negative Z)
        &device,
    );

    let log_scales = Tensor::<MainBackend, 2>::from_floats([[-2.0, -2.0, -2.0]], &device);

    let quats = Tensor::<MainBackend, 2>::from_floats([[1.0, 0.0, 0.0, 0.0]], &device);

    let sh_coeffs = Tensor::<MainBackend, 3>::from_floats([[[1.0, 1.0, 1.0]]], &device);

    let raw_opacity = Tensor::<MainBackend, 1>::from_floats([2.0], &device);

    let (_output, aux) = <MainBackend as SplatForward<MainBackend>>::render_splats(
        &cam,
        img_size,
        means.into_primitive().tensor(),
        log_scales.into_primitive().tensor(),
        quats.into_primitive().tensor(),
        sh_coeffs.into_primitive().tensor(),
        raw_opacity.into_primitive().tensor(),
        Vec3::ZERO,
        true,
    );

    // Check that 0 splats are visible
    let num_visible = aux.num_visible();
    let num_visible_val = num_visible.to_data().as_slice::<i32>().expect("Wrong type")[0];
    assert_eq!(
        num_visible_val, 0,
        "Expected 0 visible splats behind camera, got {num_visible_val}",
    );
}
