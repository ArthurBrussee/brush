use crate::burn_glue::SplatForwardDiff;
use assert_approx_eq::assert_approx_eq;
use brush_render::{camera::Camera, gaussian_splats::SplatRenderMode};
use burn::{
    backend::Autodiff,
    tensor::{Distribution, Tensor, TensorPrimitive},
};
use burn_wgpu::{CubeBackend, WgpuDevice, WgpuRuntime};
use glam::Vec3;

type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TestBackend = Autodiff<InnerBackend>;

#[test]
fn diffs_at_all() {
    // Check if backward pass doesn't hard crash or anything.
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
    let means = Tensor::<TestBackend, 2>::zeros([num_points, 3], &device);
    let log_scales = Tensor::<TestBackend, 2>::ones([num_points, 3], &device) * 2.0;
    let quats: Tensor<TestBackend, 2> =
        Tensor::<TestBackend, 1>::from_floats(glam::Quat::IDENTITY.to_array(), &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, num_points);
    let sh_coeffs = Tensor::<TestBackend, 3>::ones([num_points, 1, 3], &device);
    let raw_opacity = Tensor::<TestBackend, 1>::zeros([num_points], &device);

    let result = <TestBackend as SplatForwardDiff<TestBackend>>::render_splats(
        &cam,
        img_size,
        means.into_primitive().tensor(),
        log_scales.into_primitive().tensor(),
        quats.into_primitive().tensor(),
        sh_coeffs.into_primitive().tensor(),
        raw_opacity.into_primitive().tensor(),
        SplatRenderMode::Default,
        Vec3::ZERO,
    );
    result.aux.validate_values();

    let output: Tensor<TestBackend, 3> = Tensor::from_primitive(TensorPrimitive::Float(result.img));
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
fn diffs_many_splats() {
    // Test backward pass with a ton of splats to verify 2D dispatch works correctly.
    // This exceeds the 1D 65535 * 256 = 16.7M limit.
    let num_points = 30_000_000;
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(64, 64);
    let device = WgpuDevice::DefaultDevice;

    // Create random gaussians spread in front of the camera
    let means = Tensor::<TestBackend, 2>::random(
        [num_points, 3],
        Distribution::Uniform(-2.0, 2.0),
        &device,
    );
    // Small scales so they don't cover everything
    let log_scales = Tensor::<TestBackend, 2>::random(
        [num_points, 3],
        Distribution::Uniform(-4.0, -2.0),
        &device,
    );
    // Random rotations (will be normalized)
    let quats = Tensor::<TestBackend, 2>::random(
        [num_points, 4],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    // Simple SH coefficients (just base color)
    let sh_coeffs = Tensor::<TestBackend, 3>::random(
        [num_points, 1, 3],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );
    // Some visible, some not
    let raw_opacity =
        Tensor::<TestBackend, 1>::random([num_points], Distribution::Uniform(-2.0, 2.0), &device);

    let result = <TestBackend as SplatForwardDiff<TestBackend>>::render_splats(
        &cam,
        img_size,
        means.into_primitive().tensor(),
        log_scales.into_primitive().tensor(),
        quats.into_primitive().tensor(),
        sh_coeffs.into_primitive().tensor(),
        raw_opacity.into_primitive().tensor(),
        SplatRenderMode::Default,
        Vec3::ZERO,
    );
    result.aux.validate_values();
}
