use crate::{
    MainBackend, TextureMode,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats, render_splats},
};
use assert_approx_eq::assert_approx_eq;
use burn::tensor::{Distribution, Tensor};
use burn_wgpu::WgpuDevice;
use glam::Vec3;
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn renders_at_all() {
    // Splats sit at the camera origin so they're culled by the near plane.
    // With a black background that means every pixel must read back as zero.
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, 0.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(32, 32);
    let device = brush_kernel::test_helpers::test_device().await;
    let num_points = 8;
    let means = Tensor::<MainBackend, 2>::zeros([num_points, 3], &device);
    let log_scales = Tensor::<MainBackend, 2>::ones([num_points, 3], &device) * 2.0;
    let quats: Tensor<MainBackend, 2> =
        Tensor::<MainBackend, 1>::from_floats(glam::Quat::IDENTITY.to_array(), &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, num_points);
    let sh_coeffs = Tensor::<MainBackend, 3>::ones([num_points, 1, 3], &device);
    let raw_opacity = Tensor::<MainBackend, 1>::zeros([num_points], &device);

    let splats = Splats::from_tensor_data(
        means,
        quats,
        log_scales,
        sh_coeffs,
        raw_opacity,
        SplatRenderMode::Default,
    );
    let (output, _render_aux) =
        render_splats(splats, &cam, img_size, Vec3::ZERO, None, TextureMode::Float).await;

    let rgb = output.clone().slice([0..32, 0..32, 0..3]);
    let alpha = output.slice([0..32, 0..32, 3..4]);
    let rgb_mean = rgb
        .mean()
        .to_data_async()
        .await
        .expect("readback")
        .as_slice::<f32>()
        .expect("Wrong type")[0];
    let alpha_mean = alpha
        .mean()
        .to_data_async()
        .await
        .expect("readback")
        .as_slice::<f32>()
        .expect("Wrong type")[0];
    assert_approx_eq!(rgb_mean, 0.0, 1e-5);
    assert_approx_eq!(alpha_mean, 0.0);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn renders_many_splats() {
    // Test rendering with a ton of gaussians to verify 2D dispatch works correctly.
    // This exceeds the 1D 65535 * 256 = 16.7M limit.
    let num_splats = 30_000_000;
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(64, 64);
    let device = brush_kernel::test_helpers::test_device().await;

    // Create random gaussians spread in front of the camera
    let means = Tensor::<MainBackend, 2>::random(
        [num_splats, 3],
        Distribution::Uniform(-2.0, 2.0),
        &device,
    );
    // Small scales so they don't cover everything
    let log_scales = Tensor::<MainBackend, 2>::random(
        [num_splats, 3],
        Distribution::Uniform(-4.0, -2.0),
        &device,
    );
    // Random rotations (will be normalized)
    let quats = Tensor::<MainBackend, 2>::random(
        [num_splats, 4],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    // Simple SH coefficients (just base color)
    let sh_coeffs = Tensor::<MainBackend, 3>::random(
        [num_splats, 1, 3],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );
    // Some visible, some not
    let raw_opacity =
        Tensor::<MainBackend, 1>::random([num_splats], Distribution::Uniform(-2.0, 2.0), &device);

    let splats = Splats::from_tensor_data(
        means,
        quats,
        log_scales,
        sh_coeffs,
        raw_opacity,
        SplatRenderMode::Default,
    );
    let (output, aux) =
        render_splats(splats, &cam, img_size, Vec3::ZERO, None, TextureMode::Float).await;

    assert!(
        aux.num_visible > 0,
        "30M splats in front of camera, none survived projection"
    );
    let pixels = read_finite(output).await;
    let any_nonbg = pixels.chunks_exact(4).any(|c| c[3] > 1e-3);
    assert!(any_nonbg, "30M splats rendered to an entirely empty image");
}

// ---------- Shared helpers for the stress / invariance tests ----------

// Pull pixels off device and assert no NaNs/infs.
async fn read_finite(output: Tensor<MainBackend, 3>) -> Vec<f32> {
    let data = output
        .to_data_async()
        .await
        .expect("readback")
        .to_vec::<f32>()
        .expect("data vec");
    assert!(data.iter().all(|v| v.is_finite()), "NaNs or infs in output");
    data
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[derive(Clone)]
struct Scene {
    means: Vec<[f32; 3]>,
    quats: Vec<[f32; 4]>,
    log_scales: Vec<[f32; 3]>,
    sh_dc: Vec<[f32; 3]>,
    raw_opacity: Vec<f32>,
}

impl Scene {
    fn len(&self) -> usize {
        self.means.len()
    }

    fn push(&mut self, other: &Self) {
        self.means.extend_from_slice(&other.means);
        self.quats.extend_from_slice(&other.quats);
        self.log_scales.extend_from_slice(&other.log_scales);
        self.sh_dc.extend_from_slice(&other.sh_dc);
        self.raw_opacity.extend_from_slice(&other.raw_opacity);
    }
}

// Deterministic pseudo-random generator so tests are reproducible.
fn rng_scene(
    num_splats: usize,
    mean_range: f32,
    log_scale_range: (f32, f32),
    opacity_range: (f32, f32),
    seed: u64,
) -> Scene {
    use std::num::Wrapping;
    // SplitMix64 — tiny, no deps, deterministic.
    let mut state = Wrapping(seed);
    let mut next = || {
        state += Wrapping(0x9E3779B97F4A7C15u64);
        let mut z = state.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z as f64 / u64::MAX as f64) as f32
    };
    let mut uniform = |lo: f32, hi: f32| lo + next() * (hi - lo);

    let mut means = Vec::with_capacity(num_splats);
    let mut quats = Vec::with_capacity(num_splats);
    let mut log_scales = Vec::with_capacity(num_splats);
    let mut sh_dc = Vec::with_capacity(num_splats);
    let mut raw_opacity = Vec::with_capacity(num_splats);
    for _ in 0..num_splats {
        means.push([
            uniform(-mean_range, mean_range),
            uniform(-mean_range, mean_range),
            uniform(-mean_range, mean_range),
        ]);
        // Non-normalized, will be normalized in-shader.
        let q = [
            uniform(-1.0, 1.0),
            uniform(-1.0, 1.0),
            uniform(-1.0, 1.0),
            uniform(-1.0, 1.0),
        ];
        quats.push(q);
        log_scales.push([
            uniform(log_scale_range.0, log_scale_range.1),
            uniform(log_scale_range.0, log_scale_range.1),
            uniform(log_scale_range.0, log_scale_range.1),
        ]);
        sh_dc.push([uniform(0.0, 1.0), uniform(0.0, 1.0), uniform(0.0, 1.0)]);
        raw_opacity.push(uniform(opacity_range.0, opacity_range.1));
    }
    Scene {
        means,
        quats,
        log_scales,
        sh_dc,
        raw_opacity,
    }
}

fn scene_to_splats(scene: &Scene, device: &WgpuDevice) -> Splats<MainBackend> {
    let n = scene.len();
    let means = Tensor::<MainBackend, 1>::from_floats(
        scene
            .means
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )
    .reshape([n, 3]);
    let quats = Tensor::<MainBackend, 1>::from_floats(
        scene
            .quats
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )
    .reshape([n, 4]);
    let log_scales = Tensor::<MainBackend, 1>::from_floats(
        scene
            .log_scales
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )
    .reshape([n, 3]);
    let sh = Tensor::<MainBackend, 1>::from_floats(
        scene
            .sh_dc
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )
    .reshape([n, 1, 3]);
    let opac = Tensor::<MainBackend, 1>::from_floats(scene.raw_opacity.as_slice(), device);
    Splats::from_tensor_data(means, quats, log_scales, sh, opac, SplatRenderMode::Default)
}

async fn render_scene(
    scene: &Scene,
    cam: &Camera,
    img_size: glam::UVec2,
    device: &WgpuDevice,
) -> Vec<f32> {
    let splats = scene_to_splats(scene, device);
    let (output, _aux) =
        render_splats(splats, cam, img_size, Vec3::ZERO, None, TextureMode::Float).await;
    read_finite(output).await
}

// Same scene rendered twice must produce bit-identical output.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn render_is_deterministic_on_large_splats() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(256, 256);
    let device = brush_kernel::test_helpers::test_device().await;

    let scene = rng_scene(20_000, 2.0, (0.5, 3.0), (-1.0, 2.0), 0xA11CE);

    let a = render_scene(&scene, &cam, img_size, &device).await;
    let b = render_scene(&scene, &cam, img_size, &device).await;

    let diff = max_abs_diff(&a, &b);
    assert_eq!(
        diff, 0.0,
        "render is nondeterministic across runs (max diff {diff})",
    );
}

// Appending culled splats (off-screen / behind camera / near-zero opacity)
// must leave the render bit-identical.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn hidden_splats_do_not_perturb_render() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(256, 256);
    let device = brush_kernel::test_helpers::test_device().await;

    let visible = rng_scene(5_000, 1.5, (0.0, 2.5), (0.0, 3.0), 0xBEEF);

    // Culled batch: mix of (a) opacity way below 1/255, (b) behind camera, and
    // (c) astronomically far off-screen. All must be rejected by project_forward.
    let mut hidden = rng_scene(20_000, 1.0, (-1.0, 1.0), (-20.0, -20.0), 0xDEAD);
    for (i, m) in hidden.means.iter_mut().enumerate() {
        match i % 3 {
            0 => { /* opacity already -20 → culled */ }
            1 => {
                *m = [0.0, 0.0, 1000.0]; // behind camera after viewmat
            }
            _ => {
                *m = [1e6, 1e6, 10.0]; // way off-screen
            }
        }
    }

    let mut combined = visible.clone();
    combined.push(&hidden);

    let visible_only = render_scene(&visible, &cam, img_size, &device).await;
    let with_hidden = render_scene(&combined, &cam, img_size, &device).await;

    let diff = max_abs_diff(&visible_only, &with_hidden);
    assert!(
        diff < 1e-5,
        "hidden splats changed the render (max diff {diff})",
    );
}

// Prepending culled splats shifts every real splat's global_gid; any
// bug in the global→compact gather step would show up as a perturbation.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn culled_prefix_does_not_perturb_render() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(256, 256);
    let device = brush_kernel::test_helpers::test_device().await;

    let visible = rng_scene(4_000, 1.5, (0.0, 2.5), (0.0, 3.0), 0xC0FFEE);

    // Culled prefix: opacity well below threshold (-20 → sigmoid ≈ 0).
    let prefix = rng_scene(50_000, 1.0, (-1.0, 1.0), (-20.0, -20.0), 0xF00D);

    let mut combined = prefix.clone();
    combined.push(&visible);

    let visible_only = render_scene(&visible, &cam, img_size, &device).await;
    let with_prefix = render_scene(&combined, &cam, img_size, &device).await;

    let diff = max_abs_diff(&visible_only, &with_prefix);
    assert!(
        diff < 1e-5,
        "culled prefix changed the render (max diff {diff})",
    );
}

// 120k fullscreen-sized splats stress the intersection buffer. Checks
// determinism (even after an unrelated render in between) plus "no dropped
// tiles" — every tile must receive contributions.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn mega_stress_fullscreen_splats() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(512, 512);
    let device = brush_kernel::test_helpers::test_device().await;

    // Force every splat to be huge: exp(3.5) ≈ 33 world units, at distance 5
    // with our focal this projects to a footprint larger than the image. Every
    // visible splat should end up hitting every tile.
    let scene = rng_scene(120_000, 0.1, (3.5, 4.0), (-3.0, -1.5), 0x5EED);

    let a = render_scene(&scene, &cam, img_size, &device).await;

    // Render a small unrelated scene in between to shake any cached buffer
    // state on the device.
    let filler = rng_scene(100, 0.5, (-1.0, 0.5), (0.0, 1.0), 0xFACE);
    let _ = render_scene(&filler, &cam, img_size, &device).await;

    let b = render_scene(&scene, &cam, img_size, &device).await;

    let diff = max_abs_diff(&a, &b);
    // Not bit-exact by design: project_forward uses atomicAdd(&num_visible)
    // to reserve compact-order slots, so tied depths (plentiful with 120k
    // splats packed into a narrow z-range) can land in different presort
    // positions run-to-run. Fixing this deterministically (prefix-sum
    // compaction or a 64-bit tie-broken depth sort) costs ~3-5% on real
    // workloads, so we tolerate a small blend-order delta instead.
    assert!(
        diff < 5e-5,
        "mega stress render is nondeterministic beyond tie-break tolerance (max diff {diff})"
    );

    // Per-tile alpha: no dropped tile. Image is [h, w, 4].
    let w = img_size.x as usize;
    let h = img_size.y as usize;
    let tile = 16usize;
    for ty in 0..(h / tile) {
        for tx in 0..(w / tile) {
            let mut sum = 0.0f32;
            for y in 0..tile {
                for x in 0..tile {
                    let pix = (ty * tile + y) * w + (tx * tile + x);
                    sum += a[pix * 4 + 3];
                }
            }
            assert!(
                sum > 1e-3,
                "dropped tile at ({tx},{ty}) in mega stress — alpha sum {sum}",
            );
        }
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn renders_large_rotated_splats() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(256, 256);
    let device = brush_kernel::test_helpers::test_device().await;

    // Big anisotropic splats stacked at origin — every splat covers the
    // whole image. If PF and MG disagree on tile count for even one splat,
    // other splats' intersection records get clobbered.
    let num_splats = 2048;
    let means = Tensor::<MainBackend, 2>::zeros([num_splats, 3], &device);
    let log_scales =
        Tensor::<MainBackend, 2>::random([num_splats, 3], Distribution::Uniform(1.0, 3.0), &device);
    let quats = Tensor::<MainBackend, 2>::random(
        [num_splats, 4],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let sh_coeffs = Tensor::<MainBackend, 3>::ones([num_splats, 1, 3], &device) * 0.5;
    // Low per-splat opacity so T doesn't hit the early-out for many splats.
    let raw_opacity = Tensor::<MainBackend, 1>::ones([num_splats], &device) * -4.0;

    let splats = Splats::from_tensor_data(
        means,
        quats,
        log_scales,
        sh_coeffs,
        raw_opacity,
        SplatRenderMode::Default,
    );
    let (output, _aux) =
        render_splats(splats, &cam, img_size, Vec3::ZERO, None, TextureMode::Float).await;

    // Every tile must have nonzero alpha — a dropped tile shows up as all zeros.
    let alpha = output
        .slice([0..img_size.y as usize, 0..img_size.x as usize, 3..4])
        .to_data_async()
        .await
        .expect("readback alpha")
        .to_vec::<f32>()
        .expect("alpha vec");

    let tile = 16usize;
    let w = img_size.x as usize;
    let h = img_size.y as usize;
    for ty in 0..(h / tile) {
        for tx in 0..(w / tile) {
            let mut sum = 0.0f32;
            for y in 0..tile {
                for x in 0..tile {
                    sum += alpha[(ty * tile + y) * w + (tx * tile + x)];
                }
            }
            let mean = sum / ((tile * tile) as f32);
            assert!(
                mean > 1e-3,
                "tile ({tx},{ty}) has mean alpha {mean} — looks like a dropped tile",
            );
        }
    }
}

// Overlapping anisotropic splats — worst-case per-splat tile coverage.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn renders_many_large_splats_stress() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(128, 128);
    let device = brush_kernel::test_helpers::test_device().await;

    let num_splats = 200_000;
    let means = Tensor::<MainBackend, 2>::random(
        [num_splats, 3],
        Distribution::Uniform(-0.5, 0.5),
        &device,
    );
    let log_scales = Tensor::<MainBackend, 2>::random(
        [num_splats, 3],
        Distribution::Uniform(-1.0, 2.5),
        &device,
    );
    let quats = Tensor::<MainBackend, 2>::random(
        [num_splats, 4],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let sh_coeffs = Tensor::<MainBackend, 3>::random(
        [num_splats, 1, 3],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let raw_opacity =
        Tensor::<MainBackend, 1>::random([num_splats], Distribution::Uniform(-2.0, 2.0), &device);

    let splats = Splats::from_tensor_data(
        means,
        quats,
        log_scales,
        sh_coeffs,
        raw_opacity,
        SplatRenderMode::Default,
    );
    let (output, _aux) =
        render_splats(splats, &cam, img_size, Vec3::ZERO, None, TextureMode::Float).await;

    // Sanity: no NaNs, alpha everywhere.
    let data = output
        .to_data_async()
        .await
        .expect("readback")
        .to_vec::<f32>()
        .expect("data vec");
    assert!(data.iter().all(|v| v.is_finite()), "NaNs in output");

    let alpha: Vec<f32> = data
        .chunks(4)
        .map(|chunk| *chunk.last().expect("alpha"))
        .collect();
    let tile = 16usize;
    let w = img_size.x as usize;
    let h = img_size.y as usize;
    let mut dropped_tiles = 0usize;
    for ty in 0..(h / tile) {
        for tx in 0..(w / tile) {
            let mut sum = 0.0f32;
            for y in 0..tile {
                for x in 0..tile {
                    sum += alpha[(ty * tile + y) * w + (tx * tile + x)];
                }
            }
            if sum < 1e-4 {
                dropped_tiles += 1;
            }
        }
    }
    // With 200k splats everywhere, every tile should have contributions.
    assert_eq!(dropped_tiles, 0, "detected dropped tiles in stress render");
}

#[allow(clippy::should_panic_without_expect)]
#[wasm_bindgen_test(unsupported = tokio::test)]
#[should_panic]
async fn render_panics_loudly_on_nan_positions() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(32, 32);
    let device = brush_kernel::test_helpers::test_device().await;
    let n = 16;

    let mut means_vec = vec![0.0f32; n * 3];
    means_vec[3] = f32::NAN; // one NaN position
    let means =
        Tensor::<MainBackend, 1>::from_floats(means_vec.as_slice(), &device).reshape([n, 3]);
    let quats: Tensor<MainBackend, 2> =
        Tensor::<MainBackend, 1>::from_floats([1.0, 0.0, 0.0, 0.0], &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, n);
    let log_scales = Tensor::<MainBackend, 2>::zeros([n, 3], &device);
    let sh_coeffs = Tensor::<MainBackend, 3>::ones([n, 1, 3], &device);
    let raw_opacity = Tensor::<MainBackend, 1>::zeros([n], &device);
    let splats = Splats::from_tensor_data(
        means,
        quats,
        log_scales,
        sh_coeffs,
        raw_opacity,
        SplatRenderMode::Default,
    );
    let _ = render_splats(splats, &cam, img_size, Vec3::ZERO, None, TextureMode::Float).await;
}

// Zero-splat Splats must not crash and must render every pixel as the
// background color. Reading pixels back forces fusion to flush, which is
// what catches bugs in the empty-tensor code paths.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn zero_splats_renders_background() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(32, 32);
    let device = brush_kernel::test_helpers::test_device().await;

    let splats = Splats::from_tensor_data(
        Tensor::<MainBackend, 2>::zeros([0, 3], &device),
        Tensor::<MainBackend, 2>::zeros([0, 4], &device),
        Tensor::<MainBackend, 2>::zeros([0, 3], &device),
        Tensor::<MainBackend, 3>::zeros([0, 1, 3], &device),
        Tensor::<MainBackend, 1>::zeros([0], &device),
        SplatRenderMode::Default,
    );
    assert_eq!(splats.num_splats(), 0);

    let bg = glam::vec3(0.7, 0.3, 0.1);
    let (output, _aux) = render_splats(splats, &cam, img_size, bg, None, TextureMode::Float).await;
    let pixels = output
        .to_data_async()
        .await
        .expect("readback")
        .to_vec::<f32>()
        .expect("data vec");
    let n_pixels = (img_size.x * img_size.y) as usize;
    assert_eq!(pixels.len(), n_pixels * 4);
    for (i, chunk) in pixels.chunks_exact(4).enumerate() {
        let [r, g, b, a] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        assert!(
            (r - bg.x).abs() < 1e-5
                && (g - bg.y).abs() < 1e-5
                && (b - bg.z).abs() < 1e-5
                && a.abs() < 1e-5,
            "pixel {i} = ({r}, {g}, {b}, {a}), expected background ({}, {}, {}, 0)",
            bg.x,
            bg.y,
            bg.z,
        );
    }
}

// Zero-length quats must be culled by PF, adding them to a scene must
// not change the rendered pixels.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn zero_quaternion_splats_dont_poison_render() {
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(64, 64);
    let device = brush_kernel::test_helpers::test_device().await;

    // Half valid splats, half with zero-length quaternion. The zero-quat ones
    // should be culled cleanly and the output should match rendering just the
    // valid half.
    let valid = rng_scene(500, 1.0, (-1.0, 1.0), (0.0, 2.0), 0xDEAD);
    let mut with_zeros = valid.clone();
    // Append 500 splats with zero quaternions at the same valid positions.
    for i in 0..500 {
        with_zeros.means.push(valid.means[i]);
        with_zeros.quats.push([0.0, 0.0, 0.0, 0.0]);
        with_zeros.log_scales.push(valid.log_scales[i]);
        with_zeros.sh_dc.push(valid.sh_dc[i]);
        with_zeros.raw_opacity.push(valid.raw_opacity[i]);
    }

    let clean = render_scene(&valid, &cam, img_size, &device).await;
    let with_zeros_px = render_scene(&with_zeros, &cam, img_size, &device).await;
    assert!(
        max_abs_diff(&clean, &with_zeros_px) < 1e-5,
        "zero-quaternion splats perturbed the render (they should have been culled)"
    );
}
