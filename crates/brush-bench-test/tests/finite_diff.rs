//! Finite-difference gradient sanity check for the splat backward pass.
//!
//! For each parameter category, perturb a single scalar component, render
//! both sides of a central difference, and compare to the analytical
//! gradient from `loss.backward()`. Broad-strokes — a few indices per
//! category at default tolerances — to flag whether any category's
//! backward is grossly wrong before drilling in.
//!
//! Scene is hand-tuned to avoid pipeline discontinuities (opacity well
//! above the 1/255 cutoff, splats comfortably inside the image, scales
//! away from f16 quantization limits) so central differences are
//! second-order accurate.

use brush_render::gaussian_splats::RasterPass;
use brush_render::{
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::{
        CameraModel, kannala_brandt_4::KannalaBrandt4Params,
        radial_tangential_8::RadialTangential8Params, thin_prism_fisheye::ThinPrismFisheyeParams,
    },
};
use brush_render_bwd::render_splats_with_pass;

/// Finite-diff tests need the C^1 cutoff so analytical and numerical
/// agree at typical eps; production paths use the hard step.
const PASS: RasterPass = RasterPass::BackwardSmoothCutoff;
use burn::tensor::{Gradients, Tensor, s};
use glam::Vec3;

#[derive(Clone)]
struct Scene {
    means: Vec<f32>,
    rots: Vec<f32>,
    log_scales: Vec<f32>,
    sh_dc: Vec<f32>,
    raw_opac: Vec<f32>,
}

/// 4 splats placed comfortably inside the image, fully visible, moderate
/// opacity, non-axis-aligned rotations so the full rotation Jacobian is
/// exercised.
fn base_scene() -> Scene {
    Scene {
        means: vec![
            0.20, -0.10, 0.00, //
            -0.30, 0.40, 0.20, //
            0.10, 0.30, -0.30, //
            -0.20, -0.20, 0.10, //
        ],
        // Non-unit, non-axis-aligned — shader normalizes.
        rots: vec![
            0.90, 0.10, 0.05, 0.03, //
            0.70, 0.20, 0.30, 0.10, //
            0.50, 0.40, 0.30, 0.20, //
            0.80, 0.10, 0.10, 0.20, //
        ],
        log_scales: vec![
            -1.4, -1.5, -1.6, //
            -1.5, -1.4, -1.3, //
            -1.7, -1.5, -1.4, //
            -1.3, -1.6, -1.5, //
        ],
        sh_dc: vec![
            0.45, 0.55, 0.50, //
            0.60, 0.40, 0.30, //
            0.35, 0.50, 0.65, //
            0.50, 0.45, 0.55, //
        ],
        raw_opac: vec![2.5, 2.0, 2.2, 2.4],
    }
}

fn std_cam() -> Camera {
    Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

fn build_splats(scene: &Scene, device: &burn::tensor::Device) -> Splats {
    Splats::from_raw(
        scene.means.clone(),
        scene.rots.clone(),
        scene.log_scales.clone(),
        scene.sh_dc.clone(),
        scene.raw_opac.clone(),
        SplatRenderMode::Default,
        device,
    )
}

async fn render_value(
    scene: &Scene,
    cam: &Camera,
    img_size: glam::UVec2,
    device: &burn::tensor::Device,
) -> f32 {
    let splats = build_splats(scene, device);
    let diff = {
        let splats = splats;
        let cam: &Camera = cam;
        let background = Vec3::ZERO;
        async move { render_splats_with_pass(splats, cam, img_size, background, PASS, 0.0).await }
    }
    .await;
    diff.img
        .mean()
        .into_scalar_async::<f32>()
        .await
        .expect("loss readback")
}

async fn analytical_grads(
    scene: &Scene,
    cam: &Camera,
    img_size: glam::UVec2,
    device: &burn::tensor::Device,
) -> (Splats, Gradients) {
    let splats = build_splats(scene, device);
    let diff = {
        let splats = splats.clone();
        let cam: &Camera = cam;
        let background = Vec3::ZERO;
        async move { render_splats_with_pass(splats, cam, img_size, background, PASS, 0.0).await }
    }
    .await;
    let grads = diff.img.mean().backward();
    (splats, grads)
}

#[derive(Clone, Copy)]
enum Lane {
    Mean,
    Rot,
    LogScale,
    ShDc,
    RawOpac,
}

fn lane_name(lane: Lane) -> &'static str {
    match lane {
        Lane::Mean => "means",
        Lane::Rot => "rots",
        Lane::LogScale => "log_scales",
        Lane::ShDc => "sh_dc",
        Lane::RawOpac => "raw_opac",
    }
}

fn perturb(scene: &mut Scene, lane: Lane, splat: usize, comp: usize, delta: f32) {
    match lane {
        Lane::Mean => scene.means[splat * 3 + comp] += delta,
        Lane::Rot => scene.rots[splat * 4 + comp] += delta,
        Lane::LogScale => scene.log_scales[splat * 3 + comp] += delta,
        Lane::ShDc => scene.sh_dc[splat * 3 + comp] += delta,
        Lane::RawOpac => {
            assert_eq!(comp, 0, "raw_opac is per-splat scalar");
            scene.raw_opac[splat] += delta;
        }
    }
}

async fn read_first<const D: usize>(t: Tensor<D>) -> f32 {
    t.into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .expect("vec")[0]
}

async fn analytical_at(
    splats: &Splats,
    grads: &Gradients,
    lane: Lane,
    splat: usize,
    comp: usize,
) -> f32 {
    match lane {
        Lane::Mean => {
            let g = splats.transforms.grad(grads).expect("transforms grad");
            read_first(g.slice(s![splat..splat + 1, comp..comp + 1])).await
        }
        Lane::Rot => {
            let g = splats.transforms.grad(grads).expect("transforms grad");
            let c = 3 + comp;
            read_first(g.slice(s![splat..splat + 1, c..c + 1])).await
        }
        Lane::LogScale => {
            let g = splats.transforms.grad(grads).expect("transforms grad");
            let c = 7 + comp;
            read_first(g.slice(s![splat..splat + 1, c..c + 1])).await
        }
        Lane::ShDc => {
            let g = splats.sh_coeffs.grad(grads).expect("sh grad");
            read_first(g.slice(s![splat..splat + 1, 0..1, comp..comp + 1])).await
        }
        Lane::RawOpac => {
            let g = splats.raw_opacities.grad(grads).expect("opac grad");
            read_first(g.slice(s![splat..splat + 1])).await
        }
    }
}

#[tokio::test]
async fn finite_difference_gradient_broad() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(32, 32);
    let scene = base_scene();

    let eps = 3e-4_f32;
    let rel_tol = 0.01_f32;
    let abs_tol = 5e-5_f32;

    let (splats, grads) = analytical_grads(&scene, &cam, img_size, &device).await;

    let cases: &[(Lane, usize, usize)] = &[
        (Lane::Mean, 0, 0),
        (Lane::Mean, 0, 2),
        (Lane::Mean, 1, 1),
        (Lane::Rot, 0, 0),
        (Lane::Rot, 1, 2),
        (Lane::LogScale, 0, 0),
        (Lane::LogScale, 1, 1),
        (Lane::ShDc, 0, 0),
        (Lane::ShDc, 1, 1),
        (Lane::ShDc, 2, 2),
        (Lane::RawOpac, 0, 0),
        (Lane::RawOpac, 2, 0),
    ];
    let mut rows: Vec<(Lane, usize, usize, f32, f32)> = Vec::with_capacity(cases.len());
    for (lane, splat, comp) in cases {
        let mut s_plus = scene.clone();
        perturb(&mut s_plus, *lane, *splat, *comp, eps);
        let l_plus = render_value(&s_plus, &cam, img_size, &device).await;

        let mut s_minus = scene.clone();
        perturb(&mut s_minus, *lane, *splat, *comp, -eps);
        let l_minus = render_value(&s_minus, &cam, img_size, &device).await;

        let numerical = (l_plus - l_minus) / (2.0 * eps);
        let an = analytical_at(&splats, &grads, *lane, *splat, *comp).await;

        rows.push((*lane, *splat, *comp, numerical, an));
    }
    let mut failed: Vec<String> = Vec::new();
    for (lane, splat, comp, numerical, an) in &rows {
        let abs_err = (numerical - an).abs();
        let scale = numerical.abs().max(an.abs()).max(1e-8);
        let tol = abs_tol + rel_tol * scale;

        if abs_err > tol {
            failed.push(format!(
                "{}[{},{}]: numerical {numerical:.6} vs analytical {an:.6} \
                 (|Δ|={abs_err:.3e} > tol {tol:.3e})",
                lane_name(*lane),
                splat,
                comp,
            ));
        }
    }

    assert!(
        failed.is_empty(),
        "finite-diff vs analytical mismatch:\n  {}",
        failed.join("\n  "),
    );
}

/// Tangential quat perturbations only — radial direction is projected
/// out by the shader's quat normalization. Strongly anisotropic scales
/// so z-axis rotation gives a measurable gradient.
#[tokio::test]
async fn finite_diff_tangential_quat() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);
    let eps = 3e-4_f32;

    let mut scene = base_scene();
    scene.rots[0] = 1.0;
    scene.rots[1] = 0.0;
    scene.rots[2] = 0.0;
    scene.rots[3] = 0.0;
    scene.log_scales[0] = -1.0;
    scene.log_scales[1] = -2.8;
    scene.log_scales[2] = -1.5;

    let (splats, grads) = analytical_grads(&scene, &cam, img_size, &device).await;
    let g = splats.transforms.grad(&grads).expect("transforms grad");
    let q_grad: Vec<f32> = g
        .slice(s![0..1, 3..7])
        .into_data_async()
        .await
        .expect("rb")
        .into_vec::<f32>()
        .expect("v");

    // Radial direction (parallel to identity quat) must have ~0 grad —
    // the normalization projects it out.
    let radial = [1.0_f32, 0.0, 0.0, 0.0];
    let radial_grad: f32 = (0..4).map(|i| q_grad[i] * radial[i]).sum();
    assert!(
        radial_grad.abs() < 1e-4,
        "radial quat grad should be ~0 (normalization), got {radial_grad}",
    );

    // Unit tangents: i / j / k components of the quat, plus one mixed
    // direction. All orthogonal to the identity quat.
    let tangents: &[[f32; 4]] = &[
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.6, 0.0, 0.8],
    ];

    let mut failed: Vec<String> = Vec::new();
    for t in tangents {
        let mut s_plus = scene.clone();
        let mut s_minus = scene.clone();
        for i in 0..4 {
            s_plus.rots[i] += eps * t[i];
            s_minus.rots[i] -= eps * t[i];
        }
        let l_plus = render_value(&s_plus, &cam, img_size, &device).await;
        let l_minus = render_value(&s_minus, &cam, img_size, &device).await;
        let numerical = (l_plus - l_minus) / (2.0 * eps);
        let an: f32 = (0..4).map(|i| q_grad[i] * t[i]).sum();

        let abs_err = (numerical - an).abs();
        let scale = numerical.abs().max(an.abs()).max(1e-8);

        let tol = 5e-5_f32 + 0.01 * scale;
        if abs_err > tol {
            failed.push(format!(
                "{t:?}: num {numerical:.6} vs an {an:.6} (|Δ|={abs_err:.3e} > tol {tol:.3e})"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "tangent-quat mismatches:\n  {}",
        failed.join("\n  ")
    );
}

/// Same broad-strokes check but in `SplatRenderMode::Mip`. Mip adds a
/// blur term to small splats; backward path is distinct.
#[tokio::test]
async fn finite_diff_broad_mip_mode() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(32, 32);
    let scene = base_scene();
    let eps = 3e-4_f32;
    let rel_tol = 0.03_f32; // looser — mip-mode amplifies noise
    let abs_tol = 1e-4_f32;

    // Mip-mode versions of the helpers (inline so we don't mutate the
    // Default-mode helpers above).
    async fn render_value_mip(
        scene: &Scene,
        cam: &Camera,
        img_size: glam::UVec2,
        device: &burn::tensor::Device,
    ) -> f32 {
        let splats = Splats::from_raw(
            scene.means.clone(),
            scene.rots.clone(),
            scene.log_scales.clone(),
            scene.sh_dc.clone(),
            scene.raw_opac.clone(),
            SplatRenderMode::Mip,
            device,
        );
        let diff = render_splats_with_pass(splats, cam, img_size, Vec3::ZERO, PASS, 0.0).await;
        diff.img
            .mean()
            .into_scalar_async::<f32>()
            .await
            .expect("rb")
    }

    async fn grads_mip(
        scene: &Scene,
        cam: &Camera,
        img_size: glam::UVec2,
        device: &burn::tensor::Device,
    ) -> (Splats, Gradients) {
        let splats = Splats::from_raw(
            scene.means.clone(),
            scene.rots.clone(),
            scene.log_scales.clone(),
            scene.sh_dc.clone(),
            scene.raw_opac.clone(),
            SplatRenderMode::Mip,
            device,
        );
        let diff =
            render_splats_with_pass(splats.clone(), cam, img_size, Vec3::ZERO, PASS, 0.0).await;
        let g = diff.img.mean().backward();
        (splats, g)
    }

    let (splats, grads) = grads_mip(&scene, &cam, img_size, &device).await;
    let cases: &[(Lane, usize, usize)] = &[
        (Lane::Mean, 0, 0),
        (Lane::Mean, 1, 1),
        (Lane::LogScale, 0, 0),
        (Lane::LogScale, 1, 1),
        (Lane::ShDc, 0, 0),
        (Lane::RawOpac, 0, 0),
    ];

    let mut failed: Vec<String> = Vec::new();
    for (lane, splat, comp) in cases {
        let mut s_plus = scene.clone();
        perturb(&mut s_plus, *lane, *splat, *comp, eps);
        let l_plus = render_value_mip(&s_plus, &cam, img_size, &device).await;
        let mut s_minus = scene.clone();
        perturb(&mut s_minus, *lane, *splat, *comp, -eps);
        let l_minus = render_value_mip(&s_minus, &cam, img_size, &device).await;
        let numerical = (l_plus - l_minus) / (2.0 * eps);
        let an = analytical_at(&splats, &grads, *lane, *splat, *comp).await;

        let abs_err = (numerical - an).abs();
        let scale = numerical.abs().max(an.abs()).max(1e-8);
        let tol = abs_tol + rel_tol * scale;
        if abs_err > tol {
            failed.push(format!(
                "{}[{},{}]: num {numerical:.6} an {an:.6} (|Δ|={abs_err:.3e} > {tol:.3e})",
                lane_name(*lane),
                splat,
                comp,
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "mip-mode mismatches:\n  {}",
        failed.join("\n  ")
    );
}

/// Per-pixel weighted-sum loss. `mean()` averages all pixel grads
/// uniformly, so any per-pixel asymmetry in the backward is invisible.
/// Multiplying by a fixed random weight map and summing exposes pixels
/// where forward and backward disagree on the local Jacobian.
#[tokio::test]
async fn finite_diff_weighted_loss() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(32, 32);
    let scene = base_scene();
    let eps = 3e-4_f32;

    // Deterministic weights so this matches across runs.
    let h = img_size.y as usize;
    let w = img_size.x as usize;
    let c = 4;
    let mut weights_data = Vec::with_capacity(h * w * c);
    let mut state: u64 = 0xCAFE_F00D;
    for _ in 0..(h * w * c) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((state >> 33) as u32) as f32 / u32::MAX as f32;
        weights_data.push(f); // [0, 1)
    }
    let weights: Tensor<3> =
        Tensor::<1>::from_floats(weights_data.as_slice(), &device).reshape([h, w, c]);

    async fn weighted_value(
        scene: &Scene,
        weights: Tensor<3>,
        cam: &Camera,
        img_size: glam::UVec2,
        device: &burn::tensor::Device,
    ) -> f32 {
        let splats = build_splats(scene, device);
        let diff = render_splats_with_pass(splats, cam, img_size, Vec3::ZERO, PASS, 0.0).await;
        (diff.img * weights)
            .sum()
            .into_scalar_async::<f32>()
            .await
            .expect("rb")
    }

    async fn weighted_grads(
        scene: &Scene,
        weights: Tensor<3>,
        cam: &Camera,
        img_size: glam::UVec2,
        device: &burn::tensor::Device,
    ) -> (Splats, Gradients) {
        let splats = build_splats(scene, device);
        let diff =
            render_splats_with_pass(splats.clone(), cam, img_size, Vec3::ZERO, PASS, 0.0).await;
        let loss = (diff.img * weights).sum();
        (splats, loss.backward())
    }

    let (splats, grads) = weighted_grads(&scene, weights.clone(), &cam, img_size, &device).await;

    let cases: &[(Lane, usize, usize)] = &[
        (Lane::Mean, 0, 0),
        (Lane::Mean, 1, 1),
        (Lane::LogScale, 0, 0),
        (Lane::ShDc, 1, 1),
        (Lane::RawOpac, 0, 0),
        (Lane::RawOpac, 2, 0),
    ];

    let mut failed: Vec<String> = Vec::new();
    for (lane, splat, comp) in cases {
        let mut s_plus = scene.clone();
        perturb(&mut s_plus, *lane, *splat, *comp, eps);
        let l_plus = weighted_value(&s_plus, weights.clone(), &cam, img_size, &device).await;
        let mut s_minus = scene.clone();
        perturb(&mut s_minus, *lane, *splat, *comp, -eps);
        let l_minus = weighted_value(&s_minus, weights.clone(), &cam, img_size, &device).await;
        let numerical = (l_plus - l_minus) / (2.0 * eps);
        let an = analytical_at(&splats, &grads, *lane, *splat, *comp).await;

        let abs_err = (numerical - an).abs();
        let scale = numerical.abs().max(an.abs()).max(1e-8);
        let tol = 0.5 + 0.02 * scale; // sum-loss values are O(100s), tolerate larger absolute
        if abs_err > tol {
            failed.push(format!(
                "{}[{},{}]: num {numerical:.6} an {an:.6} (|Δ|={abs_err:.3e} > {tol:.3e})",
                lane_name(*lane),
                splat,
                comp,
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "weighted-loss mismatches:\n  {}",
        failed.join("\n  ")
    );
}

// ---- Fuzz helpers ----

struct Sm64(std::num::Wrapping<u64>);

impl Sm64 {
    fn new(seed: u64) -> Self {
        Self(std::num::Wrapping(seed))
    }
    fn u64_(&mut self) -> u64 {
        self.0 += std::num::Wrapping(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f01(&mut self) -> f32 {
        (self.u64_() as f64 / u64::MAX as f64) as f32
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f01() * (hi - lo)
    }
    fn usize_in(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.u64_() as usize % (hi - lo))
    }
}

fn random_scene(seed: u64, n: usize) -> Scene {
    let mut rng = Sm64::new(seed.wrapping_mul(0x517C_C1B7_2722_0A95));
    Scene {
        means: (0..n * 3).map(|_| rng.uniform(-1.0, 1.0)).collect(),
        rots: (0..n * 4).map(|_| rng.uniform(-1.0, 1.0)).collect(),
        // Range chosen to keep splats visible but not span an entire tile:
        // exp(-2.5)≈0.08 to exp(0)=1.0. Avoids cov2d blowing up.
        log_scales: (0..n * 3).map(|_| rng.uniform(-2.5, 0.0)).collect(),
        sh_dc: (0..n * 3).map(|_| rng.uniform(0.2, 0.8)).collect(),
        // raw_opac ≥ 0.5 → sigmoid ≥ 0.62, comfortably above the 1/255
        // cutoff. Avoids the known discontinuity from the cutoff test.
        raw_opac: (0..n).map(|_| rng.uniform(0.5, 3.0)).collect(),
    }
}

fn random_camera(seed: u64) -> Camera {
    let mut rng = Sm64::new(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0xCAFE);
    let dist = rng.uniform(2.5, 5.0);
    let cam_pos = glam::vec3(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -dist);
    let fov = rng.uniform(0.4, 0.9) as f64;
    Camera::new(
        cam_pos,
        glam::Quat::IDENTITY,
        fov,
        fov,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

/// Pick a uniformly random parameter slot for a given scene size.
fn random_param(rng: &mut Sm64, n_splats: usize) -> (Lane, usize, usize) {
    let lane = match rng.usize_in(0, 5) {
        0 => Lane::Mean,
        1 => Lane::Rot,
        2 => Lane::LogScale,
        3 => Lane::ShDc,
        _ => Lane::RawOpac,
    };
    let splat = rng.usize_in(0, n_splats);
    let comp = match lane {
        Lane::Mean | Lane::LogScale | Lane::ShDc => rng.usize_in(0, 3),
        Lane::Rot => rng.usize_in(0, 4),
        Lane::RawOpac => 0,
    };
    (lane, splat, comp)
}

/// Same as `random_param` but never picks `Lane::ShDc`. Used when the
/// scene packs higher-than-degree-0 SH in `Scene::sh_dc` — the generic
/// `perturb` indexes `sh_dc[splat * 3 + comp]`, which is correct for
/// degree 0 but lands on the wrong coefficient for higher degrees.
/// Higher-band SH gradients are already covered by `finite_diff_sh_degree2`.
fn random_param_no_sh(rng: &mut Sm64, n_splats: usize) -> (Lane, usize, usize) {
    let lane = match rng.usize_in(0, 4) {
        0 => Lane::Mean,
        1 => Lane::Rot,
        2 => Lane::LogScale,
        _ => Lane::RawOpac,
    };
    let splat = rng.usize_in(0, n_splats);
    let comp = match lane {
        Lane::Mean | Lane::LogScale => rng.usize_in(0, 3),
        Lane::Rot => rng.usize_in(0, 4),
        Lane::ShDc => unreachable!(),
        Lane::RawOpac => 0,
    };
    (lane, splat, comp)
}

/// Fuzz: random scenes × random parameter picks. Reproduce a specific
/// failure by setting `seed = N` in the loop.
#[tokio::test]
async fn fuzz_finite_diff_random_scenes() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_add(0xDEAD_BEEF));
        let n = rng.usize_in(3, 9);
        (random_scene(seed, n), n, "rng".into())
    };
    let results = run_obscure_fuzz(&device, glam::uvec2(32, 32), 5, scene_fn, random_camera).await;
    assert_fuzz_clean(&results, 0.02, 2e-4, "rng");
}

/// **T early-out** (`rasterize.rs:130`): `next_t <= 1e-4 → done`. With
/// many opaque splats stacked front-to-back, the rasterizer stops
/// after a few contribute, and behind-them splats get zero forward
/// contribution. Their analytical gradient must also be zero.
#[tokio::test]
async fn finite_diff_t_early_out_back_splats() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(32, 32);

    // 30 opaque, large splats at the same xy, stepping back in z.
    // Front-to-back sort means the closest few saturate T < 1e-4, the
    // rest are skipped entirely.
    let n = 30;
    let mut means = Vec::with_capacity(n * 3);
    for i in 0..n {
        means.extend_from_slice(&[0.0, 0.0, i as f32 * 0.3]);
    }
    let rots: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    let log_scales: Vec<f32> = (0..n).flat_map(|_| [-0.3, -0.3, -0.3]).collect();
    let sh_dc: Vec<f32> = (0..n)
        .flat_map(|i| {
            let c = 0.3 + 0.4 * (i as f32 / n as f32);
            [c, c, c]
        })
        .collect();
    let raw_opac: Vec<f32> = (0..n).map(|_| 4.0).collect(); // sigmoid ≈ 0.982 → ~3 splats kill T
    let scene = Scene {
        means,
        rots,
        log_scales,
        sh_dc,
        raw_opac,
    };

    let (splats, grads) = analytical_grads(&scene, &cam, img_size, &device).await;

    // Three probes: near (idx 1), middle (idx 10), far (idx 25).
    // Expect: near has large grad, far has ~zero grad. Both should
    // agree with numerical.
    let eps = 1e-3_f32;
    for splat_idx in [1_usize, 5, 10, 15, 20, 25, 29] {
        let mut s_plus = scene.clone();
        s_plus.raw_opac[splat_idx] += eps;
        let l_plus = render_value(&s_plus, &cam, img_size, &device).await;
        let mut s_minus = scene.clone();
        s_minus.raw_opac[splat_idx] -= eps;
        let l_minus = render_value(&s_minus, &cam, img_size, &device).await;
        let numerical = (l_plus - l_minus) / (2.0 * eps);
        let g = splats.raw_opacities.grad(&grads).expect("opac grad");
        let an = read_first(g.slice(s![splat_idx..splat_idx + 1])).await;
        let abs_diff = (numerical - an).abs();
        assert!(
            abs_diff < 5e-4,
            "splat {splat_idx}: numerical {numerical} vs analytical {an}, abs diff {abs_diff}"
        );
    }
}

// ---- Camera-model fuzz (Pinhole / KB4 / RT8) ----

/// Random camera with a randomly chosen model. Splats around the
/// origin → fov range chosen so they're well within view for all
/// three models, and distortion params are small so projection stays
/// well-conditioned. The KB4/RT8 paths use different cov2d Jacobian
/// formulas than Pinhole — this fuzz exercises both backward paths.
fn random_camera_with_model(seed: u64) -> (Camera, &'static str) {
    let mut rng = Sm64::new(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0xC0DE);
    let dist = rng.uniform(2.5, 5.0);
    let cam_pos = glam::vec3(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), -dist);
    let fov = rng.uniform(0.5, 1.0) as f64;
    let center = glam::vec2(0.5, 0.5);
    let model = match rng.usize_in(0, 4) {
        0 => CameraModel::Pinhole,
        1 => CameraModel::KannalaBrandt4(KannalaBrandt4Params {
            k1: rng.uniform(-0.05, 0.05),
            k2: rng.uniform(-0.02, 0.02),
            k3: rng.uniform(-0.01, 0.01),
            k4: rng.uniform(-0.005, 0.005),
        }),
        2 => CameraModel::RadialTangential8(RadialTangential8Params {
            k1: rng.uniform(-0.1, 0.1),
            k2: rng.uniform(-0.05, 0.05),
            k3: rng.uniform(-0.01, 0.01),
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: rng.uniform(-0.01, 0.01),
            p2: rng.uniform(-0.01, 0.01),
        }),
        _ => CameraModel::ThinPrismFisheye(ThinPrismFisheyeParams {
            kb4: KannalaBrandt4Params {
                k1: rng.uniform(-0.05, 0.05),
                k2: rng.uniform(-0.02, 0.02),
                k3: rng.uniform(-0.005, 0.005),
                k4: rng.uniform(-0.001, 0.001),
            },
            p1: rng.uniform(-0.01, 0.01),
            p2: rng.uniform(-0.01, 0.01),
            sx1: rng.uniform(-0.005, 0.005),
            sy1: rng.uniform(-0.005, 0.005),
        }),
    };
    let label = match model {
        CameraModel::Pinhole => "Pinhole",
        CameraModel::KannalaBrandt4(_) => "KB4",
        CameraModel::RadialTangential8(_) => "RT8",
        CameraModel::ThinPrismFisheye(_) => "TPF",
    };
    (
        Camera::new(cam_pos, glam::Quat::IDENTITY, fov, fov, center, model),
        label,
    )
}

/// Fuzz with random camera model — Pinhole / KB4 / RT8 picked per seed.
#[tokio::test]
async fn fuzz_finite_diff_camera_models() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_add(0xC0DE_BEEF));
        let n = rng.usize_in(3, 9);
        let (_, model_name) = random_camera_with_model(seed);
        (random_scene(seed, n), n, model_name.into())
    };
    let cam_fn = |seed: u64| random_camera_with_model(seed).0;
    let results = run_obscure_fuzz(&device, glam::uvec2(32, 32), 20, scene_fn, cam_fn).await;
    assert_fuzz_clean(&results, 0.02, 2e-4, "cam-models");
}

// ---- Obscure fuzz: stress less-covered axes ----

/// Row layout for fuzz-style results.
/// (`min_rel`, seed, splat, comp, lane, `best_numerical`, an, `best_eps`, tag)
type FuzzRow = (f32, u64, usize, usize, Lane, f32, f32, f32, String);

/// Shared fuzz inner loop. `scene_fn` and `cam_fn` per seed; this drives
/// the eps sweep, picks `picks_per_scene` random params per scene, and
/// returns the sorted-descending rows.
async fn run_obscure_fuzz<S, C>(
    device: &burn::tensor::Device,
    img_size: glam::UVec2,
    n_iter: u64,
    scene_fn: S,
    cam_fn: C,
) -> Vec<FuzzRow>
where
    S: Fn(u64) -> (Scene, usize, String),
    C: Fn(u64) -> Camera,
{
    run_obscure_fuzz_with(device, img_size, n_iter, scene_fn, cam_fn, random_param).await
}

/// Default eps used by every fuzz probe. With the smooth alpha cutoff
/// (`ALPHA_CUTOFF_BAND`), any reasonable single eps in this range gives
/// clean finite-diff agreement, so the previous 5-eps min-of-sweep
/// machinery isn't needed.
const FUZZ_EPS: f32 = 3e-4;

async fn run_obscure_fuzz_with<S, C, P>(
    device: &burn::tensor::Device,
    img_size: glam::UVec2,
    n_iter: u64,
    scene_fn: S,
    cam_fn: C,
    param_fn: P,
) -> Vec<FuzzRow>
where
    S: Fn(u64) -> (Scene, usize, String),
    C: Fn(u64) -> Camera,
    P: Fn(&mut Sm64, usize) -> (Lane, usize, usize),
{
    let mut results: Vec<FuzzRow> = Vec::with_capacity(n_iter as usize);

    for seed in 0..n_iter {
        let (scene, n, tag) = scene_fn(seed);
        let cam = cam_fn(seed);
        let (splats, grads) = analytical_grads(&scene, &cam, img_size, device).await;

        let mut rng = Sm64::new(seed.wrapping_mul(0xA5A5_5A5A).wrapping_add(0x1234));
        let (lane, splat, comp) = param_fn(&mut rng, n);
        let an = analytical_at(&splats, &grads, lane, splat, comp).await;

        let mut s_plus = scene.clone();
        perturb(&mut s_plus, lane, splat, comp, FUZZ_EPS);
        let l_plus = render_value(&s_plus, &cam, img_size, device).await;
        let mut s_minus = scene.clone();
        perturb(&mut s_minus, lane, splat, comp, -FUZZ_EPS);
        let l_minus = render_value(&s_minus, &cam, img_size, device).await;
        let numerical = (l_plus - l_minus) / (2.0 * FUZZ_EPS);
        let scale = numerical.abs().max(an.abs()).max(1e-8);
        let rel = (numerical - an).abs() / scale;
        results.push((
            rel,
            seed,
            splat,
            comp,
            lane,
            numerical,
            an,
            FUZZ_EPS,
            tag.clone(),
        ));
    }
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    results
}

/// Assert worst-case mismatch under combined abs+rel tolerance. Pure
/// rel-err is misleading at the signal floor (small gradients can
/// fluctuate by large relative amounts even when abs-err is tiny);
/// combined tol matches the broad/tangent tests.
fn assert_fuzz_clean(results: &[FuzzRow], rel_tol: f32, abs_tol: f32, label: &str) {
    let mut failures: Vec<String> = Vec::new();
    for (_, seed, splat, comp, lane, num, an, eps, tag) in results {
        let abs_err = (num - an).abs();
        let scale = num.abs().max(an.abs()).max(1e-8);
        let tol = abs_tol + rel_tol * scale;
        if abs_err > tol {
            failures.push(format!(
                "seed {seed} tag {tag} splat {splat} comp {comp} param {} num {num:.6} an {an:.6} (|Δ|={abs_err:.3e} > tol {tol:.3e}) eps {eps}",
                lane_name(*lane),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "[{label}] {} mismatches under abs_tol={abs_tol:.0e} + rel_tol={:.1}% × scale:\n  {}",
        failures.len(),
        rel_tol * 100.0,
        failures.join("\n  "),
    );
}

/// **Non-identity camera rotation** — every fuzz so far used
/// `Quat::IDENTITY`. This exercises the `view_rotation()` multiply in
/// both cov2d construction (`helpers.rs:118`) and the projection vjp
/// (`project_backwards.rs:218,222`).
#[tokio::test]
async fn fuzz_obscure_rotated_cameras() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(32, 32);

    let cam_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0xCAFE_C0DE) ^ 0x1234_5678);
        // Random axis-angle rotation. Modest angles (±60°) so most
        // splats stay in view.
        let ax = rng.uniform(-1.0, 1.0);
        let ay = rng.uniform(-1.0, 1.0);
        let az = rng.uniform(-1.0, 1.0);
        let axis = glam::Vec3::new(ax, ay, az).normalize_or_zero();
        let angle = rng.uniform(-1.0, 1.0); // radians, ~±57°
        let rot = if axis == glam::Vec3::ZERO {
            glam::Quat::IDENTITY
        } else {
            glam::Quat::from_axis_angle(axis, angle)
        };
        // Camera position: 4 units out, facing the origin direction
        // implied by the rotation.
        let cam_pos = rot * glam::vec3(0.0, 0.0, -4.0);
        let fov = rng.uniform(0.5, 0.9) as f64;
        Camera::new(
            cam_pos,
            rot,
            fov,
            fov,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    };
    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_add(0xCAFE));
        let n = rng.usize_in(3, 8);
        (random_scene(seed, n), n, "rot".into())
    };

    let results = run_obscure_fuzz(&device, img_size, 20, scene_fn, cam_fn).await;
    assert_fuzz_clean(&results, 0.03, 5e-4, "rot-cam");
}

/// **Off-center principal point + asymmetric focal length**. All
/// previous tests put cx, cy at the image center and used fx=fy. These
/// fields are independent in `PinholeParams` and could mask bugs if
/// only the symmetric case was ever tested.
#[tokio::test]
async fn fuzz_obscure_off_center_principal_and_asymmetric_focal() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(48, 48);

    let cam_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0xBEEF_FEED));
        let dist = rng.uniform(3.0, 5.0);
        // Random principal point inside the central 60% — keep it well
        // off-center, but not so far that all splats land out of view.
        let cx_n = rng.uniform(0.2, 0.8);
        let cy_n = rng.uniform(0.2, 0.8);
        // Strongly asymmetric fov (up to ~2:1 aspect ratio difference).
        let fov_x = rng.uniform(0.4, 1.0) as f64;
        let fov_y = rng.uniform(0.4, 1.0) as f64;
        Camera::new(
            glam::vec3(0.0, 0.0, -dist),
            glam::Quat::IDENTITY,
            fov_x,
            fov_y,
            glam::vec2(cx_n, cy_n),
            CameraModel::Pinhole,
        )
    };
    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_add(0xFEEB));
        let n = rng.usize_in(3, 8);
        (random_scene(seed, n), n, "offc".into())
    };

    let results = run_obscure_fuzz(&device, img_size, 20, scene_fn, cam_fn).await;
    assert_fuzz_clean(&results, 0.03, 5e-4, "off-center");
}

/// **Asymmetric image shapes** — non-square + small + extreme aspect.
/// Image-size handling code (tile counts, bbox clamping, pixel-rank
/// arithmetic) can have aspect-ratio-dependent off-by-ones that
/// square-image fuzz misses.
#[tokio::test]
async fn fuzz_obscure_asymmetric_images() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

    // Mix of small, tall, wide, big, tile-misaligned image sizes.
    let sizes = [
        glam::uvec2(15, 47),  // small, very tall, non-tile-aligned
        glam::uvec2(63, 16),  // wide, tile-aligned y
        glam::uvec2(17, 17),  // tiny, just over one tile each way
        glam::uvec2(81, 33),  // odd sizes
        glam::uvec2(128, 64), // 2:1 aspect, tile-aligned
        glam::uvec2(32, 95),  // unusual
    ];

    let mut all_results: Vec<FuzzRow> = Vec::new();
    for img_size in sizes {
        let scene_fn = |seed: u64| {
            let mut rng = Sm64::new(seed.wrapping_add(0xC0DE_F00D));
            let n = rng.usize_in(3, 7);
            (
                random_scene(seed, n),
                n,
                format!("{}x{}", img_size.x, img_size.y),
            )
        };
        let cam_fn = |seed: u64| {
            let mut rng = Sm64::new(seed.wrapping_mul(0xABCD));
            let fov = rng.uniform(0.5, 0.9) as f64;
            // Asymmetric fov to match the aspect of the image roughly.
            let fov_x = fov;
            let fov_y = fov * img_size.y as f64 / img_size.x as f64;
            Camera::new(
                glam::vec3(0.0, 0.0, -3.5),
                glam::Quat::IDENTITY,
                fov_x,
                fov_y,
                glam::vec2(0.5, 0.5),
                CameraModel::Pinhole,
            )
        };
        let results = run_obscure_fuzz(&device, img_size, 5, scene_fn, cam_fn).await;
        all_results.extend(results);
    }

    all_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    assert_fuzz_clean(&all_results, 0.03, 5e-4, "asym-img");
}

/// **Anisotropic extreme scales** — splats with one axis 100× the
/// others. Stresses the cov2d eigenstructure (high condition number
/// → conic inverse blows up off-axis). The `compensate_cov2d` 0.3
/// regularization should keep the conic finite; the gradient still
/// needs to be correct.
#[tokio::test]
async fn fuzz_obscure_anisotropic_scales() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(48, 48);

    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0xDEAD_F00D));
        let n = rng.usize_in(3, 8);
        let mut scene = random_scene(seed, n);
        // Override scales: one axis around exp(-1)≈0.37, others around
        // exp(-3)≈0.05 → ~7:1 anisotropy. Per splat, rotate the long
        // axis around the splat index so different splats stress
        // different orientations.
        for i in 0..n {
            let big = rng.uniform(-0.5, 0.5);
            let small_a = rng.uniform(-3.5, -2.5);
            let small_b = rng.uniform(-3.5, -2.5);
            let pattern = [
                [big, small_a, small_b],
                [small_a, big, small_b],
                [small_a, small_b, big],
            ];
            scene.log_scales[i * 3] = pattern[i % 3][0];
            scene.log_scales[i * 3 + 1] = pattern[i % 3][1];
            scene.log_scales[i * 3 + 2] = pattern[i % 3][2];
        }
        (scene, n, "aniso".into())
    };
    let cam_fn = random_camera;

    let results = run_obscure_fuzz(&device, img_size, 20, scene_fn, cam_fn).await;
    assert_fuzz_clean(&results, 0.03, 5e-4, "aniso");
}

/// **Depth extremes** — splats placed across a wide range, including
/// near the camera (z ≈ 0.1 after view transform, just above the
/// 0.01 near-plane cull) and very far (z ≈ 30). Stresses the
/// perspective Jacobian whose magnitude scales as 1/z.
/// **Near-camera fuzz.** Splats with cam-space depth in [0.05, 1.0]
/// — close enough that 1/z amplifies projection-Jacobian-derived
/// gradients substantially. Cycles all four camera models per seed
/// (Pinhole, KB4, RT8, `ThinPrismFisheye`) so the fisheye-specific
/// near-pole behavior gets stressed. The KB4 / TPF models use
/// `atan(r/z)` for the angular projection; numerical sensitivity at
/// small z is a known foot-gun and this fuzz watches for it.
#[tokio::test]
async fn fuzz_obscure_near_camera_splats() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(48, 48);

    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0xCAB0_05E0));
        let n = rng.usize_in(3, 7);
        let mut scene = random_scene(seed, n);
        // Camera sits at z=-3; world z = -2.9 → cam depth 0.1 (well
        // above the 0.01 near-plane). Spread splats across [0.1, 1.0]
        // so different `1/z` regimes get hit. World (x,y) is bounded
        // *proportional to cam-depth* so all splats stay well inside
        // the image (otherwise the Pinhole Jacobian's screen-space
        // clamp activates — a known piecewise-smooth discontinuity
        // that f-d trips on but isn't a kernel bug).
        for i in 0..n {
            let cam_z = 0.1 + rng.uniform(0.0, 0.9);
            scene.means[i * 3] = rng.uniform(-0.3, 0.3) * cam_z;
            scene.means[i * 3 + 1] = rng.uniform(-0.3, 0.3) * cam_z;
            scene.means[i * 3 + 2] = cam_z - 3.0;
        }
        // Keep splats small so projected size (∝ 1/z) doesn't cover
        // the whole image at z=0.1.
        for s in &mut scene.log_scales {
            *s -= 2.5;
        }
        let (_, label) = random_camera_with_model(seed);
        (scene, n, label.into())
    };
    let cam_fn = |seed: u64| {
        // Anchor camera at z=-3 with a fixed wide-ish fov; only the
        // model (and its params) vary by seed.
        let (cam, _label) = random_camera_with_model(seed);
        Camera::new(
            glam::vec3(0.0, 0.0, -3.0),
            glam::Quat::IDENTITY,
            0.8,
            0.8,
            glam::vec2(0.5, 0.5),
            cam.camera_model,
        )
    };
    let results = run_obscure_fuzz(&device, img_size, 20, scene_fn, cam_fn).await;
    assert_fuzz_clean(&results, 0.03, 5e-4, "near-cam");
}

#[tokio::test]
async fn fuzz_obscure_depth_extremes() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(48, 48);

    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0xC001_D00D));
        let n = rng.usize_in(4, 9);
        let mut scene = random_scene(seed, n);
        // Spread splats across a wide z range. Camera at z=-3,
        // identity rotation, so world z=0 → cam depth 3. World z =
        // -2.5 → cam depth 0.5 (close!), world z = 27 → cam depth 30.
        for i in 0..n {
            scene.means[i * 3] = rng.uniform(-0.4, 0.4);
            scene.means[i * 3 + 1] = rng.uniform(-0.4, 0.4);
            // Lerp z across the splat index, with jitter.
            let t = i as f32 / (n - 1).max(1) as f32;
            let z = -2.5 + t * 29.5 + rng.uniform(-0.3, 0.3);
            scene.means[i * 3 + 2] = z;
        }
        (scene, n, "depth".into())
    };
    let cam_fn = |_seed: u64| {
        Camera::new(
            glam::vec3(0.0, 0.0, -3.0),
            glam::Quat::IDENTITY,
            0.7,
            0.7,
            glam::vec2(0.5, 0.5),
            CameraModel::Pinhole,
        )
    };

    let results = run_obscure_fuzz(&device, img_size, 15, scene_fn, cam_fn).await;
    assert_fuzz_clean(&results, 0.03, 5e-4, "depth-ext");
}

/// **Heavy KB4 / RT8 distortion** — the camera-model fuzz earlier used
/// small k coefficients. Crank them up: KB4 with k1=±0.3, RT8 with
/// k1=±0.4. This exercises nonlinear paths in the projection Jacobian
/// that small-distortion fuzz may not.
#[tokio::test]
async fn fuzz_obscure_heavy_distortion() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(48, 48);

    let cam_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0xF15E_BEEF));
        let dist = rng.uniform(3.0, 5.0);
        let cam_pos = glam::vec3(0.0, 0.0, -dist);
        let fov = rng.uniform(0.5, 0.9) as f64;
        let center = glam::vec2(0.5, 0.5);
        // Cycle KB4 / RT8 / ThinPrismFisheye per seed; all with strong distortion.
        let model = match seed % 3 {
            0 => CameraModel::KannalaBrandt4(KannalaBrandt4Params {
                k1: rng.uniform(-0.3, 0.3),
                k2: rng.uniform(-0.15, 0.15),
                k3: rng.uniform(-0.05, 0.05),
                k4: rng.uniform(-0.02, 0.02),
            }),
            1 => CameraModel::RadialTangential8(RadialTangential8Params {
                k1: rng.uniform(-0.4, 0.4),
                k2: rng.uniform(-0.2, 0.2),
                k3: rng.uniform(-0.05, 0.05),
                k4: rng.uniform(-0.01, 0.01),
                k5: rng.uniform(-0.005, 0.005),
                k6: 0.0,
                p1: rng.uniform(-0.05, 0.05),
                p2: rng.uniform(-0.05, 0.05),
            }),
            _ => CameraModel::ThinPrismFisheye(ThinPrismFisheyeParams {
                kb4: KannalaBrandt4Params {
                    k1: rng.uniform(-0.2, 0.2),
                    k2: rng.uniform(-0.1, 0.1),
                    k3: rng.uniform(-0.02, 0.02),
                    k4: rng.uniform(-0.005, 0.005),
                },
                p1: rng.uniform(-0.03, 0.03),
                p2: rng.uniform(-0.03, 0.03),
                sx1: rng.uniform(-0.02, 0.02),
                sy1: rng.uniform(-0.02, 0.02),
            }),
        };
        Camera::new(cam_pos, glam::Quat::IDENTITY, fov, fov, center, model)
    };
    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0xB1B1));
        let n = rng.usize_in(3, 8);
        // Keep splats slightly tighter than default so they stay
        // inside the heavily-distorted FOV.
        let mut scene = random_scene(seed, n);
        for i in 0..n {
            scene.means[i * 3] = rng.uniform(-0.5, 0.5);
            scene.means[i * 3 + 1] = rng.uniform(-0.5, 0.5);
            scene.means[i * 3 + 2] = rng.uniform(-0.5, 0.5);
        }
        let tag = if seed.is_multiple_of(2) { "KB4" } else { "RT8" };
        (scene, n, tag.into())
    };

    let results = run_obscure_fuzz(&device, img_size, 20, scene_fn, cam_fn).await;
    assert_fuzz_clean(&results, 0.03, 5e-4, "heavy-dist");
}

/// **High SH degree (3) fuzz** — flagged a real backward bug
/// (`project_backwards.rs:166`): the SH VJP writes gradients to the
/// coefficient buffer but does NOT propagate `v_color → v_viewdir →
/// v_mean`. For SH degree > 0 the splat color depends on the
/// viewdir, which depends on the mean — that path is missing.
///
/// Marked `#[ignore]` until the missing gradient is added. The focused
/// tripwire test `finite_diff_means_through_high_sh_documents_bug`
/// passes today (asserting the disagreement exists) and will fail
/// when the bug is fixed, signaling that this fuzz should be unignored.
#[tokio::test]
async fn fuzz_obscure_sh_degree3() {
    use brush_render::sh::sh_coeffs_for_degree;
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(32, 32);
    let degree = 3_u32;
    let n_coeffs = sh_coeffs_for_degree(degree) as usize; // 16

    let scene_fn = |seed: u64| {
        let mut rng = Sm64::new(seed.wrapping_mul(0x5417_BABE));
        let n = rng.usize_in(3, 7);
        let base = random_scene(seed, n);
        // Expand DC-only sh into 16-coeff buffer per splat.
        let mut sh = vec![0.0_f32; n * n_coeffs * 3];
        for splat in 0..n {
            for ch in 0..3 {
                sh[splat * n_coeffs * 3 + ch] = base.sh_dc[splat * 3 + ch];
            }
            // Non-zero higher bands so backward has gradient to flow.
            for coef in 1..n_coeffs {
                for ch in 0..3 {
                    sh[(splat * n_coeffs + coef) * 3 + ch] = rng.uniform(-0.1, 0.1);
                }
            }
        }
        (Scene { sh_dc: sh, ..base }, n, "sh3".into())
    };
    let cam_fn = random_camera;

    // Skip ShDc lane: generic perturb assumes 3 floats/splat for sh_dc
    // which is wrong for degree 3 (48 floats/splat). SH coefficient
    // backward already covered by `finite_diff_sh_degree2`; this fuzz
    // exercises means/rots/log_scales/raw_opac under scenes carrying
    // populated higher-band SH.
    let results =
        run_obscure_fuzz_with(&device, img_size, 25, scene_fn, cam_fn, random_param_no_sh).await;
    assert_fuzz_clean(&results, 0.03, 5e-4, "sh3");
}

/// **Regression test for the SH-viewdir-through-means backward path.**
/// Originally documented a bug: `project_backwards.rs` called
/// `sh_coeffs_to_color_vjp` which wrote gradients to the SH coefficient
/// buffer but did NOT backpropagate `v_color → viewdir → mean`. For
/// SH degree ≥ 1, splat color depends on viewdir which depends on
/// mean. The missing path caused means gradients to disagree with
/// finite differences by 10-100% on viewdir-sensitive components.
///
/// Fixed by adding `sh_color_viewdir_vjp` (symbolic derivatives of
/// the SH basis polynomials w.r.t. viewdir) and chaining through the
/// `normalize(mean - cam_pos)` Jacobian. This test now asserts the
/// fix holds — kept around as a regression tripwire for this specific
/// 2-splat / degree-2 scene that originally exposed the bug.
#[tokio::test]
async fn finite_diff_means_through_high_sh_documents_bug() {
    use brush_render::sh::sh_coeffs_for_degree;

    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(48, 48);
    let degree = 2_u32;
    let n_coeffs = sh_coeffs_for_degree(degree) as usize; // 9

    // Two splats placed off-axis so viewdir is non-trivial and
    // perturbing means changes the SH-evaluated color noticeably.
    let n = 2;
    let mut sh = vec![0.0_f32; n * n_coeffs * 3];
    // Splat 0 at (0.4, -0.2, 0), splat 1 at (-0.3, 0.3, 0.2).
    let means = vec![0.4, -0.2, 0.0, -0.3, 0.3, 0.2];
    let rots = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let log_scales = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
    let raw_opac = vec![2.0, 2.0];
    // SH: DC = mid-gray, high bands deliberately large so viewdir
    // dependence is strong.
    for splat in 0..n {
        sh[splat * n_coeffs * 3] = 0.5; // DC R
        sh[splat * n_coeffs * 3 + 1] = 0.5; // DC G
        sh[splat * n_coeffs * 3 + 2] = 0.5; // DC B
        for coef in 1..n_coeffs {
            for ch in 0..3 {
                sh[(splat * n_coeffs + coef) * 3 + ch] = 0.4 * ((splat + coef + ch) as f32).sin();
            }
        }
    }
    let scene = Scene {
        means,
        rots,
        log_scales,
        sh_dc: sh,
        raw_opac,
    };

    let (splats, grads) = analytical_grads(&scene, &cam, img_size, &device).await;

    // The viewdir→mean path was fixed in `project_backwards.rs` (added
    // a `sh_color_viewdir_vjp` and chained through the normalize
    // Jacobian). Take best-of-sweep per component; tile-boundary noise
    // is per-eps, real residual bugs would float to the top.
    let sweep_eps = [3e-2_f32, 3e-3, 1e-3, 3e-4, 1e-4];
    let mut failed = Vec::new();
    for splat_idx in 0..n {
        for comp in 0..3 {
            let an = analytical_at(&splats, &grads, Lane::Mean, splat_idx, comp).await;
            let mut best_rel = f32::MAX;
            let mut best_num = 0.0;
            for eps in sweep_eps {
                let mut s_plus = scene.clone();
                perturb(&mut s_plus, Lane::Mean, splat_idx, comp, eps);
                let l_plus = render_value(&s_plus, &cam, img_size, &device).await;
                let mut s_minus = scene.clone();
                perturb(&mut s_minus, Lane::Mean, splat_idx, comp, -eps);
                let l_minus = render_value(&s_minus, &cam, img_size, &device).await;
                let num = (l_plus - l_minus) / (2.0 * eps);
                let scale = num.abs().max(an.abs()).max(1e-8);
                let rel = (num - an).abs() / scale;
                if rel < best_rel {
                    best_rel = rel;
                    best_num = num;
                }
            }
            // Combined tolerance to discount near-zero noise floor.
            let abs_err = (best_num - an).abs();
            let scale = best_num.abs().max(an.abs()).max(1e-8);
            let tol = 5e-4 + 0.05 * scale;
            if abs_err > tol {
                failed.push(format!(
                    "means[{splat_idx},{comp}]: num {best_num:.6} vs an {an:.6} (|Δ|={abs_err:.3e} > {tol:.3e})"
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "viewdir→mean backward path regressed:\n  {}",
        failed.join("\n  "),
    );
}

/// **Kitchen sink**: every axis randomized at once (rotated camera +
/// off-center principal + asymmetric focal + non-square image + KB4 or
/// RT8 + anisotropic scales + mid-range depths). If any pairwise
/// interaction bug exists this is most likely to surface it.
#[tokio::test]
async fn fuzz_obscure_kitchen_sink() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();

    let mut all_results: Vec<FuzzRow> = Vec::new();
    let img_choices = [
        glam::uvec2(32, 32),
        glam::uvec2(48, 33),
        glam::uvec2(17, 65),
    ];
    for img_idx in 0..3 {
        let img_size = img_choices[img_idx];
        let scene_fn = |seed: u64| {
            let mut rng = Sm64::new(seed.wrapping_mul(0xC11A_BAD0));
            let n = rng.usize_in(3, 8);
            let mut scene = random_scene(seed, n);
            // Anisotropic scales
            for i in 0..n {
                let axis = i % 3;
                let big = rng.uniform(-1.0, 0.0);
                let small = rng.uniform(-2.5, -1.5);
                for c in 0..3 {
                    scene.log_scales[i * 3 + c] = if c == axis { big } else { small };
                }
            }
            (scene, n, format!("img{img_idx}"))
        };
        let cam_fn = |seed: u64| {
            let mut rng = Sm64::new(seed.wrapping_mul(0x9999_AAAA) ^ 0x55);
            let ax = rng.uniform(-1.0, 1.0);
            let ay = rng.uniform(-1.0, 1.0);
            let az = rng.uniform(-1.0, 1.0);
            let axis = glam::Vec3::new(ax, ay, az).normalize_or_zero();
            let angle = rng.uniform(-0.6, 0.6); // ±34°
            let rot = if axis == glam::Vec3::ZERO {
                glam::Quat::IDENTITY
            } else {
                glam::Quat::from_axis_angle(axis, angle)
            };
            let dist = rng.uniform(3.0, 5.0);
            let cam_pos = rot * glam::vec3(0.0, 0.0, -dist);
            let fov_x = rng.uniform(0.5, 0.9) as f64;
            let fov_y = rng.uniform(0.5, 0.9) as f64;
            let cx = rng.uniform(0.3, 0.7);
            let cy = rng.uniform(0.3, 0.7);
            let center = glam::vec2(cx, cy);
            // Cycle through camera models including ThinPrismFisheye.
            let model = match seed % 4 {
                0 => CameraModel::Pinhole,
                1 => CameraModel::KannalaBrandt4(KannalaBrandt4Params {
                    k1: rng.uniform(-0.05, 0.05),
                    k2: rng.uniform(-0.02, 0.02),
                    k3: rng.uniform(-0.005, 0.005),
                    k4: rng.uniform(-0.001, 0.001),
                }),
                2 => CameraModel::RadialTangential8(RadialTangential8Params {
                    k1: rng.uniform(-0.1, 0.1),
                    k2: rng.uniform(-0.03, 0.03),
                    k3: 0.0,
                    k4: 0.0,
                    k5: 0.0,
                    k6: 0.0,
                    p1: rng.uniform(-0.01, 0.01),
                    p2: rng.uniform(-0.01, 0.01),
                }),
                _ => CameraModel::ThinPrismFisheye(ThinPrismFisheyeParams {
                    kb4: KannalaBrandt4Params {
                        k1: rng.uniform(-0.05, 0.05),
                        k2: rng.uniform(-0.02, 0.02),
                        k3: rng.uniform(-0.005, 0.005),
                        k4: rng.uniform(-0.001, 0.001),
                    },
                    p1: rng.uniform(-0.01, 0.01),
                    p2: rng.uniform(-0.01, 0.01),
                    sx1: rng.uniform(-0.005, 0.005),
                    sy1: rng.uniform(-0.005, 0.005),
                }),
            };
            Camera::new(cam_pos, rot, fov_x, fov_y, center, model)
        };
        let results = run_obscure_fuzz(&device, img_size, 10, scene_fn, cam_fn).await;
        all_results.extend(results);
    }
    all_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    assert_fuzz_clean(&all_results, 0.03, 5e-4, "kitchen-sink");
}

/// Forward determinism check: same scene, render twice, the loss must
/// match bit-for-bit. If this fails, finite-diff comparisons against
/// analytical grads are noise-floored by the forward itself.
#[tokio::test]
async fn forward_loss_is_deterministic() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(32, 32);
    let scene = base_scene();

    let l_a = render_value(&scene, &cam, img_size, &device).await;
    let l_b = render_value(&scene, &cam, img_size, &device).await;
    let l_c = render_value(&scene, &cam, img_size, &device).await;

    assert_eq!(
        l_a, l_b,
        "forward loss is nondeterministic ({l_a} vs {l_b})"
    );
    assert_eq!(
        l_a, l_c,
        "forward loss is nondeterministic ({l_a} vs {l_c})"
    );
}
