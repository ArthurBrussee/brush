//! Input-space fuzzing for the render pipeline.
//!
//! Feeds deliberately-broken scenes (NaN, Inf, denormals, extreme
//! magnitudes, zero-length quats, ...) to the render and backward passes,
//! then lets `RenderOutput::validate` and `validate_gradient` enforce the
//! pipeline invariants. The bar is "no silent corruption": either the
//! splat renders cleanly or validation panics with a precise message.
//!
//! Forward tests call `MainBackendBase::render` directly to skip
//! `Splats::validate_values`'s up-front NaN panic; backward tests go
//! through the normal autodiff `render_splats` which asserts inputs are
//! finite (backward-pass bugs reveal themselves in gradient NaN/Inf).
//!
//! Forward tests run on `MainBackendBase` (`bwd_info = false`) so the
//! full `RenderOutput::validate` suite of invariants fires. Backward
//! tests run on `Autodiff<MainBackend>` (via fusion), exercising the
//! gradient kernels.

use brush_render::{
    MainBackendBase, RenderOutput, SplatOps,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    shaders::helpers::TILE_WIDTH,
};
use burn::backend::{Autodiff, wgpu::WgpuDevice};
use burn::tensor::{DType, Tensor, TensorData, TensorMetadata, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use burn_wgpu::WgpuRuntime;
use std::num::Wrapping;

type DiffBackend = Autodiff<brush_render::MainBackend>;

// SplitMix64 — deterministic, no test-only dep.
struct Sm64(Wrapping<u64>);

impl Sm64 {
    fn new(seed: u64) -> Self {
        Self(Wrapping(seed))
    }
    fn u64(&mut self) -> u64 {
        self.0 += Wrapping(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f01(&mut self) -> f32 {
        (self.u64() as f64 / u64::MAX as f64) as f32
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f01() * (hi - lo)
    }
    fn choice<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.u64() as usize % items.len()]
    }
    fn usize_in(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.u64() as usize % (hi - lo))
    }
}

/// One f32 slot sampled here can blow up the pipeline in a specific way.
const POISON_VALUES: &[f32] = &[
    f32::NAN,
    -f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0,
    -0.0,
    f32::MIN_POSITIVE,
    f32::MIN_POSITIVE / 2.0, // denormal
    1e-40,                   // denormal
    f32::EPSILON,
    1e38,
    -1e38,
    f32::MAX,
    -f32::MAX,
    1e20,
    -1e20,
    1.0,
    -1.0,
    // Right on the project_forward thresholds.
    0.01,
    1e10,
    1.0 / 255.0,
    // Tile boundary in pixel space.
    16.0,
];

type Scene = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn poisoned_scene(seed: u64, n: usize, poison_rate: f32) -> Scene {
    let mut rng = Sm64::new(seed);
    let mut pick = |lo: f32, hi: f32| -> f32 {
        let fallback = rng.uniform(lo, hi);
        if rng.f01() < poison_rate {
            rng.choice(POISON_VALUES)
        } else {
            fallback
        }
    };
    let mut means = Vec::with_capacity(n * 3);
    let mut rots = Vec::with_capacity(n * 4);
    let mut ls = Vec::with_capacity(n * 3);
    let mut dc = Vec::with_capacity(n * 3);
    let mut opac = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..3 {
            means.push(pick(-3.0, 3.0));
        }
        for _ in 0..4 {
            rots.push(pick(-1.0, 1.0));
        }
        for _ in 0..3 {
            ls.push(pick(-4.0, 2.0));
        }
        for _ in 0..3 {
            dc.push(pick(0.0, 1.0));
        }
        opac.push(pick(-2.0, 2.0));
    }
    (means, rots, ls, dc, opac)
}

/// Finite-only scene — random values within sane training ranges. Used
/// for backward-pass fuzzing where inputs must survive `validate_values`.
fn finite_scene(seed: u64, n: usize) -> Scene {
    let mut rng = Sm64::new(seed);
    let mut means = Vec::with_capacity(n * 3);
    let mut rots = Vec::with_capacity(n * 4);
    let mut ls = Vec::with_capacity(n * 3);
    let mut dc = Vec::with_capacity(n * 3);
    let mut opac = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..3 {
            means.push(rng.uniform(-3.0, 3.0));
        }
        for _ in 0..4 {
            rots.push(rng.uniform(-1.0, 1.0));
        }
        for _ in 0..3 {
            ls.push(rng.uniform(-4.0, 2.0));
        }
        for _ in 0..3 {
            dc.push(rng.uniform(0.2, 0.8));
        }
        opac.push(rng.uniform(0.0, 3.0));
    }
    (means, rots, ls, dc, opac)
}

fn cube_tensor<const D: usize>(
    device: &WgpuDevice,
    shape: [usize; D],
    data: &[f32],
) -> CubeTensor<WgpuRuntime> {
    let t: Tensor<MainBackendBase, D> =
        Tensor::from_data(TensorData::new(data.to_vec(), shape), device);
    let prim = t.into_primitive().tensor();
    assert_eq!(prim.dtype(), DType::F32, "cube_tensor: expected F32");
    prim
}

async fn render_raw(
    device: &WgpuDevice,
    camera: &Camera,
    img_size: glam::UVec2,
    scene: &Scene,
    mode: SplatRenderMode,
) -> RenderOutput<MainBackendBase> {
    let (means, rots, ls, dc, opac) = scene;
    let n = opac.len();
    let mut transforms = Vec::with_capacity(n * 10);
    for i in 0..n {
        transforms.extend_from_slice(&means[i * 3..i * 3 + 3]);
        transforms.extend_from_slice(&rots[i * 4..i * 4 + 4]);
        transforms.extend_from_slice(&ls[i * 3..i * 3 + 3]);
    }
    MainBackendBase::render(
        camera,
        img_size,
        cube_tensor(device, [n, 10], &transforms),
        cube_tensor(device, [n, 1, 3], dc),
        cube_tensor(device, [n, 0, 3], &[]),
        cube_tensor(device, [n], opac),
        mode,
        glam::Vec3::ZERO,
        false,
    )
    .await
}

fn std_cam() -> Camera {
    Camera::new(
        glam::vec3(0.0, 0.0, -3.0),
        glam::Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    )
}

fn rand_cam(rng: &mut Sm64) -> Camera {
    let pos = glam::vec3(
        rng.uniform(-10.0, 10.0),
        rng.uniform(-10.0, 10.0),
        rng.uniform(-30.0, -0.1),
    );
    let axis = glam::vec3(
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
    )
    .normalize_or_zero();
    let angle = rng.uniform(0.0, std::f32::consts::TAU);
    let rot = if axis == glam::Vec3::ZERO {
        glam::Quat::IDENTITY
    } else {
        glam::Quat::from_axis_angle(axis, angle)
    };
    let fov = rng.uniform(0.3, 1.2) as f64;
    Camera::new(pos, rot, fov, fov, glam::vec2(0.5, 0.5))
}

fn rand_img_size(rng: &mut Sm64) -> glam::UVec2 {
    const TABLE: &[glam::UVec2] = &[
        glam::uvec2(1, 1),
        glam::uvec2(1, 17),
        glam::uvec2(17, 1),
        glam::uvec2(15, 15),
        glam::uvec2(16, 16),
        glam::uvec2(17, 17),
        glam::uvec2(33, 47),
        glam::uvec2(64, 64),
        glam::uvec2(97, 129),
        glam::uvec2(128, 128),
        glam::uvec2(257, 257),
    ];
    TABLE[(rng.u64() as usize) % TABLE.len()]
}

fn assert_basic_counts(
    out: &RenderOutput<MainBackendBase>,
    n: usize,
    img_size: glam::UVec2,
    tag: &str,
) {
    assert!(
        out.aux.num_visible <= n as u32,
        "{tag}: num_visible {} > n {n}",
        out.aux.num_visible,
    );
    let tx = img_size.x.div_ceil(TILE_WIDTH) as u64;
    let ty = img_size.y.div_ceil(TILE_WIDTH) as u64;
    let max_isects = (out.aux.num_visible as u64) * tx * ty;
    assert!(
        (out.aux.num_intersections as u64) <= max_isects,
        "{tag}: num_intersections {} > {max_isects}",
        out.aux.num_intersections,
    );
}

/// Every (slot, poison) combination, one bad slot per scene.
#[tokio::test]
async fn fuzz_single_bad_slot_combinations() {
    let device = brush_kernel::test_helpers::test_device().await;
    let img_size = glam::uvec2(48, 48);
    for slot in 0..14 {
        for &poison in POISON_VALUES {
            let n = 120;
            let mut scene: Scene = (
                (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
                (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
                vec![-1.0; n * 3],
                vec![0.5; n * 3],
                vec![2.0; n],
            );
            match slot {
                0..=2 => scene.0[slot] = poison,
                3..=6 => scene.1[slot - 3] = poison,
                7..=9 => scene.2[slot - 7] = poison,
                10..=12 => scene.3[slot - 10] = poison,
                13 => scene.4[0] = poison,
                _ => unreachable!(),
            }
            let out = render_raw(
                &device,
                &std_cam(),
                img_size,
                &scene,
                SplatRenderMode::Default,
            )
            .await;
            assert_basic_counts(&out, n, img_size, &format!("slot={slot} poison={poison:?}"));
            out.validate().await;
        }
    }
}

/// Randomized stress: scene, camera, image size, poison rate, mode.
#[tokio::test]
async fn fuzz_random_scenes() {
    let device = brush_kernel::test_helpers::test_device().await;
    for seed in 0..100u64 {
        let mut rng = Sm64::new(seed.wrapping_mul(0xA5A5_CAFE));
        let n = rng.usize_in(1, 256);
        let poison_rate = rng.uniform(0.0, 0.95);
        let img_size = rand_img_size(&mut rng);
        let cam = rand_cam(&mut rng);
        let mode = if rng.f01() < 0.3 {
            SplatRenderMode::Mip
        } else {
            SplatRenderMode::Default
        };
        let scene = poisoned_scene(seed, n, poison_rate);
        let out = render_raw(&device, &cam, img_size, &scene, mode).await;
        let tag = format!("seed={seed} n={n} rate={poison_rate:.2} img={img_size:?} mode={mode:?}");
        assert_basic_counts(&out, n, img_size, &tag);
        out.validate().await;
    }
}

type BadGeomCase = fn(&mut Sm64, usize) -> Scene;

/// Every splat's geometry is deliberately bad in one of the canonical ways.
/// PF must cull them all (`num_visible == 0`).
#[tokio::test]
async fn fuzz_bad_geometry_is_fully_culled() {
    let device = brush_kernel::test_helpers::test_device().await;
    let img_size = glam::uvec2(64, 64);
    let n = 16;
    let cam = std_cam();

    let cases: &[(&str, BadGeomCase)] = &[
        ("nan_positions", |_, n| {
            (
                vec![f32::NAN; n * 3],
                (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
                vec![0.0; n * 3],
                vec![0.5; n * 3],
                vec![0.0; n],
            )
        }),
        ("inf_positions", |_, n| {
            (
                (0..n * 3)
                    .map(|i| {
                        if i % 2 == 0 {
                            f32::INFINITY
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect(),
                (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
                vec![0.0; n * 3],
                vec![0.5; n * 3],
                vec![0.0; n],
            )
        }),
        ("nan_quats", |rng, n| {
            let mut rots = Vec::with_capacity(n * 4);
            for _ in 0..n {
                let bad = rng.usize_in(0, 4);
                for s in 0..4 {
                    rots.push(if s == bad {
                        f32::NAN
                    } else {
                        rng.uniform(-1.0, 1.0)
                    });
                }
            }
            (
                (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
                rots,
                vec![0.0; n * 3],
                vec![0.5; n * 3],
                vec![0.0; n],
            )
        }),
        ("zero_quats", |_, n| {
            (
                (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
                vec![0.0; n * 4],
                vec![0.0; n * 3],
                vec![0.5; n * 3],
                vec![0.0; n],
            )
        }),
        ("nan_scales", |rng, n| {
            let mut ls = Vec::with_capacity(n * 3);
            for _ in 0..n {
                let bad = rng.usize_in(0, 3);
                for s in 0..3 {
                    ls.push(if s == bad {
                        f32::NAN
                    } else {
                        rng.uniform(-4.0, 4.0)
                    });
                }
            }
            (
                (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
                (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
                ls,
                vec![0.5; n * 3],
                vec![0.0; n],
            )
        }),
        ("inf_scales", |_, n| {
            (
                (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
                (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
                vec![120.0; n * 3],
                vec![0.5; n * 3],
                vec![0.0; n],
            )
        }),
        ("nan_opac", |_, n| {
            (
                (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
                (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
                vec![0.0; n * 3],
                vec![0.5; n * 3],
                vec![f32::NAN; n],
            )
        }),
    ];

    for (tag, make) in cases {
        let mut rng = Sm64::new(0xDEAD_BEEF);
        let scene = make(&mut rng, n);
        let out = render_raw(&device, &cam, img_size, &scene, SplatRenderMode::Default).await;
        assert_eq!(
            out.aux.num_visible, 0,
            "{tag}: {} visible, expected 0",
            out.aux.num_visible
        );
        assert_eq!(out.aux.num_intersections, 0);
        out.validate().await;
    }
}

/// Valid-but-extreme inputs must NOT be culled. Guards against over-eager
/// filtering — a huge `log_scale` or big finite color is a legitimate
/// training state and the pipeline should render it.
#[tokio::test]
async fn fuzz_valid_but_extreme_stays_visible() {
    let device = brush_kernel::test_helpers::test_device().await;
    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);
    let n = 4;

    // 40 is about the largest log_scale before scale*scale overflows f32.
    for &ls_val in &[-30.0_f32, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0] {
        let scene: Scene = (
            (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
            (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
            (0..n).flat_map(|_| [ls_val, ls_val, ls_val]).collect(),
            vec![0.5; n * 3],
            vec![2.0; n],
        );
        let out = render_raw(&device, &cam, img_size, &scene, SplatRenderMode::Default).await;
        assert_eq!(
            out.aux.num_visible, n as u32,
            "log_scale={ls_val} over-culled"
        );
        out.validate().await;
    }

    for &mag in &[1e10_f32, 1e25, f32::MAX / 2.0] {
        let scene: Scene = (
            (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect(),
            (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
            vec![-1.0; n * 3],
            (0..n).flat_map(|_| [mag, -mag, mag]).collect(),
            vec![3.0; n],
        );
        let out = render_raw(&device, &cam, img_size, &scene, SplatRenderMode::Default).await;
        assert_eq!(out.aux.num_visible, n as u32, "color mag={mag} over-culled");
        out.validate().await;
    }
}

/// Run the full autodiff forward + backward on finite random scenes and
/// confirm every gradient is finite. Exercises the diff kernel path that
/// the forward-only fuzz never touches (`project_backwards` and
/// `rasterize_backwards`).
#[tokio::test]
async fn fuzz_bwd_random_scenes_gradients_are_finite() {
    let device = brush_kernel::test_helpers::test_device().await;
    for seed in 0..100u64 {
        let mut rng = Sm64::new(0xBDBD_BDBD ^ seed.wrapping_mul(0xA5A5_CAFE));
        let n = rng.usize_in(4, 256);
        let img_size = glam::uvec2(rng.usize_in(16, 128) as u32, rng.usize_in(16, 128) as u32);
        let cam = rand_cam(&mut rng);
        let mode = if rng.f01() < 0.3 {
            SplatRenderMode::Mip
        } else {
            SplatRenderMode::Default
        };
        let (means, rots, ls, dc, opac) = finite_scene(seed, n);

        let splats: Splats<DiffBackend> =
            Splats::from_raw(means, rots, ls, dc, opac, mode, &device);
        let diff =
            brush_render_bwd::render_splats(splats.clone(), &cam, img_size, glam::Vec3::ZERO).await;
        let img: Tensor<DiffBackend, 3> = Tensor::from_primitive(TensorPrimitive::Float(diff.img));
        let loss = img.mean();
        let grads = loss.backward();
        splats.validate_grads(&grads).await;
    }
}

/// Extreme-but-finite scenes through the backward pass. `log_scale` up
/// to ~40 is the practical training upper bound; past that cov2d
/// overflows and the forward culls. Colors up to `f32::MAX` are OK
/// because `project_visible` clamps SH-derived color to ±100 before the
/// f16 cast, which keeps the rasterize backward chain finite.
#[tokio::test]
async fn fuzz_bwd_extreme_inputs_stay_finite() {
    let device = brush_kernel::test_helpers::test_device().await;
    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);
    let n = 8;

    for &ls_val in &[-20.0_f32, -5.0, 0.0, 5.0, 15.0, 30.0, 40.0] {
        for &mag in &[0.1_f32, 10.0, 1e6, f32::MAX / 2.0] {
            let means: Vec<f32> = (0..n).flat_map(|_| [0.0, 0.0, 3.0]).collect();
            let rots: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
            let ls: Vec<f32> = (0..n).flat_map(|_| [ls_val, ls_val, ls_val]).collect();
            let dc: Vec<f32> = (0..n).flat_map(|_| [mag, -mag, mag]).collect();
            let opac = vec![2.0f32; n];

            let splats: Splats<DiffBackend> =
                Splats::from_raw(means, rots, ls, dc, opac, SplatRenderMode::Default, &device);
            let diff =
                brush_render_bwd::render_splats(splats.clone(), &cam, img_size, glam::Vec3::ZERO)
                    .await;
            let img: Tensor<DiffBackend, 3> =
                Tensor::from_primitive(TensorPrimitive::Float(diff.img));
            let loss = img.mean();
            let grads = loss.backward();
            splats.validate_grads(&grads).await;
        }
    }
}
