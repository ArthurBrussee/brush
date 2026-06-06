//! RaDe-GS geometry helpers.
//!
//! The renderer emits, per pixel, an alpha-blended view-space normal `N` and
//! the alpha-weighted ray-plane depth (channels 4..8 of the geometry render).
//! Per-pixel surface depth is `depth_raw / alpha` (radial distance along the
//! pixel ray). These helpers are pure burn tensor ops so they compose with
//! autodiff (depth/normal losses) and also back the viewer's depth/normal
//! channels. The viewer-side colormap + RGBA8 packing lives in the app crate
//! (`ui::geo_view`).

use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Int, Tensor, s};
use glam::UVec2;

use crate::camera::Camera;

/// Channel count of the geometry render: `rgba` + `Nx,Ny,Nz,depth` + `zz`
/// (`zz = Sum(w*z^2)`, the second depth moment for the distortion loss).
pub const GEO_CHANNELS: usize = 9;

/// Camera-space ray directions (`z = 1`) for each pixel center, `[H, W, 3]`.
/// Pinhole approximation — ignores lens distortion, which is fine here.
pub fn camera_ray_grid(geo: &Tensor<3>, camera: &Camera, img_size: UVec2) -> Tensor<3> {
    let device = geo.device();
    let [h, w, _] = geo.dims();
    let pin = camera.build_pinhole_params(img_size);

    let xs = Tensor::<1, Int>::arange(0..w as i64, &device)
        .float()
        .add_scalar(0.5)
        .sub_scalar(pin.cx)
        .div_scalar(pin.fx)
        .reshape([1, w, 1])
        .expand([h, w, 1]);
    let ys = Tensor::<1, Int>::arange(0..h as i64, &device)
        .float()
        .add_scalar(0.5)
        .sub_scalar(pin.cy)
        .div_scalar(pin.fy)
        .reshape([h, 1, 1])
        .expand([h, w, 1]);
    let zs = Tensor::ones([h, w, 1], &device);
    Tensor::cat(vec![xs, ys, zs], 2)
}

// All helpers below take the 4-channel *geometry* tensor `[H, W, 4]` =
// `[Nx, Ny, Nz, depth_raw]` (the blended view-space normal + alpha-weighted
// ray-plane depth), not the full rgba+geo render. Callers slice it out
// (`out_img[.., .., 4..8]`) and pass coverage alpha separately.

/// Per-pixel radial depth `depth_raw / alpha`, `[H, W, 1]` (RaDe-GS). `alpha`
/// is the coverage (channel 3 of the full render); empty pixels divide by a
/// clamped alpha → ~0 and should be masked by alpha downstream.
pub fn rendered_depth(geo: Tensor<3>, alpha: Tensor<3>) -> Tensor<3> {
    let depth_raw = geo.slice(s![.., .., 3..4]);
    depth_raw / alpha.clamp_min(1e-6)
}

/// Rendered unit normal map `[H, W, 3]` from the geometry tensor.
pub fn rendered_normal(geo: Tensor<3>) -> Tensor<3> {
    normalize3(geo.slice(s![.., .., 0..3]))
}

fn normalize3(v: Tensor<3>) -> Tensor<3> {
    // Epsilon *inside* the sqrt so the gradient stays finite for near-zero
    // vectors (blended normal ≈ 0 at uncovered pixels). `sqrt'(0) = ∞`,
    // which would otherwise NaN the rotation gradients.
    let len = (v.clone().powf_scalar(2.0).sum_dim(2) + 1e-8).sqrt();
    v / len
}

/// Per-pixel cross product of two `[H, W, 3]` vector fields.
fn cross3(a: Tensor<3>, b: Tensor<3>) -> Tensor<3> {
    let ax = a.clone().slice(s![.., .., 0..1]);
    let ay = a.clone().slice(s![.., .., 1..2]);
    let az = a.slice(s![.., .., 2..3]);
    let bx = b.clone().slice(s![.., .., 0..1]);
    let by = b.clone().slice(s![.., .., 1..2]);
    let bz = b.slice(s![.., .., 2..3]);
    let cx = ay.clone() * bz.clone() - az.clone() * by.clone();
    let cy = az * bx.clone() - ax.clone() * bz;
    let cz = ax * by - ay * bx;
    Tensor::cat(vec![cx, cy, cz], 2)
}

/// Elementwise `max(a, b)` without a dedicated op: `(a + b + |a-b|) / 2`.
fn elem_max(a: Tensor<3>, b: Tensor<3>) -> Tensor<3> {
    (a.clone() + b.clone() + (a - b).abs()) * 0.5
}

/// PGSR image-edge weight (`get_img_grad_weight`) over the interior grid
/// `[H-2, W-2, 1]`. Low at color edges (where the local-planarity assumption
/// fails and the finite-difference normal is unreliable), high in smooth
/// regions: `(1 - normalized_grad).clamp(0,1)²`, with the gradient the
/// per-pixel max of the central x/y abs differences, min-max normalized.
/// Should be fed a detached (constant) image.
fn image_edge_weight(image: Tensor<3>) -> Tensor<3> {
    let [h, w, _] = image.dims();
    // Central differences over the interior, matching PGSR.
    let right = image.clone().slice(s![1..h - 1, 2..w, ..]);
    let left = image.clone().slice(s![1..h - 1, 0..w - 2, ..]);
    let bottom = image.clone().slice(s![2..h, 1..w - 1, ..]);
    let top = image.slice(s![0..h - 2, 1..w - 1, ..]);
    let gx = (right - left).abs().mean_dim(2);
    let gy = (top - bottom).abs().mean_dim(2);
    let g = elem_max(gx, gy);
    let gmin = g.clone().min().reshape([1, 1, 1]);
    let gmax = g.clone().max().reshape([1, 1, 1]);
    let gn = (g - gmin.clone()) / (gmax - gmin + 1e-8);
    (-gn + 1.0).clamp(0.0, 1.0).powf_scalar(2.0)
}

/// Single-view depth↔normal consistency (PGSR Eq. 6). Compares the *raw*
/// α-weighted blended normal (exactly as PGSR's rasterizer emits it — NOT
/// renormalized, magnitude ≈ accumulated alpha) against the unit normal from
/// the depth point map scaled by the rendered alpha: `‖N_d·α − N_blend‖₁`,
/// summed over channels and multiplied by the image-edge weight. The depth
/// normal uses PGSR's central-difference stencil (`(right−left) × (top−bottom)`,
/// `depth_pcd2normal`) and is oriented camera-facing. The α magnitude is the
/// implicit coverage mask (empty pixels → both terms → 0). `α` is detached, à
/// la PGSR. Result is `[H-2, W-2, 1]` (the interior).
pub fn depth_normal_consistency(
    geo: Tensor<3>,
    alpha: Tensor<3>,
    image: Tensor<3>,
    camera: &Camera,
    img_size: UVec2,
) -> Tensor<3> {
    let [h, w, _] = geo.dims();
    // Raw blended normal (channels 0..3), NOT normalized — magnitude encodes
    // the per-pixel normal agreement, which the loss also drives.
    let n_blend = geo.clone().slice(s![.., .., 0..3]);
    let ray = camera_ray_grid(&geo, camera, img_size);
    let depth = rendered_depth(geo, alpha.clone());
    // Camera-space surface points: radial depth times the *unit* ray.
    let points = normalize3(ray.clone()) * depth;

    // Central differences over the interior, matching PGSR's depth_pcd2normal.
    let right = points.clone().slice(s![1..h - 1, 2..w, ..]);
    let left = points.clone().slice(s![1..h - 1, 0..w - 2, ..]);
    let bottom = points.clone().slice(s![2..h, 1..w - 1, ..]);
    let top = points.slice(s![0..h - 2, 1..w - 1, ..]);
    let n_fd = normalize3(cross3(right - left, top - bottom));

    let ray_c = ray.slice(s![1..h - 1, 1..w - 1, ..]);
    // Orient N_d to face the camera (N_d·ray < 0), like the rendered normal.
    let orient = -(n_fd.clone() * ray_c).sum_dim(2).sign();
    let n_fd = n_fd * orient;

    // depth_normal = unit normal · rendered_alpha (detached), à la PGSR.
    let alpha_c = alpha.slice(s![1..h - 1, 1..w - 1, ..]).detach();
    let n_r = n_blend.slice(s![1..h - 1, 1..w - 1, ..]);
    let l1 = (n_fd * alpha_c - n_r).abs().sum_dim(2);
    l1 * image_edge_weight(image)
}

/// Depth-distortion loss (the squared / weighted-variance form of 2DGS `L_d`):
/// per pixel `M0·M2 − M1²` with `M0 = Σw = alpha`, `M1 = Σw·z` (the raw depth
/// channel), `M2 = Σw·z²` (the `zz` channel). This equals ½`Σᵢⱼ wᵢwⱼ(zᵢ−zⱼ)²`,
/// so minimizing it concentrates each ray's weight onto a single depth. All
/// three moments are differentiable render outputs, so it composes with autodiff
/// (no custom loss backward). `geo` is the geometry slice `[H, W, 5]`
/// (`N, depth, zz`); `alpha` is coverage. Result `[H, W, 1]`, clamped ≥ 0.
pub fn depth_distortion(geo: Tensor<3>, alpha: Tensor<3>) -> Tensor<3> {
    let m1 = geo.clone().slice(s![.., .., 3..4]);
    let m2 = geo.slice(s![.., .., 4..5]);
    (alpha * m2 - m1.clone() * m1).clamp_min(0.0)
}

/// Nearest-neighbour resize of a `[Hd, Wd, 1]` map to `[h, w, 1]`. Nearest
/// avoids fabricating depth across discontinuities when upsampling.
fn resize_nearest(x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
    let [hd, wd, _] = x.dims();
    let out = interpolate(
        x.reshape([1, 1, hd, wd]),
        [h, w],
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    out.reshape([h, w, 1])
}

/// Metric depth supervision: L1 between the RaDe-GS rendered depth and the
/// per-view `LiDAR` depth. The GT is `z`-depth (perpendicular) in metres at its
/// native low resolution; it is nearest-upsampled to the render size and
/// converted to radial (`radial = z·‖ray‖`) to match `rendered_depth`. Each
/// pixel is soft-weighted by its `[0, 1]` confidence (no hard threshold), gated
/// by coverage (`alpha > 0.5`) and a positive GT. Returns `(|Δ|·w, w)` so the
/// caller forms the confidence-weighted mean `sum(weighted) / sum(w)`.
pub fn depth_l1_loss(
    geo: Tensor<3>,
    alpha: Tensor<3>,
    gt_z: Tensor<3>,
    gt_conf: Tensor<3>,
    camera: &Camera,
    img_size: UVec2,
) -> (Tensor<3>, Tensor<3>) {
    let [h, w, _] = geo.dims();
    let gt_z = resize_nearest(gt_z.detach(), h, w);
    let gt_conf = resize_nearest(gt_conf.detach(), h, w);

    // GT z-depth -> radial, using the per-pixel ray length (matches the
    // radial convention of `rendered_depth`).
    let ray_len = camera_ray_grid(&geo, camera, img_size)
        .powf_scalar(2.0)
        .sum_dim(2)
        .sqrt();
    let gt_radial = (gt_z.clone() * ray_len).detach();

    // Confidence (already in [0, 1]) is the per-pixel weight, gated by coverage
    // and a positive GT.
    let weight = (gt_conf.clamp(0.0, 1.0)
        * alpha.clone().greater_elem(0.5).float()
        * gt_z.greater_elem(0.0).float())
    .detach();

    let rendered = rendered_depth(geo, alpha);
    ((rendered - gt_radial).abs() * weight.clone(), weight)
}
