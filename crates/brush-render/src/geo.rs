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

/// Geometry render channels: `rgba` + `Nx,Ny,Nz,depth` + normalized GOF
/// distortion + its two mapped-depth moments (backward-only).
pub const GEO_CHANNELS: usize = 11;

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

/// Single-view depth-normal consistency (GOF): cosine error `1 − ⟨N_render,
/// N_depth⟩` against the central-difference normal of the unprojected depth
/// map, camera-facing oriented. No edge weight or coverage mask, matching
/// GOF. Result is the `[H-2, W-2, 1]` interior.
pub fn depth_normal_consistency(
    geo: Tensor<3>,
    alpha: Tensor<3>,
    camera: &Camera,
    img_size: UVec2,
) -> Tensor<3> {
    let [h, w, _] = geo.dims();
    let n_render = rendered_normal(geo.clone());
    let ray = camera_ray_grid(&geo, camera, img_size);
    let depth = rendered_depth(geo, alpha);
    // Camera-space surface points: radial depth times the *unit* ray.
    let points = normalize3(ray.clone()) * depth;

    // Central differences over the interior (GOF/PGSR `depth_to_normal`).
    let right = points.clone().slice(s![1..h - 1, 2..w, ..]);
    let left = points.clone().slice(s![1..h - 1, 0..w - 2, ..]);
    let bottom = points.clone().slice(s![2..h, 1..w - 1, ..]);
    let top = points.slice(s![0..h - 2, 1..w - 1, ..]);
    let n_fd = normalize3(cross3(right - left, top - bottom));

    let ray_c = ray.slice(s![1..h - 1, 1..w - 1, ..]);
    // Orient N_d to face the camera (N_d·ray < 0), like the rendered normal.
    let orient = -(n_fd.clone() * ray_c).sum_dim(2).sign();
    let n_fd = n_fd * orient;

    let n_r = n_render.slice(s![1..h - 1, 1..w - 1, ..]);
    // Cosine error 1 - <N_render, N_depth>.
    -(n_r * n_fd).sum_dim(2) + 1.0
}

/// Depth-distortion loss (GOF): per pixel `Σᵢ>ⱼ wᵢwⱼ(mᵢ−mⱼ)² / ((1−T)²+ε)`
/// over NDC-mapped depths, accumulated and normalized in the rasterizer with
/// a full backward. Reads that channel off the `[H, W, 7]` geometry slice.
pub fn depth_distortion(geo: Tensor<3>) -> Tensor<3> {
    geo.slice(s![.., .., 4..5]).clamp_min(0.0)
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

/// Metric depth supervision: L1 between rendered depth (radial, converted to
/// `z`) and the per-view `LiDAR` z-depth, evaluated sparsely at the `LiDAR`
/// grid (the GT is a point sample; the render is downscaled to it, not the
/// GT upscaled). Soft-weighted by `[0, 1]` confidence, gated by coverage and
/// a positive GT. Returns `(|Δ|·w, w)` for the caller's weighted mean.
pub fn depth_l1_loss(
    geo: Tensor<3>,
    alpha: Tensor<3>,
    gt_z: Tensor<3>,
    gt_conf: Tensor<3>,
    camera: &Camera,
    img_size: UVec2,
) -> (Tensor<3>, Tensor<3>) {
    let [hd, wd, _] = gt_z.dims();

    // Rendered radial depth -> z, at render resolution.
    let ray_len = camera_ray_grid(&geo, camera, img_size)
        .powf_scalar(2.0)
        .sum_dim(2)
        .sqrt();
    let rendered_z = rendered_depth(geo, alpha.clone()) / ray_len;

    // Downscale rendered depth + coverage to the GT (LiDAR) grid (sparse).
    let down_z = resize_nearest(rendered_z, hd, wd);
    let down_alpha = resize_nearest(alpha, hd, wd);

    let gt_z = gt_z.detach();
    // Confidence (already in [0, 1]) is the per-pixel weight, gated by
    // coverage and a positive GT.
    let weight = (gt_conf.detach().clamp(0.0, 1.0)
        * down_alpha.greater_elem(0.5).float()
        * gt_z.clone().greater_elem(0.0).float())
    .detach();

    let err = (down_z - gt_z).abs();
    (err * weight.clone(), weight)
}
