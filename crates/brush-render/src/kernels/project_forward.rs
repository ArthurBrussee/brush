//! Project & cull pass.
//!
//! `project_forward` is the sole visibility gate. Guards are positive-
//! phrased so NaN reliably fails them (NaN comparisons are unordered).
//! Finite out-of-distribution values pass; calc_cov2d clamps overflow
//! internally.

use super::helpers::{
    calc_cov2d, compensate_cov2d, compute_bbox_extent, count_contributing_tiles, read_mean_viewspace,
    read_quat_unorm, read_scale, get_tile_bbox, is_finite_f32, sigmoid,
};
use super::types::ProjectUniforms;
use crate::kernels::camera_model::{CameraModel, project};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn project_forward_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &mut Tensor<u32>,
    depths: &mut Tensor<f32>,
    num_visible: &mut Tensor<Atomic<u32>>,
    intersect_counts: &mut Tensor<u32>,
    num_intersections: &mut Tensor<Atomic<u32>>,
    max_radius: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] camera_model: CameraModel,
) {
    let global_gid = ABSOLUTE_POS as u32;
    if global_gid >= u.total_splats {
        terminate!();
    }

    // means(3) + quats(4) + log_scales(3)
    let base = (global_gid * 10u32) as usize;

    let mean_c = read_mean_viewspace(transforms, base, u);
    if !(mean_c.is_finite() && mean_c.z() >= 0.01f32 && mean_c.z() <= 1.0e10f32) {
        terminate!();
    }

    let scale = read_scale(transforms, base);
    if !scale.is_finite() {
        terminate!();
    }

    let quat_unorm = read_quat_unorm(transforms, base);
    let qnorm_sq = quat_unorm.dot(quat_unorm);
    if !(qnorm_sq >= 1.0e-6f32 && is_finite_f32(qnorm_sq)) {
        terminate!();
    }

    let raw_opac = raw_opacities[global_gid as usize];
    if !is_finite_f32(raw_opac) {
        terminate!();
    }

    let quat = quat_unorm.normalize();

    let raw_cov = calc_cov2d(scale, quat, mean_c, u, camera_model);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let opac = sigmoid(raw_opac) * filter_comp;

    if !cov.is_finite() {
        terminate!();
    }

    let (mean2d_x, mean2d_y) = project(mean_c, u.pinhole_params, camera_model);

    if !(opac >= 1.0f32 / 255.0f32) {
        terminate!();
    }

    let power_threshold = f32::ln(opac * 255.0f32);
    let conic = cov.inverse();
    let (ex, ey) = compute_bbox_extent(conic, power_threshold);
    if !(ex >= 0.0f32 && ey >= 0.0f32) {
        terminate!();
    }

    let img_w_f = u.img_w as f32;
    let img_h_f = u.img_h as f32;
    let on_screen = mean2d_x + ex > 0.0f32
        && mean2d_x - ex < img_w_f
        && mean2d_y + ey > 0.0f32
        && mean2d_y - ey < img_h_f;
    if !on_screen {
        terminate!();
    }

    let bb = get_tile_bbox(mean2d_x, mean2d_y, ex, ey, u.tile_bw, u.tile_bh);
    let num_tiles_hit = count_contributing_tiles(bb, mean2d_x, mean2d_y, conic, power_threshold);

    intersect_counts[global_gid as usize] = num_tiles_hit;
    Atomic::fetch_add(&num_intersections[0], num_tiles_hit);

    // Screen-space radius (pixels) for the small-splat prior.
    max_radius[global_gid as usize] = f32::max(ex / img_w_f, ey / img_h_f);

    let write_id = Atomic::fetch_add(&num_visible[0], 1u32);
    global_from_compact_gid[write_id as usize] = global_gid;
    depths[write_id as usize] = mean_c.z();
}
