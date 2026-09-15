//! Compute projected splat data for visible gaussians. PF already
//! culled non-finite-cov2d splats so this kernel trusts `calc_cov2d`.

use super::helpers::{
    apply_scale_floor, calc_cov2d, compensate_cov2d, is_finite_f32, read_quat_unorm, read_scale,
    sigmoid, world_to_cam, write_projected_splat,
};
use super::sh::{num_sh_coeffs, sh_coeffs_to_color};
use super::types::{ProjectUniforms, Splat, Vec3A};
use crate::kernels::camera_model::{CameraModel, project};
use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

// The `#[cube]` macro's terminal `write_projected_splat(...)` call expands to
// `expr();` plus a trailing `()` placeholder, which trips
// `semicolon_if_nothing_returned`. False positive — the macro already provides
// the semicolon. Silence here rather than in the kernel body.
#[allow(clippy::semicolon_if_nothing_returned)]
#[cube(launch)]
pub fn project_visible_kernel(
    transforms: &Tensor<f32>,
    coeffs: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    min_scale: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    compact_from_global: &mut Tensor<u32>,
    projected: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] sh_degree: u32,
    #[comptime] camera_model: CameraModel,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];
    // Inverse map so the backward's compact gradients can be gathered per
    // global splat. Offset by one: row 0 of those buffers is the zero row
    // that culled splats point at.
    compact_from_global[global_gid as usize] = compact_gid + 1u32;

    // means(3) + quats(4) + log_scales(3)
    let base = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = read_scale(transforms, base);
    let quat_unorm = read_quat_unorm(transforms, base);
    let quat = quat_unorm.normalize();

    let opac_sig = sigmoid(raw_opacities[global_gid as usize]);
    let floor = apply_scale_floor(scale, opac_sig, min_scale, global_gid, has_min_scale);

    let mean_c = world_to_cam(mean, u);
    let raw_cov = calc_cov2d(floor.scale, quat, mean_c, u, camera_model);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let opac = floor.opac * filter_comp;
    let conic = cov.inverse();

    let (mean2d_x, mean2d_y) = project(mean_c, u.pinhole_params, camera_model);

    // Viewdir. Safe to normalize: splats with length(mean - cam) == 0
    // would already be culled in PF.
    let v = mean.sub(u.camera_pos()).normalize();

    let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
    let raw = sh_coeffs_to_color(coeffs, coeff_base, sh_degree, v);
    // SH-to-color offset.
    let cr = raw.x() + 0.5f32;
    let cg = raw.y() + 0.5f32;
    let cb = raw.z() + 0.5f32;

    // Scrub NaN / Inf and clamp so the rasterize backward's gradient
    // term can't amplify past f32 range.
    let cr_c = clamp(select(is_finite_f32(cr), cr, 0.0f32), -100.0f32, 100.0f32);
    let cg_c = clamp(select(is_finite_f32(cg), cg, 0.0f32), -100.0f32, 100.0f32);
    let cb_c = clamp(select(is_finite_f32(cb), cb, 0.0f32), -100.0f32, 100.0f32);

    write_projected_splat(
        projected,
        compact_gid,
        Splat {
            xy_x: mean2d_x,
            xy_y: mean2d_y,
            conic_x: conic.c00,
            conic_y: conic.c01,
            conic_z: conic.c11,
            color_a: opac,
            color_r: cr_c,
            color_g: cg_c,
            color_b: cb_c,
        },
    );
}
