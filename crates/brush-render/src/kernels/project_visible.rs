//! Compute projected splat data for visible gaussians. PF already
//! culled non-finite-cov2d splats so this kernel trusts `calc_cov2d`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::{
    calc_cov2d, compensate_cov2d, inverse_sym2, is_finite_f32, sigmoid, world_to_cam,
    write_projected_splat,
};
use super::sh::{num_sh_coeffs, sh_coeffs_to_color};
use super::types::{ProjectUniforms, Quat, Splat, Vec3A};

pub const WG_SIZE: u32 = 256;

#[cube(launch_unchecked)]
pub fn project_visible_kernel(
    transforms: &Tensor<f32>,
    coeffs: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    projected: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] sh_degree: u32,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];

    // means(3) + quats(4) + log_scales(3)
    let base = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = Vec3A::new(
        f32::exp(transforms[base + 7]),
        f32::exp(transforms[base + 8]),
        f32::exp(transforms[base + 9]),
    );
    let quat_unorm = Quat::new(
        transforms[base + 3],
        transforms[base + 4],
        transforms[base + 5],
        transforms[base + 6],
    );
    let quat = quat_unorm.normalize();

    let mean_c = world_to_cam(mean, u);
    let raw_cov = calc_cov2d(scale, quat, mean_c, u);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let opac = sigmoid(raw_opacities[global_gid as usize]) * filter_comp;
    let conic = inverse_sym2(cov);

    let inv_z = 1.0f32 / mean_c.z();
    let mean2d_x = u.focal_x * mean_c.x() * inv_z + u.pixel_center_x;
    let mean2d_y = u.focal_y * mean_c.y() * inv_z + u.pixel_center_y;

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
