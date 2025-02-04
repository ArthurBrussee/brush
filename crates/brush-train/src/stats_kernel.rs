use burn_jit::cubecl;
use burn_jit::cubecl::{cube, prelude::*};

#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn stats_gather_kernel(
    gs_ids: &Tensor<u32>,
    num_visible: &Tensor<u32>,
    radii: &Tensor<f32>,
    refine_weight: &Tensor<f32>,
    accum_refine_weight: &mut Tensor<f32>,
    visible_counts: &mut Tensor<u32>,
    max_radii: &mut Tensor<f32>,
    #[comptime] w: u32,
    #[comptime] h: u32,
) {
    let compact_gid = ABSOLUTE_POS_X;
    let num_vis = num_visible[0];

    if compact_gid >= num_vis {
        terminate!();
    }

    let global_gid = gs_ids[compact_gid];
    let radius = radii[global_gid];

    accum_refine_weight[global_gid] += refine_weight[compact_gid];
    visible_counts[global_gid] += 1;

    let radii_norm = radius / comptime!(if w > h { w as f32 } else { h as f32 });
    max_radii[global_gid] = f32::max(radii_norm, max_radii[global_gid]);
}
