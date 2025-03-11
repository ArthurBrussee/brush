use burn_cubecl::cubecl;
use burn_cubecl::cubecl::{cube, prelude::*};

#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn stats_gather_kernel(
    gs_ids: &Tensor<u32>,
    num_visible: &Tensor<u32>,
    refine_weight: &Tensor<Line<f32>>,
    accum_refine_weight: &mut Tensor<f32>,
    #[comptime] w: u32,
    #[comptime] h: u32,
) {
    let compact_gid = ABSOLUTE_POS_X;
    let num_vis = num_visible[0];

    if compact_gid >= num_vis {
        terminate!();
    }

    let global_gid = gs_ids[compact_gid];

    let mut line: Line<f32> = Line::empty(2);

    // Nb: Clippy reports a warning here about a useless conversion but it's wrong.
    line[0] = comptime!(w as f32 / 2.0);
    line[1] = comptime!(h as f32 / 2.0);

    let refine_grads = refine_weight[compact_gid] * line;
    let refine_norm =
        f32::sqrt(refine_grads[0] * refine_grads[0] + refine_grads[1] * refine_grads[1]);

    accum_refine_weight[global_gid] = f32::max(accum_refine_weight[global_gid], refine_norm);
}
