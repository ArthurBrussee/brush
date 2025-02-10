use burn_jit::cubecl;
use burn_jit::cubecl::{cube, prelude::*};

#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn stats_gather_kernel(
    gs_ids: &Tensor<u32>,
    num_visible: &Tensor<u32>,
    radii: &Tensor<f32>,
    xy_grads: &Tensor<Line<f32>>,
    norm_sum: &mut Tensor<f32>,
    counts: &mut Tensor<u32>,
    max_radii: &mut Tensor<f32>,
    #[comptime] w: u32,
    #[comptime] h: u32,
) {
    let compact_gid = ABSOLUTE_POS_X;
    let num_vis = num_visible[0];

    if compact_gid >= num_vis {
        terminate!();
    }

    let mut line: Line<f32> = Line::empty(2);

    // Nb: Clippy reports a warning here about a useless conversion but it's wrong.
    line[0] = comptime!(w as f32 / 2.0);
    line[1] = comptime!(h as f32 / 2.0);

    let xy_grad = xy_grads[compact_gid] * line;
    let xy_grad_norm = f32::sqrt(xy_grad[0] * xy_grad[0] + xy_grad[1] * xy_grad[1]);

    let global_gid = gs_ids[compact_gid];
    let radius = radii[global_gid];

    norm_sum[global_gid] += xy_grad_norm;
    counts[global_gid] += 1;

    let radii_norm = radius / comptime!(if w > h { w as f32 } else { h as f32 });
    max_radii[global_gid] = f32::max(radii_norm, max_radii[global_gid]);
}
