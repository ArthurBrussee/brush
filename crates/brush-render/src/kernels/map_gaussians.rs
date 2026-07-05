//! Map per-splat tile counts to per-intersection (tile_id, compact_gid)
//! pairs.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::read_main_splat;
use super::tile_intersect::{accutile_write_tiles, compute_snugbox, snugbox_is_valid};

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
pub fn map_gaussians_to_intersect_kernel(
    projected: &Tensor<f32>,
    splat_cum_hit_counts: &Tensor<u32>,
    tile_id_from_isect: &mut Tensor<u32>,
    compact_gid_from_isect: &mut Tensor<u32>,
    tile_bw: u32,
    tile_bh: u32,
    num_visible: u32,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= num_visible {
        terminate!();
    }

    let (xy_x, xy_y, conic, opac) = read_main_splat(projected, compact_gid);

    let power_threshold = f32::ln(opac * 255.0f32);
    let sb = compute_snugbox(conic, power_threshold, xy_x, xy_y, tile_bw, tile_bh);
    if !snugbox_is_valid(sb) {
        terminate!();
    }

    let prev_idx = max(compact_gid, 1u32) - 1u32;
    let base_isect_id = select(
        compact_gid == 0u32,
        0u32,
        splat_cum_hit_counts[prev_idx as usize],
    );
    let pf_count = splat_cum_hit_counts[compact_gid as usize] - base_isect_id;

    let sentinel_tile_id = tile_bw * tile_bh;

    for pad_idx in accutile_write_tiles(
        sb,
        conic,
        tile_bw,
        base_isect_id,
        compact_gid,
        pf_count,
        tile_id_from_isect,
        compact_gid_from_isect,
    )..pf_count
    {
        let isect_id = base_isect_id + pad_idx;
        tile_id_from_isect[isect_id as usize] = sentinel_tile_id;
        compact_gid_from_isect[isect_id as usize] = compact_gid;
    }
}
