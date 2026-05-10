//! Map per-splat tile counts to per-intersection (tile_id, compact_gid)
//! pairs.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::{
    compute_bbox_extent, count_contributing_tiles, get_tile_bbox, read_main_splat, tile_rect,
    will_primitive_contribute,
};

pub const WG_SIZE: u32 = 256;

#[cube(launch_unchecked)]
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
    let (ex, ey) = compute_bbox_extent(conic, power_threshold);
    let bb = get_tile_bbox(xy_x, xy_y, ex, ey, tile_bw, tile_bh);

    // Inclusive prefix sum: use cum[compact_gid - 1] as base (or 0 for first).
    // Index with `max(compact_gid, 1) - 1` so the read is always in-bounds.
    let prev_idx = max(compact_gid, 1u32) - 1u32;
    let base_isect_id = select(
        compact_gid == 0u32,
        0u32,
        splat_cum_hit_counts[prev_idx as usize],
    );
    // Slot budget reserved for this splat in PF.
    let pf_count = splat_cum_hit_counts[compact_gid as usize] - base_isect_id;
    // What this kernel's loop body will actually count. Should match
    // `pf_count` because PF runs the same `count_contributing_tiles`
    // helper, but the two dispatches go through separate shader
    // optimisation passes; we belt-and-suspenders the mismatch below.
    let local_count = count_contributing_tiles(bb, xy_x, xy_y, conic, power_threshold);
    let writable = min(local_count, pf_count);

    // Tile id past the valid range — radix-sorts after every real tile
    // and lives outside `tile_offsets`, so the rasterize pass never
    // visits these padded slots.
    let sentinel_tile_id = tile_bw * tile_bh;

    let bb_w = bb.max_x - bb.min_x;
    let num_tiles_bbox = (bb.max_y - bb.min_y) * bb_w;
    let mut num_tiles_hit = 0u32;
    for tile_idx in 0u32..num_tiles_bbox {
        let tx = (tile_idx % bb_w) + bb.min_x;
        let ty = (tile_idx / bb_w) + bb.min_y;
        let rect = tile_rect(tx, ty);
        if will_primitive_contribute(rect, xy_x, xy_y, conic, power_threshold)
            && num_tiles_hit < writable
        {
            let tile_id = tx + ty * tile_bw;
            let isect_id = base_isect_id + num_tiles_hit;
            tile_id_from_isect[isect_id as usize] = tile_id;
            compact_gid_from_isect[isect_id as usize] = compact_gid;
            num_tiles_hit += 1u32;
        }
    }

    // Pad the leftover budget with sentinel rows so no slot in
    // `[base_isect_id, base_isect_id + pf_count)` is left uninitialised.
    for pad_idx in writable..pf_count {
        let isect_id = base_isect_id + pad_idx;
        tile_id_from_isect[isect_id as usize] = sentinel_tile_id;
        compact_gid_from_isect[isect_id as usize] = compact_gid;
    }
}
