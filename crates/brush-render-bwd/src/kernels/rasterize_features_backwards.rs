//! Backward pass for the feature rasterizer.
//!
//! Geometry is a constant in the feature pass (`DiG` detaches it), so the
//! only gradient is `∂L/∂features[g] = Σ_pix vis(g, pix) · v_out[pix]`.
//! The visibility weights depend only on prefix products of alpha, so a
//! per-pixel *forward-order* replay of the rasterize walk recovers them
//! exactly — no back-to-front trick needed. Each thread owns one pixel
//! (same layout as the forward kernel) and scatters `v_out · vis` into
//! the dense `[N, feat_dim]` gradient with [`AtomicAddF32`].
//!
//! Contention note: pixels of a tile hit the same splats, so the atomics
//! serialize within a tile. At `DiG`'s low feature resolution this is
//! acceptable; if it shows up in profiles, switch to per-splat-thread
//! accumulation like `rasterize_backwards`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use brush_render::kernels::helpers::{
    ALPHA_CUTOFF_MID, PROJECTED_LANES, TILE_SIZE, TILE_WIDTH, calc_sigma, map_1d_to_2d,
};
use brush_render::kernels::rasterize_features::FEATURE_SPATIAL_LANES;
use brush_render::kernels::types::{RasterizeUniforms, Sym2};

use super::rasterize_backwards::AtomicAddF32;

#[cube(launch)]
pub fn rasterize_features_backwards_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    v_output: &Tensor<f32>,
    v_features: &mut Tensor<Atomic<A::Storage>>,
    u: RasterizeUniforms,
    #[comptime] feat_dim: usize,
) {
    let global_id = ABSOLUTE_POS as u32;
    let (pix_x, pix_y) = map_1d_to_2d(global_id, u.tile_bw);
    let pix_id = pix_x + pix_y * u.img_w;
    let pixel_coord_x = pix_x as f32 + 0.5f32;
    let pixel_coord_y = pix_y as f32 + 0.5f32;
    let tile_loc_x = pix_x / TILE_WIDTH;
    let tile_loc_y = pix_y / TILE_WIDTH;
    let tile_id = tile_loc_x + tile_loc_y * u.tile_bw;
    let inside = pix_x < u.img_w && pix_y < u.img_h;

    let mut local_batch = Shared::new_slice((TILE_SIZE * FEATURE_SPATIAL_LANES) as usize);
    let mut load_gid = Shared::new_slice(TILE_SIZE as usize);
    let num_done_atomic = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let mut range = Shared::new_slice(2usize);

    let local_idx = UNIT_POS;
    if local_idx == 0u32 {
        range[0] = tile_offsets[(tile_id * 2u32) as usize];
        range[1] = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
        Atomic::store(&num_done_atomic[0], 0u32);
    }

    let range_lo = workgroup_uniform_load(&range[0]);
    let range_hi = workgroup_uniform_load(&range[1]);

    // Preload this pixel's incoming gradient — read repeatedly below.
    // The alpha channel (index feat_dim) carries no gradient: the loss
    // detaches alpha before the normalization, matching the reference.
    let out_chans = comptime![feat_dim + 1];
    let mut v_pix = Array::<f32>::new(feat_dim);
    if inside {
        let base = pix_id as usize * out_chans;
        for d in 0..feat_dim {
            v_pix[d] = v_output[base + d];
        }
    }

    let mut t_acc = 1.0f32;
    let mut done = !inside;

    if done {
        Atomic::fetch_add(&num_done_atomic[0], 1u32);
    }
    sync_cube();

    let mut batch_start = range_lo;
    while batch_start < range_hi {
        if workgroup_uniform_load_atomic(&num_done_atomic[0]) >= TILE_SIZE {
            break;
        }
        let remaining = min(TILE_SIZE, range_hi - batch_start);
        let load_isect_id = batch_start + local_idx;
        if local_idx < remaining {
            let compact_gid = compact_gid_from_isect[load_isect_id as usize];
            let src_base = (compact_gid * PROJECTED_LANES) as usize;
            let dst_base = (local_idx * FEATURE_SPATIAL_LANES) as usize;
            #[unroll]
            for lane in 0..FEATURE_SPATIAL_LANES as usize {
                local_batch[dst_base + lane] = projected[src_base + lane];
            }
            load_gid[local_idx as usize] = global_from_compact_gid[compact_gid as usize];
        }
        sync_cube();

        let was_done = done;
        let mut t = 0u32;
        while !done && t < remaining {
            let dst_base = (t * FEATURE_SPATIAL_LANES) as usize;
            let xy_x = local_batch[dst_base];
            let xy_y = local_batch[dst_base + 1];
            let conic = Sym2 {
                c00: local_batch[dst_base + 2],
                c01: local_batch[dst_base + 3],
                c11: local_batch[dst_base + 4],
            };
            let color_a = local_batch[dst_base + 5];
            let sigma = calc_sigma(pixel_coord_x, pixel_coord_y, conic, xy_x, xy_y);
            let alpha = min(0.999f32, color_a * f32::exp(-sigma));

            if sigma >= 0.0f32 && alpha >= ALPHA_CUTOFF_MID {
                let next_t = t_acc * (1.0f32 - alpha);
                if next_t <= 1.0e-4f32 {
                    done = true;
                } else {
                    let vis = alpha * t_acc;
                    let gid = load_gid[t as usize];
                    let feat_base = gid as usize * feat_dim;
                    for d in 0..feat_dim {
                        A::add(&v_features[feat_base + d], v_pix[d] * vis);
                    }
                    t_acc = next_t;
                }
            }
            t += 1u32;
        }
        if !was_done && done {
            Atomic::fetch_add(&num_done_atomic[0], 1u32);
        }
        batch_start += TILE_SIZE;
    }
}
