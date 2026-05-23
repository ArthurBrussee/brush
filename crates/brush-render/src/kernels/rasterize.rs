//! Tile-based gaussian rasterizer.
//!
//! One workgroup of `TILE_SIZE` threads per tile, each thread processes a
//! single pixel. Threads cooperate to load splats into a workgroup-shared
//! `local_batch` then iterate splats across all pixels.
//!
//! `bwd_info` enables: (a) writing `out_img` as 4 f32s (rgba) so the
//! backward kernel can recover the final color/alpha; (b) marking
//! `visible` splats; (c) shrinking `tile_offsets[tile*2+1]` to "one past
//! the last splat any pixel actually consumed" so the backward kernel's
//! outer loop ends early. When `bwd_info=false` the kernel writes a
//! packed u8x4 to `out_img` and skips the backward bookkeeping.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::{
    PROJECTED_LANES, PROJECTED_LANES_USIZE, TILE_SIZE, TILE_WIDTH, calc_sigma, map_1d_to_2d,
};
use super::types::{RasterizeUniforms, Sym2};

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_kernel(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    projected: &Tensor<f32>,
    out_img_packed: &mut Tensor<u32>,
    out_img_f32: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    u: RasterizeUniforms,
    #[comptime] bwd_info: bool,
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

    // Workgroup-shared splat batch + bookkeeping. The bwd-only `load_gid`
    // gets a comptime-tiny size when `bwd_info=false` so we don't pay
    // 1 KiB of static shared mem on the forward-only variant.
    let mut local_batch = SharedMemory::<f32>::new((TILE_SIZE * PROJECTED_LANES) as usize);
    let mut load_gid =
        SharedMemory::<u32>::new(comptime![if bwd_info { TILE_SIZE } else { 1u32 }] as usize);
    let num_done_atomic = SharedMemory::<Atomic<u32>>::new(1usize);
    let max_useful_isect = SharedMemory::<Atomic<u32>>::new(1usize);
    let mut range = SharedMemory::<u32>::new(2usize);

    let local_idx = UNIT_POS;
    if local_idx == 0u32 {
        range[0] = tile_offsets[(tile_id * 2u32) as usize];
        range[1] = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
        Atomic::store(&num_done_atomic[0], 0u32);
    }

    // Uniform-marked loads so the loop bounds + early-exit don't trip
    // WebGPU's "workgroupBarrier in non-uniform control flow" check.
    let range_lo = workgroup_uniform_load(&range[0]);
    let range_hi = workgroup_uniform_load(&range[1]);

    if comptime![bwd_info] && local_idx == 0u32 {
        Atomic::store(&max_useful_isect[0], range_lo);
    }

    let mut t_acc = 1.0f32;
    let mut pix_r = 0.0f32;
    let mut pix_g = 0.0f32;
    let mut pix_b = 0.0f32;
    let mut done = !inside;
    let mut last_useful_isect = range_lo;

    if done {
        Atomic::fetch_add(&num_done_atomic[0], 1u32);
    }
    sync_cube();

    let mut batch_start = range_lo;
    while batch_start < range_hi {
        // Doubles as the sync between previous iter's local_batch reads
        // and this iter's overwrites.
        if workgroup_uniform_load_atomic(&num_done_atomic[0]) >= TILE_SIZE {
            break;
        }
        let remaining = min(TILE_SIZE, range_hi - batch_start);
        let load_isect_id = batch_start + local_idx;
        let mut compact_gid = 0u32;
        if local_idx < remaining {
            compact_gid = compact_gid_from_isect[load_isect_id as usize];
        }
        if local_idx < remaining {
            let src_base = (compact_gid * PROJECTED_LANES) as usize;
            let dst_base = (local_idx * PROJECTED_LANES) as usize;
            #[unroll]
            for lane in 0..PROJECTED_LANES_USIZE {
                local_batch[dst_base + lane] = projected[src_base + lane];
            }
            if comptime![bwd_info] {
                load_gid[local_idx as usize] = global_from_compact_gid[compact_gid as usize];
            }
        }
        sync_cube();

        let was_done = done;
        let mut t = 0u32;
        while !done && t < remaining {
            let dst_base = (t * PROJECTED_LANES) as usize;
            // Read the spatial fields first; defer color loads to the
            // contributing branch so non-contributing splats don't pay
            // for the rgb reads.
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

            if sigma >= 0.0f32 && alpha >= 1.0f32 / 255.0f32 {
                let next_t = t_acc * (1.0f32 - alpha);
                if next_t <= 1.0e-4f32 {
                    done = true;
                } else {
                    if comptime![bwd_info] {
                        visible[load_gid[t as usize] as usize] = 1.0f32;
                    }
                    let vis = alpha * t_acc;
                    pix_r += max(local_batch[dst_base + 6], 0.0f32) * vis;
                    pix_g += max(local_batch[dst_base + 7], 0.0f32) * vis;
                    pix_b += max(local_batch[dst_base + 8], 0.0f32) * vis;
                    t_acc = next_t;
                    last_useful_isect = batch_start + t + 1u32;
                }
            }
            t += 1u32;
        }
        if !was_done && done {
            Atomic::fetch_add(&num_done_atomic[0], 1u32);
        }
        batch_start += TILE_SIZE;
    }

    if inside {
        let final_r = pix_r + t_acc * u.bg_r;
        let final_g = pix_g + t_acc * u.bg_g;
        let final_b = pix_b + t_acc * u.bg_b;
        let final_a = 1.0f32 - t_acc;
        if comptime![bwd_info] {
            let base = (pix_id * 4u32) as usize;
            out_img_f32[base] = final_r;
            out_img_f32[base + 1] = final_g;
            out_img_f32[base + 2] = final_b;
            out_img_f32[base + 3] = final_a;
        } else {
            let r = clamp(final_r * 255.0f32, 0.0f32, 255.0f32) as u32;
            let g = clamp(final_g * 255.0f32, 0.0f32, 255.0f32) as u32;
            let b = clamp(final_b * 255.0f32, 0.0f32, 255.0f32) as u32;
            let a = clamp(final_a * 255.0f32, 0.0f32, 255.0f32) as u32;
            let packed = r | (g << 8u32) | (b << 16u32) | (a << 24u32);
            out_img_packed[pix_id as usize] = packed;
        }
    }

    if comptime![bwd_info] {
        Atomic::fetch_max(&max_useful_isect[0], last_useful_isect);
        sync_cube();
        if local_idx == 0u32 {
            tile_offsets[(tile_id * 2u32 + 1u32) as usize] = Atomic::load(&max_useful_isect[0]);
        }
    }
}
