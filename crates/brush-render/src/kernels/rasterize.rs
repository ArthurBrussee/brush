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
    ALPHA_CUTOFF_MID, PROJECTED_GEO_LANES, PROJECTED_LANES, PROJECTED_LANES_USIZE, TILE_SIZE,
    TILE_WIDTH, alpha_cutoff_weight, calc_sigma, dist_ndc, map_1d_to_2d,
};
use super::types::{RasterizeUniforms, Sym2};

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_kernel(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    projected: &Tensor<f32>,
    projected_geo: &Tensor<f32>,
    out_img_packed: &mut Tensor<u32>,
    out_img_f32: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    u: RasterizeUniforms,
    #[comptime] bwd_info: bool,
    #[comptime] smooth_cutoff: bool,
    #[comptime] geo: bool,
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
    let mut local_batch = Shared::new_slice((TILE_SIZE * PROJECTED_LANES) as usize);
    let mut load_gid =
        Shared::new_slice(comptime![if bwd_info { TILE_SIZE } else { 1u32 }] as usize);
    let num_done_atomic = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let max_useful_isect = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let mut range = Shared::new_slice(2usize);

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
    // RaDe-GS geometry accumulators: alpha-blended view-space normal + the
    // per-pixel ray-plane depth (depth varies across each splat footprint).
    let mut pix_nx = 0.0f32;
    let mut pix_ny = 0.0f32;
    let mut pix_nz = 0.0f32;
    let mut pix_depth = 0.0f32;
    // Depth-distortion (GOF): squared pairwise error `Sum_{i>j} w_i w_j
    // (m_i - m_j)^2` over NDC-mapped depths `m = far(t-near)/((far-near)t)`,
    // accumulated front-to-back via the prefix moments `S1 = Sum(w*m)`,
    // `S2 = Sum(w*m^2)` (error_i = m_i^2*A_<i + S2_<i - 2*m_i*S1_<i), and
    // normalized by the final accumulated alpha squared at write-out.
    // Matches GOF's renderCUDA; the squared form is sign-correct regardless
    // of how per-pixel ray-plane depths order against the blend order.
    let mut pix_dist = 0.0f32;
    let mut pix_dist_s1 = 0.0f32;
    let mut pix_dist_s2 = 0.0f32;
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

            let w_cut = if comptime![smooth_cutoff] {
                alpha_cutoff_weight(alpha)
            } else {
                select(alpha >= ALPHA_CUTOFF_MID, 1.0f32, 0.0f32)
            };
            if sigma >= 0.0f32 && w_cut > 0.0f32 {
                let alpha_eff = alpha * w_cut;
                let next_t = t_acc * (1.0f32 - alpha_eff);
                if next_t <= 1.0e-4f32 {
                    done = true;
                } else {
                    if comptime![bwd_info] {
                        visible[load_gid[t as usize] as usize] = 1.0f32;
                    }
                    let vis = alpha_eff * t_acc;
                    pix_r += max(local_batch[dst_base + 6], 0.0f32) * vis;
                    pix_g += max(local_batch[dst_base + 7], 0.0f32) * vis;
                    pix_b += max(local_batch[dst_base + 8], 0.0f32) * vis;
                    if comptime![geo] {
                        // Read geo from global (not staged in shared — the 6
                        // lanes would push workgroup shared past wgpu's 16 KiB
                        // default limit). `projected_geo` is L1-cached.
                        let gid_t = compact_gid_from_isect[(batch_start + t) as usize];
                        let geo_base = (gid_t * PROJECTED_GEO_LANES) as usize;
                        let off_x = xy_x - pixel_coord_x;
                        let off_y = xy_y - pixel_coord_y;
                        let t_pix = projected_geo[geo_base] * off_x
                            + projected_geo[geo_base + 1] * off_y
                            + projected_geo[geo_base + 2];
                        // Distortion increment: prefix weight `1 - t_acc` and
                        // prefix moments, all *before* this splat updates them.
                        let m = dist_ndc(t_pix);
                        let w_prev = 1.0f32 - t_acc;
                        pix_dist += vis * (m * m * w_prev + pix_dist_s2 - 2.0f32 * m * pix_dist_s1);
                        pix_dist_s1 += m * vis;
                        pix_dist_s2 += m * m * vis;
                        pix_depth += t_pix * vis;
                        pix_nx += projected_geo[geo_base + 3] * vis;
                        pix_ny += projected_geo[geo_base + 4] * vis;
                        pix_nz += projected_geo[geo_base + 5] * vis;
                    }
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
            let nchan = comptime![if geo { 11u32 } else { 4u32 }];
            let base = (pix_id * nchan) as usize;
            out_img_f32[base] = final_r;
            out_img_f32[base + 1] = final_g;
            out_img_f32[base + 2] = final_b;
            out_img_f32[base + 3] = final_a;
            if comptime![geo] {
                // Geometry has no background; leftover transmittance contributes 0.
                out_img_f32[base + 4] = pix_nx;
                out_img_f32[base + 5] = pix_ny;
                out_img_f32[base + 6] = pix_nz;
                out_img_f32[base + 7] = pix_depth;
                // Normalized distortion (GOF: raw / (1-T)^2), then the raw
                // mapped-depth moments S1/S2 the backward needs to
                // reconstruct per-splat pairwise sums.
                out_img_f32[base + 8] = pix_dist / (final_a * final_a + 1e-7f32);
                out_img_f32[base + 9] = pix_dist_s1;
                out_img_f32[base + 10] = pix_dist_s2;
            }
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
