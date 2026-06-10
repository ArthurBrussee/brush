//! Tile-cooperative opacity-along-ray integrator (the GOF "integrate" pass
//! for mesh extraction): given world-space query points and a view's
//! tile-sorted gaussian list from the forward render, evaluates
//! `alpha(p) = 1 - prod_g (1 - alpha_g(p))` with each gaussian evaluated at
//! `t = min(t*, t_point)` along the ray (`t*` = its peak depth, from the
//! view2gaussian formulation).
//!
//! Structure mirrors `rasterize`: one workgroup per tile, gaussians staged
//! through shared memory, every thread integrating its own vertex. Per view
//! the host first projects vertices to tiles ([`project_vertices_kernel`]),
//! histograms + prefix-sums tile ids, and scatters vertex ids into
//! per-tile slices. Off-screen / behind-near-plane vertices get the
//! sentinel tile `tile_bw * tile_bh`, which the dispatch never covers.

use super::helpers::{TILE_WIDTH, sigmoid, world_to_cam};
use super::types::{ProjectUniforms, Quat, Vec3A};
use crate::kernels::camera_model::{CameraModel, project};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

/// Workgroup size for the tiled integrate kernel. Matches rasterize's
/// `TILE_SIZE = TILE_WIDTH * TILE_WIDTH` so the cooperative-loading
/// batch size is the same.
pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

/// Per-gaussian floats stored in the workgroup-shared `local_batch`:
/// 3 mean (world) + 4 quat + 3 log_scale + 1 raw_opac = 11.
pub const LOCAL_BATCH_LANES: u32 = 11;

// Match integrate_alpha's constants.
const NEAR_PLANE: f32 = 0.2;
const ALPHA_CUTOFF: f32 = 1.0 / 255.0;
const ALPHA_CLAMP: f32 = 0.999;
const T_EARLY_OUT: f32 = 1.0e-4;
const COLOR_BLEND_POWER: f32 = 8.0;

/// Per-view vertex projection: writes each vertex's tile id (or the
/// off-screen sentinel), camera-space depth, GOF-convention ray dir
/// (`p_cam.xy / p_cam.z`), and projected pixel coords. Sentinel vertices'
/// other outputs stay garbage; the integrate kernel never reads them.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_vertices_kernel(
    points: &Tensor<f32>,
    tile_ids: &mut Tensor<u32>,
    depths: &mut Tensor<f32>,
    ray_dir_xy: &mut Tensor<f32>,
    pix_x: &mut Tensor<u32>,
    pix_y: &mut Tensor<u32>,
    n_points: u32,
    u: ProjectUniforms,
    #[comptime] camera_model: CameraModel,
) {
    let pid = ABSOLUTE_POS as u32;
    if pid >= n_points {
        terminate!();
    }
    let sentinel = u.tile_bw * u.tile_bh;

    let pidx3 = (pid * 3u32) as usize;
    let p_world = Vec3A::new(points[pidx3], points[pidx3 + 1], points[pidx3 + 2]);
    let p_cam = world_to_cam(p_world, u);
    let depth = p_cam.z();
    if !(depth > NEAR_PLANE) {
        tile_ids[pid as usize] = sentinel;
        terminate!();
    }

    let inv_z = 1.0f32 / depth;
    let rx = p_cam.x() * inv_z;
    let ry = p_cam.y() * inv_z;

    let (px, py) = project(p_cam, u.pinhole_params, camera_model);
    if !(px >= 0.0f32 && px < u.img_w as f32 && py >= 0.0f32 && py < u.img_h as f32) {
        tile_ids[pid as usize] = sentinel;
        terminate!();
    }
    let tx = (px as u32) / TILE_WIDTH;
    let ty = (py as u32) / TILE_WIDTH;
    let tile_id = ty * u.tile_bw + tx;

    tile_ids[pid as usize] = tile_id;
    depths[pid as usize] = depth;
    let rd_base = (pid * 2u32) as usize;
    ray_dir_xy[rd_base] = rx;
    ray_dir_xy[rd_base + 1] = ry;
    pix_x[pid as usize] = px as u32;
    pix_y[pid as usize] = py as u32;
}

/// Atomic-add histogram of `tile_ids` into `counts[tile_id + 1]`.
///
/// `counts` must be pre-zeroed and sized `n_tiles + 1`. After an
/// inclusive prefix sum, `counts[t]` is the *exclusive* start offset
/// for tile `t`'s vertices in `sorted_indices`, and `counts[t+1]` is
/// its end. Off-screen vertices (`tile_id == n_tiles`, the sentinel)
/// are silently skipped here.
#[cube(launch)]
pub fn histogram_tile_ids_kernel(
    tile_ids: &Tensor<u32>,
    counts: &mut Tensor<Atomic<u32>>,
    n_points: u32,
    n_tiles: u32,
) {
    let pid = ABSOLUTE_POS as u32;
    if pid >= n_points {
        terminate!();
    }
    let tid = tile_ids[pid as usize];
    if tid < n_tiles {
        Atomic::fetch_add(&counts[(tid + 1u32) as usize], 1u32);
    }
}

/// Atomic-scatter the vertex ids into `sorted_indices`, grouped by
/// tile_id. Within a tile, ordering depends on the atomic-fetch race
/// and is arbitrary — that's fine, the tiled integrate kernel only
/// cares about WHICH vertices fall into each tile, not their internal
/// order. Per-tile atomics keep contention bounded (one counter per
/// tile), so this is much cheaper than a full radix sort across the
/// global vertex list.
///
/// `write_counters` must be zeroed and sized `n_tiles`.
/// `sorted_indices` must be sized at least the total number of
/// in-bounds (non-sentinel) vertices.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_vertices_kernel(
    tile_ids: &Tensor<u32>,
    vertex_tile_offsets: &Tensor<u32>,
    write_counters: &mut Tensor<Atomic<u32>>,
    sorted_indices: &mut Tensor<u32>,
    n_points: u32,
    n_tiles: u32,
) {
    let pid = ABSOLUTE_POS as u32;
    if pid >= n_points {
        terminate!();
    }
    let tid = tile_ids[pid as usize];
    if tid < n_tiles {
        let slot = Atomic::fetch_add(&write_counters[tid as usize], 1u32);
        let pos = vertex_tile_offsets[tid as usize] + slot;
        sorted_indices[pos as usize] = pid;
    }
}

/// Tile-cooperative ray-gaussian integrate.
///
/// Dispatch: `(tile_bw, tile_bh, 1)` workgroups, each of [`TILE_SIZE`]
/// threads. The kernel reads its workgroup's vertex slice from
/// `sorted_indices[vertex_tile_offsets[tile_id] .. vertex_tile_offsets[tile_id+1]]`
/// and the tile's gaussian slice from
/// `compact_gid_from_isect[tile_offsets[tile_id*2] .. tile_offsets[tile_id*2+1]]`.
///
/// Each gaussian batch is loaded cooperatively into a workgroup-shared
/// `local_batch` (`TILE_SIZE * LOCAL_BATCH_LANES` floats); all threads
/// then iterate the batch in lockstep, doing the 3D ray-gaussian
/// integrate against their own vertex.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn integrate_kernel(
    // Splat data
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    global_from_compact_gid: &Tensor<u32>,
    rendered_image: &Tensor<f32>,
    // Per-view sorted vertex side-tables
    sorted_indices: &Tensor<u32>,
    vertex_tile_offsets: &Tensor<u32>,
    // Per-vertex precomputed data, in *original* (unsorted) order
    depths: &Tensor<f32>,
    ray_dir_xy: &Tensor<f32>,
    pix_x: &Tensor<u32>,
    pix_y: &Tensor<u32>,
    // Running aggregators, in *original* (unsorted) order
    min_alpha: &mut Tensor<f32>,
    color_sum: &mut Tensor<f32>,
    weight_sum: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] track_color: bool,
) {
    // 1D dispatch — `cube_count = num_tiles`, `cube_dim = TILE_SIZE`.
    // CUBE_POS is the workgroup index, which == the tile id directly.
    let tile_id = CUBE_POS as u32;
    let local_idx = UNIT_POS;

    // Look up this tile's vertex range and gaussian range.
    let v_lo = vertex_tile_offsets[tile_id as usize];
    let v_hi = vertex_tile_offsets[(tile_id + 1u32) as usize];
    let n_verts_in_tile = v_hi - v_lo;
    let g_range_lo = tile_offsets[(tile_id * 2u32) as usize];
    let g_range_hi = tile_offsets[(tile_id * 2u32 + 1u32) as usize];

    // Per-thread vertex (if any).
    let my_sorted_pos = v_lo + local_idx;
    let mut my_vidx = 0u32;
    let mut my_depth = 0.0f32;
    let mut my_ray = Vec3A::new(0.0f32, 0.0f32, 1.0f32);
    let mut my_px = 0u32;
    let mut my_py = 0u32;
    let active = local_idx < n_verts_in_tile;
    if active {
        my_vidx = sorted_indices[my_sorted_pos as usize];
        let vidx_u = my_vidx as usize;
        my_depth = depths[vidx_u];
        let rd_base = (my_vidx * 2u32) as usize;
        my_ray = Vec3A::new(ray_dir_xy[rd_base], ray_dir_xy[rd_base + 1], 1.0f32);
        my_px = pix_x[vidx_u];
        my_py = pix_y[vidx_u];
    }

    let mut local_batch = Shared::new_slice((TILE_SIZE * LOCAL_BATCH_LANES) as usize);
    let num_done_atomic = Shared::<[Atomic<u32>]>::new_slice(1usize);
    if local_idx == 0u32 {
        Atomic::store(&num_done_atomic[0], 0u32);
    }
    sync_cube();

    let mut t_acc = 1.0f32;
    let mut done = !active || g_range_lo >= g_range_hi;
    if done {
        Atomic::fetch_add(&num_done_atomic[0], 1u32);
    }
    sync_cube();

    let mut batch_start = g_range_lo;
    while batch_start < g_range_hi {
        if workgroup_uniform_load_atomic(&num_done_atomic[0]) >= TILE_SIZE {
            break;
        }
        let remaining = min(TILE_SIZE, g_range_hi - batch_start);

        // Cooperative load: each thread fetches one gaussian's full
        // transforms row + opacity into shared memory.
        let load_isect = batch_start + local_idx;
        if local_idx < remaining {
            let compact_gid = compact_gid_from_isect[load_isect as usize];
            let global_gid = global_from_compact_gid[compact_gid as usize];
            let src_base = (global_gid * 10u32) as usize;
            let dst_base = (local_idx * LOCAL_BATCH_LANES) as usize;
            // Unrolled to keep the 11-lane copy out of a loop body the
            // cube macro would have to specialise.
            local_batch[dst_base] = transforms[src_base];
            local_batch[dst_base + 1] = transforms[src_base + 1];
            local_batch[dst_base + 2] = transforms[src_base + 2];
            local_batch[dst_base + 3] = transforms[src_base + 3];
            local_batch[dst_base + 4] = transforms[src_base + 4];
            local_batch[dst_base + 5] = transforms[src_base + 5];
            local_batch[dst_base + 6] = transforms[src_base + 6];
            local_batch[dst_base + 7] = transforms[src_base + 7];
            local_batch[dst_base + 8] = transforms[src_base + 8];
            local_batch[dst_base + 9] = transforms[src_base + 9];
            local_batch[dst_base + 10] = raw_opacities[global_gid as usize];
        }
        sync_cube();

        if !done {
            let mut t = 0u32;
            while t < remaining {
                if t_acc < T_EARLY_OUT {
                    done = true;
                    Atomic::fetch_add(&num_done_atomic[0], 1u32);
                    break;
                }
                let dst_base = (t * LOCAL_BATCH_LANES) as usize;
                let g_mean_w = Vec3A::new(
                    local_batch[dst_base],
                    local_batch[dst_base + 1],
                    local_batch[dst_base + 2],
                );
                let g_scale = Vec3A::new(
                    f32::exp(local_batch[dst_base + 7]),
                    f32::exp(local_batch[dst_base + 8]),
                    f32::exp(local_batch[dst_base + 9]),
                );
                // Build quat from staged floats and normalize on use.
                let g_quat = Quat::new(
                    local_batch[dst_base + 3],
                    local_batch[dst_base + 4],
                    local_batch[dst_base + 5],
                    local_batch[dst_base + 6],
                )
                .normalize();
                let raw_opac = local_batch[dst_base + 10];

                let g_mean_c = world_to_cam(g_mean_w, u);
                let r_gv = u.view_rotation().mul_mat3(g_quat.to_mat3());

                let r_gv_t_d = r_gv.transpose_mul_vec3(my_ray);
                let v = Vec3A::new(
                    r_gv_t_d.x() / g_scale.x(),
                    r_gv_t_d.y() / g_scale.y(),
                    r_gv_t_d.z() / g_scale.z(),
                );
                let r_gv_t_mc = r_gv.transpose_mul_vec3(g_mean_c);
                let o = Vec3A::new(
                    -r_gv_t_mc.x() / g_scale.x(),
                    -r_gv_t_mc.y() / g_scale.y(),
                    -r_gv_t_mc.z() / g_scale.z(),
                );
                let a = v.dot(v);
                let b_coef = 2.0f32 * v.dot(o);
                let c_coef = o.dot(o);

                if a > 1.0e-20f32 {
                    let t_star = -b_coef / (2.0f32 * a);
                    let t_eval = min(t_star, my_depth);
                    if t_eval > NEAR_PLANE {
                        let power = -0.5f32 * (a * t_eval * t_eval + b_coef * t_eval + c_coef);
                        let power_safe = min(power, 0.0f32);
                        let alpha_raw = sigmoid(raw_opac) * f32::exp(power_safe);
                        let alpha = min(ALPHA_CLAMP, alpha_raw);
                        if alpha >= ALPHA_CUTOFF {
                            t_acc *= 1.0f32 - alpha;
                        }
                    }
                }
                t += 1u32;
            }
        }
        sync_cube();
        batch_start += TILE_SIZE;
    }

    if active {
        let alpha = 1.0f32 - t_acc;
        let vidx_u = my_vidx as usize;
        if alpha < min_alpha[vidx_u] {
            min_alpha[vidx_u] = alpha;
        }
        if track_color {
            let vis = max(1.0f32 - alpha, 0.0f32);
            let w = f32::powf(vis, COLOR_BLEND_POWER);
            if w > 0.0f32 {
                let pix_idx = (my_py * u.img_w + my_px) as usize;
                let pix_u32 = u32::reinterpret(rendered_image[pix_idx]);
                let r_u = pix_u32 & 0xFFu32;
                let g_u = (pix_u32 >> 8u32) & 0xFFu32;
                let b_u = (pix_u32 >> 16u32) & 0xFFu32;
                let rgb_r = f32::cast_from(r_u) / 255.0f32;
                let rgb_g = f32::cast_from(g_u) / 255.0f32;
                let rgb_b = f32::cast_from(b_u) / 255.0f32;
                let base = (my_vidx * 3u32) as usize;
                color_sum[base] += w * rgb_r;
                color_sum[base + 1] += w * rgb_g;
                color_sum[base + 2] += w * rgb_b;
                weight_sum[vidx_u] += w;
            }
        }
    }
}
