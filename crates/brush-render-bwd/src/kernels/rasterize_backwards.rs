//! Per-splat backward rasterizer.
//!
//! One workgroup per tile, each thread owns one splat from the current
//! batch. Pixel state lives in shared memory and is walked in
//! forward-replay order via diagonal scheduling: at iteration `i`, thread
//! `T` is responsible for `(splat=T, pixel=i-T)`. Each thread accumulates
//! the full gradient for its splat in registers and emits a single atomic
//! add per gradient component per batch.
//!
//! The atomic accumulation is parametrised by the [`AtomicAddF32`] trait:
//! `HfAtomicAdd` (native `Atomic<f32>::fetch_add`) when the device
//! supports it, `CasAtomicAdd` (`Atomic<u32>` + CAS over the bit pattern)
//! otherwise. The host picks the impl based on `AtomicUsage::Add`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use brush_render::kernels::helpers::{
    ALPHA_CUTOFF_MID, TILE_SIZE, TILE_WIDTH, alpha_cutoff_weight, alpha_cutoff_weight_deriv,
    read_projected_splat,
};
use brush_render::kernels::types::{RasterizeUniforms, Splat, Sym2};

// SPLAT_BATCH = 32 = one Apple-Silicon SIMD group, so the per-iter
// sync_cube collapses to a SIMD-lockstep no-op on hardware.
pub const SPLAT_BATCH: u32 = 32;

/// Per-splat gradient accumulator for the rasterize backward.
#[derive(CubeType, Copy, Clone)]
pub struct SplatGrad {
    pub xy_x: f32,
    pub xy_y: f32,
    pub conic_x: f32,
    pub conic_y: f32,
    pub conic_z: f32,
    pub rgb_r: f32,
    pub rgb_g: f32,
    pub rgb_b: f32,
    pub alpha: f32,
    pub refine: f32,
}

#[cube]
fn zero_grad() -> SplatGrad {
    SplatGrad {
        xy_x: 0.0f32,
        xy_y: 0.0f32,
        conic_x: 0.0f32,
        conic_y: 0.0f32,
        conic_z: 0.0f32,
        rgb_r: 0.0f32,
        rgb_g: 0.0f32,
        rgb_b: 0.0f32,
        alpha: 0.0f32,
        refine: 0.0f32,
    }
}

/// f32-atomic-add abstraction so a single kernel covers both the native
/// `Atomic<f32>::fetch_add` path and the `Atomic<u32>` CAS fallback.
#[cube]
pub trait AtomicAddF32: Send + Sync + 'static {
    type Storage: Numeric;
    fn add(target: &Atomic<Self::Storage>, val: f32);
}

#[derive(CubeType)]
pub struct HfAtomicAdd;

#[derive(CubeType)]
pub struct CasAtomicAdd;

#[cube]
impl AtomicAddF32 for HfAtomicAdd {
    type Storage = f32;
    fn add(target: &Atomic<f32>, val: f32) {
        Atomic::fetch_add(target, val);
    }
}

#[cube]
impl AtomicAddF32 for CasAtomicAdd {
    type Storage = u32;
    fn add(target: &Atomic<u32>, val: f32) {
        let mut old_value = Atomic::load(target);
        let mut done = false;
        while !done {
            let new_bits = u32::reinterpret(f32::reinterpret(old_value) + val);
            let actual = Atomic::compare_exchange_weak(target, old_value, new_bits);
            if actual == old_value {
                done = true;
            } else {
                old_value = actual;
            }
        }
    }
}

#[cube(launch_unchecked)]
pub fn rasterize_backwards_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    v_splats: &mut Tensor<Atomic<A::Storage>>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
) {
    let (tile_id, tile_origin_x, tile_origin_y) = tile_origin(u.tile_bw);
    // Only `pix_state` lives in shared memory — it gets read-modify-
    // written each iteration (alpha decay) so threads need to see each
    // other's writes. The other per-pixel inputs (`v_output`, the alpha
    // pre-roll) are read-only post-init and L1-cached, so we re-derive
    // them inline in the inner loop. Smaller shared footprint → more
    // workgroup occupancy on Apple.
    let mut pix_state = SharedMemory::<f32>::new((TILE_SIZE * 4u32) as usize);
    load_pixel_state(output, u, tile_origin_x, tile_origin_y, &mut pix_state);
    let (range_lo, range_hi) = load_range(tile_offsets, tile_id);
    let num_splats_in_tile = range_hi - range_lo;
    let rounds = (num_splats_in_tile + SPLAT_BATCH - 1u32) / SPLAT_BATCH;

    let mut batch_idx = 0u32;
    while batch_idx < rounds {
        let (compact_gid, splat, splat_active) = load_splat_for_batch(
            compact_gid_from_isect,
            projected,
            range_lo,
            num_splats_in_tile,
            batch_idx,
        );
        let grad = accumulate_grads_for_batch(
            splat,
            splat_active,
            tile_origin_x,
            tile_origin_y,
            num_splats_in_tile,
            batch_idx,
            &mut pix_state,
            output,
            v_output,
            u,
            smooth_cutoff,
        );
        if splat_active {
            let base = (compact_gid * 10u32) as usize;
            A::add(&v_splats[base], grad.xy_x);
            A::add(&v_splats[base + 1], grad.xy_y);
            A::add(&v_splats[base + 2], grad.conic_x);
            A::add(&v_splats[base + 3], grad.conic_y);
            A::add(&v_splats[base + 4], grad.conic_z);
            A::add(&v_splats[base + 5], grad.rgb_r);
            A::add(&v_splats[base + 6], grad.rgb_g);
            A::add(&v_splats[base + 7], grad.rgb_b);
            A::add(&v_splats[base + 8], grad.alpha);
            A::add(&v_splats[base + 9], grad.refine);
        }
        batch_idx += 1u32;
    }
}

#[cube]
fn tile_origin(tile_bw: u32) -> (u32, u32, u32) {
    let tile_id = CUBE_POS as u32;
    let tile_origin_x = (tile_id % tile_bw) * TILE_WIDTH;
    let tile_origin_y = (tile_id / tile_bw) * TILE_WIDTH;
    (tile_id, tile_origin_x, tile_origin_y)
}

#[cube]
fn load_range(tile_offsets: &Tensor<u32>, tile_id: u32) -> (u32, u32) {
    let mut range_buf = SharedMemory::<u32>::new(2usize);
    if UNIT_POS == 0u32 {
        range_buf[0] = tile_offsets[(tile_id * 2u32) as usize];
        range_buf[1] = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
    }
    // Uniform-marked loads so loop bounds derived from these don't trip
    // WebGPU's "barrier in non-uniform control flow" check.
    (
        workgroup_uniform_load(&range_buf[0]),
        workgroup_uniform_load(&range_buf[1]),
    )
}

/// Seed `pix_state` with the post-rasterise RGB minus the bg pre-roll
/// (so subtracting visited splats walks back to zero) and `T=1`. Pixels
/// outside the image area get all-zero state — the inner loop's
/// `state_w > 1.0e-4` guard then skips them.
#[cube]
fn load_pixel_state(
    output: &Tensor<f32>,
    u: RasterizeUniforms,
    tile_origin_x: u32,
    tile_origin_y: u32,
    pix_state: &mut SharedMemory<f32>,
) {
    let pixels_per_load = (TILE_SIZE + SPLAT_BATCH - 1u32) / SPLAT_BATCH;
    let mut p = 0u32;
    while p < pixels_per_load {
        let pix_rank = UNIT_POS + p * SPLAT_BATCH;
        if pix_rank < TILE_SIZE {
            let pix_x = tile_origin_x + pix_rank % TILE_WIDTH;
            let pix_y = tile_origin_y + pix_rank / TILE_WIDTH;
            let inside = pix_x < u.img_w && pix_y < u.img_h;
            let s = (pix_rank * 4u32) as usize;
            if inside {
                let pix_id = pix_x + pix_y * u.img_w;
                let base = (pix_id * 4u32) as usize;
                let final_r = output[base];
                let final_g = output[base + 1];
                let final_b = output[base + 2];
                let final_a = output[base + 3];
                let t_final = 1.0f32 - final_a;
                pix_state[s] = final_r - t_final * u.bg_r;
                pix_state[s + 1] = final_g - t_final * u.bg_g;
                pix_state[s + 2] = final_b - t_final * u.bg_b;
                pix_state[s + 3] = 1.0f32;
            } else {
                pix_state[s] = 0.0f32;
                pix_state[s + 1] = 0.0f32;
                pix_state[s + 2] = 0.0f32;
                pix_state[s + 3] = 0.0f32;
            }
        }
        p += 1u32;
    }
}

#[cube]
fn load_splat_for_batch(
    compact_gid_from_isect: &Tensor<u32>,
    projected: &Tensor<f32>,
    range_lo: u32,
    num_splats_in_tile: u32,
    batch_idx: u32,
) -> (u32, Splat, bool) {
    let splat_offset = batch_idx * SPLAT_BATCH + UNIT_POS;
    let mut compact_gid = 0u32;
    let mut splat = Splat::zero();
    let mut splat_active = false;
    if splat_offset < num_splats_in_tile {
        compact_gid = compact_gid_from_isect[(range_lo + splat_offset) as usize];
        splat = read_projected_splat(projected, compact_gid);
        splat_active = true;
    }
    (compact_gid, splat, splat_active)
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn accumulate_grads_for_batch(
    splat: Splat,
    splat_active: bool,
    tile_origin_x: u32,
    tile_origin_y: u32,
    num_splats_in_tile: u32,
    batch_idx: u32,
    pix_state: &mut SharedMemory<f32>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
) -> SplatGrad {
    let conic = Sym2 {
        c00: splat.conic_x,
        c01: splat.conic_y,
        c11: splat.conic_z,
    };
    let clamped_r = max(splat.color_r, 0.0f32);
    let clamped_g = max(splat.color_g, 0.0f32);
    let clamped_b = max(splat.color_b, 0.0f32);

    let num_splats_this_batch = min(SPLAT_BATCH, num_splats_in_tile - batch_idx * SPLAT_BATCH);
    let total_iters = num_splats_this_batch + TILE_SIZE - 1u32;

    let mut grad = zero_grad();

    let mut i = 0u32;
    while i < total_iters {
        let active_iter = splat_active && i >= UNIT_POS && (i - UNIT_POS) < TILE_SIZE;

        if active_iter {
            let pixel_rank = i - UNIT_POS;
            let s = (pixel_rank * 4u32) as usize;
            let state_x = pix_state[s];
            let state_y = pix_state[s + 1];
            let state_z = pix_state[s + 2];
            let state_w = pix_state[s + 3];

            if state_w > 1.0e-4f32 {
                let pix_x = tile_origin_x + pixel_rank % TILE_WIDTH;
                let pix_y = tile_origin_y + pixel_rank / TILE_WIDTH;
                let pixel_coord_x = pix_x as f32 + 0.5f32;
                let pixel_coord_y = pix_y as f32 + 0.5f32;
                let dx = splat.xy_x - pixel_coord_x;
                let dy = splat.xy_y - pixel_coord_y;
                let sigma =
                    0.5f32 * (conic.c00 * dx * dx + conic.c11 * dy * dy) + conic.c01 * dx * dy;
                let gaussian = f32::exp(-sigma);
                let alpha = min(0.999f32, splat.color_a * gaussian);

                let w_cut = if comptime![smooth_cutoff] {
                    alpha_cutoff_weight(alpha)
                } else {
                    select(alpha >= ALPHA_CUTOFF_MID, 1.0f32, 0.0f32)
                };
                if sigma >= 0.0f32 && w_cut > 0.0f32 {
                    let alpha_eff = alpha * w_cut;
                    let next_t = state_w * (1.0f32 - alpha_eff);
                    if next_t <= 1.0e-4f32 {
                        pix_state[s + 3] = 0.0f32;
                    } else {
                        let vis = alpha_eff * state_w;
                        // Re-derive v_out and inv_final_a from `v_output` /
                        // `output` directly. These reads hit the global
                        // tensor each iter rather than shared memory, but
                        // they're L1-cached and only touched on the
                        // not-fully-transparent path. Trades a few global
                        // loads for ~5 KiB of shared memory back, which
                        // recovers an Apple-GPU occupancy slot.
                        let pix_id = pix_x + pix_y * u.img_w;
                        let pix_base = (pix_id * 4u32) as usize;
                        let v_o_x = v_output[pix_base];
                        let v_o_y = v_output[pix_base + 1];
                        let v_o_z = v_output[pix_base + 2];
                        let v_a = v_output[pix_base + 3];
                        let final_a = output[pix_base + 3];
                        let t_final = 1.0f32 - final_a;
                        let v_o_w =
                            (v_a - (u.bg_r * v_o_x + u.bg_g * v_o_y + u.bg_b * v_o_z)) * t_final;
                        // Gate the rgb VJP on the original (pre-clamp) sign:
                        // negative raw values clamp to zero and contribute
                        // no gradient.
                        grad.rgb_r += select(splat.color_r >= 0.0f32, vis * v_o_x, 0.0f32);
                        grad.rgb_g += select(splat.color_g >= 0.0f32, vis * v_o_y, 0.0f32);
                        grad.rgb_b += select(splat.color_b >= 0.0f32, vis * v_o_z, 0.0f32);

                        let ra = 1.0f32 / (1.0f32 - alpha_eff);
                        let dot_rgb = ((state_w * clamped_r - state_x) * v_o_x
                            + (state_w * clamped_g - state_y) * v_o_y
                            + (state_w * clamped_b - state_z) * v_o_z)
                            * ra;
                        let new_remain_x = state_x - vis * clamped_r;
                        let new_remain_y = state_y - vis * clamped_g;
                        let new_remain_z = state_z - vis * clamped_b;
                        // Chain through the cutoff. Hard step (production):
                        // w' = 0 and w == 1 in-branch, so the factor is 1.
                        let v_alpha_eff = dot_rgb + v_o_w * ra;
                        let dw_dalpha = if comptime![smooth_cutoff] {
                            alpha_cutoff_weight_deriv(alpha)
                        } else {
                            0.0f32 * alpha
                        };
                        let v_alpha = v_alpha_eff * (w_cut + alpha * dw_dalpha);
                        let v_sigma = -alpha * v_alpha;
                        let vxy_x = v_sigma * (conic.c00 * dx + conic.c01 * dy);
                        let vxy_y = v_sigma * (conic.c01 * dx + conic.c11 * dy);

                        // Suppress the alpha-saturated gradient term — at the
                        // cap the alpha derivative discontinuously flattens.
                        if splat.color_a * gaussian <= 0.999f32 {
                            grad.conic_x += 0.5f32 * v_sigma * dx * dx;
                            grad.conic_y += v_sigma * dx * dy;
                            grad.conic_z += 0.5f32 * v_sigma * dy * dy;
                            grad.xy_x += vxy_x;
                            grad.xy_y += vxy_y;
                            grad.alpha += v_alpha * gaussian;
                            let img_size_x = u.img_w as f32;
                            let img_size_y = u.img_h as f32;
                            let len = f32::sqrt(
                                vxy_x * img_size_x * vxy_x * img_size_x
                                    + vxy_y * img_size_y * vxy_y * img_size_y,
                            );
                            grad.refine += len / max(final_a, 1.0e-5f32);
                        }

                        pix_state[s] = new_remain_x;
                        pix_state[s + 1] = new_remain_y;
                        pix_state[s + 2] = new_remain_z;
                        pix_state[s + 3] = next_t;
                    }
                }
            }
        }

        sync_cube();
        i += 1u32;
    }
    grad
}
