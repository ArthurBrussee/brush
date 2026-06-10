//! Per-splat backward rasterizer.
//!
//! One workgroup per tile, each thread owns one splat from the current
//! batch. Pixel state lives in shared memory and is walked in
//! forward-replay order via diagonal scheduling: at iteration `i`, thread
//! `T` is responsible for `(splat=T, pixel=i-T)`. Each thread accumulates
//! the full gradient for its splat in registers and emits a single atomic
//! add per gradient component per batch.
//!
//! When `geo` is set the kernel additionally backprops the RaDe-GS geometry
//! channels: the blended view-space normal `N` (constant per splat) and the
//! ray-plane depth `t = grad·d + depth_c` (varies per pixel across the
//! footprint). Both alpha-blend with the exact same weights as rgb, so the
//! VJP mirrors the rgb path: value grads scatter into lanes 10..16
//! `(grad_x, grad_y, depth_c, n_x, n_y, n_z)`, and a `dot_geo` term feeds the
//! shared `v_alpha` (full analytic coupling through the weights).
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
    dist_ndc, dist_ndc_deriv, read_projected_geo, read_projected_splat,
};
use brush_render::kernels::types::{RasterizeUniforms, Splat, Sym2, Vec3A};

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
    // Geometry value grads (zero unless `geo`): ray-plane depth gradient
    // (grad_x, grad_y, depth_c) then the blended normal (n_x, n_y, n_z).
    pub gx: f32,
    pub gy: f32,
    pub dc: f32,
    pub n_x: f32,
    pub n_y: f32,
    pub n_z: f32,
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
        gx: 0.0f32,
        gy: 0.0f32,
        dc: 0.0f32,
        n_x: 0.0f32,
        n_y: 0.0f32,
        n_z: 0.0f32,
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

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_backwards_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    projected_geo: &Tensor<f32>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    v_splats: &mut Tensor<Atomic<A::Storage>>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
    #[comptime] geo: bool,
) {
    let (tile_id, tile_origin_x, tile_origin_y) = tile_origin(u.tile_bw);
    // `pix_state` holds the per-pixel running color (and, when geo, normal
    // + depth + distortion-moment) remainders plus transmittance. 4 floats
    // for rgb+T; geo adds (Nx,Ny,Nz,depth) and the mapped-depth moments
    // (S1, S2) the distortion weight-gradient replay walks back.
    let ps = comptime![if geo { 10u32 } else { 4u32 }];
    let mut pix_state = Shared::new_slice((TILE_SIZE * ps) as usize);
    load_pixel_state(output, u, tile_origin_x, tile_origin_y, &mut pix_state, geo);
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
        let (grad_depth, normal) = if comptime![geo] {
            read_projected_geo(projected_geo, compact_gid)
        } else {
            (
                Vec3A::new(0.0f32, 0.0f32, 0.0f32),
                Vec3A::new(0.0f32, 0.0f32, 0.0f32),
            )
        };
        let grad = accumulate_grads_for_batch(
            splat,
            grad_depth,
            normal,
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
            geo,
        );
        if splat_active {
            let stride = comptime![if geo { 16u32 } else { 10u32 }];
            let base = (compact_gid * stride) as usize;
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
            if comptime![geo] {
                A::add(&v_splats[base + 10], grad.gx);
                A::add(&v_splats[base + 11], grad.gy);
                A::add(&v_splats[base + 12], grad.dc);
                A::add(&v_splats[base + 13], grad.n_x);
                A::add(&v_splats[base + 14], grad.n_y);
                A::add(&v_splats[base + 15], grad.n_z);
            }
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
    let mut range_buf = Shared::new_slice(2usize);
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
/// (so subtracting visited splats walks back to zero) and `T=1`. When geo,
/// also seed the (Nx,Ny,Nz,D) remainder from the geometry channels (no bg).
/// Pixels outside the image area get all-zero state — the inner loop's
/// `state_w > 1.0e-4` guard then skips them.
#[cube]
fn load_pixel_state(
    output: &Tensor<f32>,
    u: RasterizeUniforms,
    tile_origin_x: u32,
    tile_origin_y: u32,
    pix_state: &mut Shared<[f32]>,
    #[comptime] geo: bool,
) {
    let ps = comptime![if geo { 10u32 } else { 4u32 }];
    let nchan = comptime![if geo { 11u32 } else { 4u32 }];
    let pixels_per_load = (TILE_SIZE + SPLAT_BATCH - 1u32) / SPLAT_BATCH;
    let mut p = 0u32;
    while p < pixels_per_load {
        let pix_rank = UNIT_POS + p * SPLAT_BATCH;
        if pix_rank < TILE_SIZE {
            let pix_x = tile_origin_x + pix_rank % TILE_WIDTH;
            let pix_y = tile_origin_y + pix_rank / TILE_WIDTH;
            let inside = pix_x < u.img_w && pix_y < u.img_h;
            let s = (pix_rank * ps) as usize;
            if inside {
                let pix_id = pix_x + pix_y * u.img_w;
                let base = (pix_id * nchan) as usize;
                let final_r = output[base];
                let final_g = output[base + 1];
                let final_b = output[base + 2];
                let final_a = output[base + 3];
                let t_final = 1.0f32 - final_a;
                pix_state[s] = final_r - t_final * u.bg_r;
                pix_state[s + 1] = final_g - t_final * u.bg_g;
                pix_state[s + 2] = final_b - t_final * u.bg_b;
                pix_state[s + 3] = 1.0f32;
                if comptime![geo] {
                    // Alpha-blended remainders (normal + depth) plus the
                    // distortion's mapped-depth moment remainders (S1, S2;
                    // channels 9/10). The normalized distortion itself
                    // (chan 8) is recomputed from these, not replayed.
                    pix_state[s + 4] = output[base + 4];
                    pix_state[s + 5] = output[base + 5];
                    pix_state[s + 6] = output[base + 6];
                    pix_state[s + 7] = output[base + 7];
                    pix_state[s + 8] = output[base + 9];
                    pix_state[s + 9] = output[base + 10];
                }
            } else {
                pix_state[s] = 0.0f32;
                pix_state[s + 1] = 0.0f32;
                pix_state[s + 2] = 0.0f32;
                pix_state[s + 3] = 0.0f32;
                if comptime![geo] {
                    pix_state[s + 4] = 0.0f32;
                    pix_state[s + 5] = 0.0f32;
                    pix_state[s + 6] = 0.0f32;
                    pix_state[s + 7] = 0.0f32;
                    pix_state[s + 8] = 0.0f32;
                    pix_state[s + 9] = 0.0f32;
                }
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
    grad_depth: Vec3A,
    normal: Vec3A,
    splat_active: bool,
    tile_origin_x: u32,
    tile_origin_y: u32,
    num_splats_in_tile: u32,
    batch_idx: u32,
    pix_state: &mut Shared<[f32]>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
    #[comptime] geo: bool,
) -> SplatGrad {
    let ps = comptime![if geo { 10u32 } else { 4u32 }];
    let nchan = comptime![if geo { 11u32 } else { 4u32 }];
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
            let s = (pixel_rank * ps) as usize;
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
                        let pix_base = (pix_id * nchan) as usize;
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

                        // Geometry: the blended normal + expected depth blend with
                        // the same weights as rgb, so their value grads scatter and
                        // `dot_geo` couples into v_alpha. The distortion (chan 8,
                        // GOF: raw/(D0^2+eps) with raw = Sum_{i>j} w_i w_j
                        // (m_i-m_j)^2 over NDC-mapped depths) contributes through
                        // three routes: its depth dependence (m_k), its weight
                        // dependence (w_k, via the same remainder pattern as the
                        // blended channels), and the (1-T)^2 normalizer.
                        let mut dot_geo = 0.0f32;
                        if comptime![geo] {
                            let v_n_x = v_output[pix_base + 4];
                            let v_n_y = v_output[pix_base + 5];
                            let v_n_z = v_output[pix_base + 6];
                            let v_d = v_output[pix_base + 7];
                            let v_dist = v_output[pix_base + 8];
                            let t_pix = grad_depth.x() * dx + grad_depth.y() * dy + grad_depth.z();

                            // Distortion pieces. D0 = Sum(w) = final_a, D1/D2 =
                            // mapped-depth moments (channels 9/10), all full-pixel
                            // sums; `vn` folds the detachedly-applied normalizer
                            // into the upstream grad.
                            let d0 = final_a;
                            let d1 = output[pix_base + 9];
                            let d2 = output[pix_base + 10];
                            let norm = 1.0f32 / (d0 * d0 + 1e-7f32);
                            let vn = v_dist * norm;
                            let m = dist_ndc(t_pix);
                            let dm_dt = dist_ndc_deriv(t_pix);

                            // dL/dt = expected-depth value grad `vis*v_d` + the
                            // distortion depth route `draw/dm_k = 2 w_k (m_k D0 -
                            // D1)` + the moment channels' own value grads
                            // (S1/S2 blend m and m^2), all chained through the
                            // NDC mapping. t is linear in the plane params
                            // (∂t/∂grad_x = dx, ∂t/∂grad_y = dy, ∂t/∂depth_c = 1).
                            let v_s1 = v_output[pix_base + 9];
                            let v_s2 = v_output[pix_base + 10];
                            let v_m = 2.0f32 * vis * (m * d0 - d1) * vn
                                + vis * (v_s1 + 2.0f32 * m * v_s2);
                            let g_t = vis * v_d + v_m * dm_dt;
                            grad.gx += g_t * dx;
                            grad.gy += g_t * dy;
                            grad.dc += g_t;
                            // `t` also depends on the splat's 2D position via
                            // `d = mean2d - pixel` (∂t/∂mean2d = (grad_x, grad_y)).
                            grad.xy_x += g_t * grad_depth.x();
                            grad.xy_y += g_t * grad_depth.y();
                            grad.n_x += vis * v_n_x;
                            grad.n_y += vis * v_n_y;
                            grad.n_z += vis * v_n_z;

                            // v_alpha couplings, all in the shared remainder
                            // pattern `(state_w*c_k - S_>=k) * ra`: the blended
                            // channels (normal, depth), the distortion's weight
                            // route with c_k = G_k = draw/dw_k = m^2 D0 + D2 -
                            // 2 m D1 (whose remainder Sum_{i>=k} w_i G_i expands
                            // over the moment remainders), and the normalizer
                            // route with c_k = 1 and dL/dD0 = -2 D0 out norm.
                            let state_nx = pix_state[s + 4];
                            let state_ny = pix_state[s + 5];
                            let state_nz = pix_state[s + 6];
                            let state_d = pix_state[s + 7];
                            let state_s1 = pix_state[s + 8];
                            let state_s2 = pix_state[s + 9];
                            let w_geq = state_w - (1.0f32 - final_a);
                            let g_k = m * m * d0 + d2 - 2.0f32 * m * d1;
                            let r_geq = d0 * state_s2 + d2 * w_geq - 2.0f32 * d1 * state_s1;
                            let out_dist = output[pix_base + 8];
                            let v_d0 = -2.0f32 * d0 * out_dist * vn;
                            dot_geo = ((state_w * normal.x() - state_nx) * v_n_x
                                + (state_w * normal.y() - state_ny) * v_n_y
                                + (state_w * normal.z() - state_nz) * v_n_z
                                + (state_w * t_pix - state_d) * v_d
                                + (state_w * m - state_s1) * v_s1
                                + (state_w * m * m - state_s2) * v_s2
                                + (state_w * g_k - r_geq) * vn
                                + (state_w - w_geq) * v_d0)
                                * ra;
                            pix_state[s + 4] = state_nx - vis * normal.x();
                            pix_state[s + 5] = state_ny - vis * normal.y();
                            pix_state[s + 6] = state_nz - vis * normal.z();
                            pix_state[s + 7] = state_d - vis * t_pix;
                            pix_state[s + 8] = state_s1 - vis * m;
                            pix_state[s + 9] = state_s2 - vis * m * m;
                        }

                        // Chain through the cutoff. Hard step (production):
                        // w' = 0 and w == 1 in-branch, so the factor is 1.
                        let v_alpha_eff = dot_rgb + dot_geo + v_o_w * ra;
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
