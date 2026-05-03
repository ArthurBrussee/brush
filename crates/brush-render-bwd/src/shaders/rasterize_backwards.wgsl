enable f16;
#import helpers;

@group(0) @binding(0) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(3) var<storage, read> output: array<vec4f>;
@group(0) @binding(4) var<storage, read> v_output: array<vec4f>;

// Per-splat backward. One workgroup per tile, each thread owns one splat
// from the current batch. Pixel state lives in shared memory and is walked
// in forward-replay order (matching the forward render's accumulation
// formulas) via diagonal scheduling: at iteration `i`, thread `T` is
// responsible for (splat=T, pixel=i-T). Each thread accumulates the full
// gradient for its splat in registers and emits a single atomic add per
// gradient component per batch — instead of the per-subgroup-per-splat
// atomic the per-pixel kernel emits.

// v_splats layout per splat (stride 10, indexed by compact_gid):
// [0..7]: projected splat grads (xy, conic, rgb)
// [8]: opacity grad
// [9]: refine weight
#ifdef HARD_FLOAT
    @group(0) @binding(5) var<storage, read_write> v_splats: array<atomic<f32>>;
    @group(0) @binding(6) var<storage, read> uniforms: helpers::RasterizeUniforms;

    fn write_grads_atomic(id: u32, grads: f32) {
        atomicAdd(&v_splats[id], grads);
    }
#else
    @group(0) @binding(5) var<storage, read_write> v_splats: array<atomic<u32>>;
    @group(0) @binding(6) var<storage, read> uniforms: helpers::RasterizeUniforms;

    fn add_bitcast(cur: u32, add: f32) -> u32 {
        return bitcast<u32>(bitcast<f32>(cur) + add);
    }
    fn write_grads_atomic(id: u32, grads: f32) {
        var old_value = atomicLoad(&v_splats[id]);
        loop {
            let cas = atomicCompareExchangeWeak(&v_splats[id], old_value, add_bitcast(old_value, grads));
            if cas.exchanged { break; } else { old_value = cas.old_value; }
        }
    }
#endif

// SPLAT_BATCH = 32 = one Apple-Silicon SIMD group, so the per-iter
// workgroupBarrier collapses to a SIMD-lockstep no-op on hardware. Larger
// (64/128/256) crossed SIMD groups and paid full barrier cost; smaller
// (16) wasted half the SIMD lanes.
const SPLAT_BATCH: u32 = 32u;
const TILE_PIXELS: u32 = helpers::TILE_SIZE;
const PIXELS_PER_LOAD: u32 = (TILE_PIXELS + SPLAT_BATCH - 1u) / SPLAT_BATCH;

// Forward-replay pixel state. We store `remain = (final_rgb - current_rgb)`
// in xyz and the running transmittance in w, so the gradient formula
// can use `remain` directly without a separate `pix_final` array. Each
// splat update does `remain -= vis * splat_rgb`. Halving shared memory
// (vs storing rgb_accum + final separately) lets more workgroups live on
// each Apple GPU SM. f16 here costs ~1e-3 relative precision but the
// values stay in [0, ~1] and the reference gradient test still passes.
//
// w <= 1e-4 is the "done" sentinel — once a pixel saturates we leave its
// remain alone and further splats skip it.
var<workgroup> pix_state: array<vec4<f16>, TILE_PIXELS>;
// Upstream gradient (post-background-subtraction) baked at load time.
// f16 because the gradient values come from L1/L2 image losses which sit
// at modest magnitudes — half precision gives ~1e-3 relative which is
// fine for the per-iter multiplications, and halving the array (from
// 4KB to 2KB) frees a workgroup-occupancy slot on Apple GPUs.
var<workgroup> pix_v_out: array<vec4<f16>, TILE_PIXELS>;
// Per-pixel `1 / max(final.alpha, 1e-5)` precomputed. The refine-weight
// gradient divides by final.alpha; precomputing the reciprocal turns the
// per-iter division into a multiplication. f16 because the only consumer
// (v_refine accumulator) is itself a per-tile-densification heuristic.
var<workgroup> pix_inv_final_a: array<f16, TILE_PIXELS>;

var<workgroup> range_uniform: vec2u;

@compute
@workgroup_size(SPLAT_BATCH, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) thread_rank: u32,
) {
    let tile_id = helpers::get_workgroup_id(wg_id, num_wgs);
    let tile_loc = vec2u(tile_id % uniforms.tile_bounds.x, tile_id / uniforms.tile_bounds.x);
    let tile_origin = tile_loc * helpers::TILE_WIDTH;

    // Each thread loads PIXELS_PER_LOAD pixels' state. Linear pixel order
    // within the tile (matches vksplat's layout — diagonal scheduling
    // doesn't benefit from Morton ordering since pixels are accessed in
    // straight-line iteration order).
    for (var p = 0u; p < PIXELS_PER_LOAD; p++) {
        let pix_rank = thread_rank + p * SPLAT_BATCH;
        if pix_rank >= TILE_PIXELS { break; }
        let local_xy = vec2u(pix_rank % helpers::TILE_WIDTH, pix_rank / helpers::TILE_WIDTH);
        let pix_loc = tile_origin + local_xy;
        let inside = pix_loc.x < uniforms.img_size.x && pix_loc.y < uniforms.img_size.y;

        if inside {
            let pix_id = pix_loc.x + pix_loc.y * uniforms.img_size.x;
            let final_color = output[pix_id];
            let v_out = v_output[pix_id];
            let T_final = 1.0 - final_color.a;
            pix_state[pix_rank] = vec4<f16>(vec4f(final_color.rgb - T_final * uniforms.background.rgb, 1.0));
            pix_v_out[pix_rank] = vec4<f16>(vec4f(v_out.rgb, (v_out.a - dot(uniforms.background.rgb, v_out.rgb)) * T_final));
            pix_inv_final_a[pix_rank] = f16(1.0 / max(final_color.a, 1e-5));
        } else {
            pix_state[pix_rank] = vec4<f16>(0.0);
            pix_v_out[pix_rank] = vec4<f16>(0.0);
            pix_inv_final_a[pix_rank] = 0.0h;
        }
    }

    if thread_rank == 0u {
        range_uniform = vec2u(tile_offsets[tile_id * 2u], tile_offsets[tile_id * 2u + 1u]);
    }
    workgroupBarrier();
    let range = workgroupUniformLoad(&range_uniform);
    let num_splats_in_tile = range.y - range.x;
    let rounds = (num_splats_in_tile + SPLAT_BATCH - 1u) / SPLAT_BATCH;

    for (var batch_idx = 0u; batch_idx < rounds; batch_idx++) {
        // Inter-batch early-exit is redundant — the forward kernel
        // already shrinks tile_offsets[tile*2+1] to the last splat any
        // pixel actually consumed, so `num_splats_in_tile` is the tight
        // upper bound and the outer `for` doesn't need a saturation
        // gate.
        // Thread T owns the T-th splat in the batch (forward order — pixel
        // 0 sees splat 0 first, then splat 1, etc.).
        let splat_offset = batch_idx * SPLAT_BATCH + thread_rank;
        var splat: helpers::ProjectedSplat;
        var compact_gid: u32 = 0u;
        var splat_active = false;

        if splat_offset < num_splats_in_tile {
            compact_gid = compact_gid_from_isect[range.x + splat_offset];
            splat = projected[compact_gid];
            splat_active = true;
        }

        let num_splats_this_batch = min(SPLAT_BATCH, num_splats_in_tile - batch_idx * SPLAT_BATCH);
        let total_iters = num_splats_this_batch + TILE_PIXELS - 1u;

        // Per-splat gradient accumulators in registers.
        var v_xy = vec2f(0.0);
        var v_conic = vec3f(0.0);
        var v_rgb_grad = vec3f(0.0);
        var v_alpha_acc = 0.0;
        var v_refine = 0.0;

        let xy = vec2f(splat.xy_x, splat.xy_y);
        let conic = vec3f(splat.conic_x, splat.conic_y, splat.conic_z);
        let opac = splat.color_a;
        let raw_rgb = vec3f(f32(splat.color_r), f32(splat.color_g), f32(splat.color_b));
        let clamped_rgb = max(raw_rgb, vec3f(0.0));

        // Diagonal walk. At iter `i`, thread T processes (its splat,
        // pixel = i - T). Each iter ends with a workgroupBarrier so the
        // write to pix_color from this iter is visible to the thread that
        // picks up the same pixel on the next diagonal.
        for (var i = 0u; i < total_iters; i++) {
            let active_iter =
                splat_active &&
                i >= thread_rank &&
                (i - thread_rank) < TILE_PIXELS;

            if active_iter {
                let pixel_rank = i - thread_rank;
                // pix_state.xyz = remain (final_rgb - current_rgb so far),
                // pix_state.w = T (transmittance). Saturated pixels carry T<=1e-4.
                let state = vec4f(pix_state[pixel_rank]);

                if state.w > 1e-4 {
                    let local_xy = vec2u(pixel_rank & 15u, pixel_rank >> 4u);
                    let pix_loc = tile_origin + local_xy;
                    let pixel_coord = vec2f(pix_loc) + 0.5;
                    let delta = xy - pixel_coord;
                    let sigma = 0.5 * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                    let gaussian = exp(-sigma);
                    let alpha = min(0.999, opac * gaussian);

                    if sigma >= 0.0 && alpha >= 1.0 / 255.0 {
                        let next_T = state.w * (1.0 - alpha);
                        if next_T <= 1e-4 {
                            pix_state[pixel_rank] = vec4<f16>(vec4f(state.xyz, 0.0));
                        } else {
                            let vis = alpha * state.w;
                            let v_out_local = vec4f(pix_v_out[pixel_rank]);
                            // new_remain = remain - vis * splat_rgb. Keep
                            // full negative form (remain - delta) so the
                            // gradient term is `-new_remain * ra`.
                            let new_remain = state.xyz - vis * clamped_rgb;

                            let v_rgb_local = select(vec3f(0.0), vis * v_out_local.rgb, raw_rgb >= vec3f(0.0));
                            v_rgb_grad += v_rgb_local;

                            let ra = 1.0 / (1.0 - alpha);
                            // (new_rgb - final.rgb) == -new_remain.
                            let v_alpha = dot(state.w * clamped_rgb - new_remain * ra, v_out_local.rgb) + v_out_local.a * ra;
                            let v_sigma = -alpha * v_alpha;
                            let v_xy_local = v_sigma * vec2f(
                                conic.x * delta.x + conic.y * delta.y,
                                conic.y * delta.x + conic.z * delta.y
                            );

                            if opac * gaussian <= 0.999 {
                                v_conic += vec3f(
                                    0.5 * v_sigma * delta.x * delta.x,
                                    v_sigma * delta.x * delta.y,
                                    0.5 * v_sigma * delta.y * delta.y,
                                );
                                v_xy += v_xy_local;
                                v_alpha_acc += v_alpha * gaussian;
                                v_refine += length(v_xy_local * vec2f(uniforms.img_size.xy)) * f32(pix_inv_final_a[pixel_rank]);
                            }

                            pix_state[pixel_rank] = vec4<f16>(vec4f(new_remain, next_T));
                        }
                    }
                }
            }

            workgroupBarrier();
        }

        if splat_active {
            let base = compact_gid * 10u;
            write_grads_atomic(base + 0u, v_xy.x);
            write_grads_atomic(base + 1u, v_xy.y);
            write_grads_atomic(base + 2u, v_conic.x);
            write_grads_atomic(base + 3u, v_conic.y);
            write_grads_atomic(base + 4u, v_conic.z);
            write_grads_atomic(base + 5u, v_rgb_grad.x);
            write_grads_atomic(base + 6u, v_rgb_grad.y);
            write_grads_atomic(base + 7u, v_rgb_grad.z);
            write_grads_atomic(base + 8u, v_alpha_acc);
            write_grads_atomic(base + 9u, v_refine);
        }
    }
}
