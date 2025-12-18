#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(5) var<storage, read> output: array<vec4f>;
@group(0) @binding(6) var<storage, read> v_output: array<vec4f>;

#ifdef HARD_FLOAT
    @group(0) @binding(7) var<storage, read_write> v_splats: array<atomic<f32>>;
    @group(0) @binding(8) var<storage, read_write> v_opacs: array<atomic<f32>>;
    @group(0) @binding(9) var<storage, read_write> v_refines: array<atomic<f32>>;

    fn write_grads_atomic(id: u32, grads: f32) {
        atomicAdd(&v_splats[id], grads);
    }
    fn write_refine_atomic(id: u32, grads: f32) {
        atomicAdd(&v_refines[id], grads);
    }
    fn write_opac_atomic(id: u32, grads: f32) {
        atomicAdd(&v_opacs[id], grads);
    }
#else
    @group(0) @binding(7) var<storage, read_write> v_splats: array<atomic<u32>>;
    @group(0) @binding(8) var<storage, read_write> v_opacs: array<atomic<u32>>;
    @group(0) @binding(9) var<storage, read_write> v_refines: array<atomic<u32>>;

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
    fn write_refine_atomic(id: u32, grads: f32) {
        var old_value = atomicLoad(&v_refines[id]);
        loop {
            let cas = atomicCompareExchangeWeak(&v_refines[id], old_value, add_bitcast(old_value, grads));
            if cas.exchanged { break; } else { old_value = cas.old_value; }
        }
    }
    fn write_opac_atomic(id: u32, grads: f32) {
        var old_value = atomicLoad(&v_opacs[id]);
        loop {
            let cas = atomicCompareExchangeWeak(&v_opacs[id], old_value, add_bitcast(old_value, grads));
            if cas.exchanged { break; } else { old_value = cas.old_value; }
        }
    }
#endif

const THREAD_COUNT: u32 = 64u;
const PIXELS_PER_THREAD: u32 = 4u;
var<workgroup> local_batch: array<helpers::ProjectedSplat, THREAD_COUNT>;
var<workgroup> load_gid: array<u32, THREAD_COUNT>;

var<workgroup> range_uniform: vec2u;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(THREAD_COUNT, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    var pix_locs = array<vec2u, PIXELS_PER_THREAD>();
    var pix_ids = array<u32, PIXELS_PER_THREAD>();
    var v_outs = array<vec4f, PIXELS_PER_THREAD>();
    var pix_outs = array<vec4f, PIXELS_PER_THREAD>();
    var dones = array<bool, PIXELS_PER_THREAD>();
    var rgb_pixel_finals = array<vec4f, PIXELS_PER_THREAD>();

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        // Process 4 consecutive pixels in the original linear order
        let thread_id = global_id.x * PIXELS_PER_THREAD + i;
        pix_locs[i] = helpers::map_1d_to_2d(thread_id, uniforms.tile_bounds.x);
        let pix_id = pix_locs[i].x + pix_locs[i].y * uniforms.img_size.x;
        pix_ids[i] = pix_id;

        if pix_locs[i].x < uniforms.img_size.x && pix_locs[i].y < uniforms.img_size.y {
            let final_color = output[pix_id];
            let v_out = v_output[pix_id];
            let T_final = 1.0f - final_color.a;
            rgb_pixel_finals[i] = vec4f(final_color.rgb - T_final * uniforms.background.rgb, final_color.a);
            v_outs[i] = vec4f(v_out.rgb, (v_out.a - dot(uniforms.background.rgb, v_out.rgb)) * T_final);
            dones[i] = false;
        } else {
            dones[i] = true;
        }

        pix_outs[i] = vec4f(0.0, 0.0, 0.0, 1.0);
    }

    let tile_loc = vec2u(pix_locs[0].x / helpers::TILE_WIDTH, pix_locs[0].y / helpers::TILE_WIDTH);
    let tile_id = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    range_uniform = vec2u(
        tile_offsets[tile_id * 2],
        tile_offsets[tile_id * 2 + 1],
    );

    // Stupid hack as Chrome isn't convinced the range variable is uniform, which it better be.
    let range = workgroupUniformLoad(&range_uniform);

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixels
    for (var batch_start = range.x; batch_start < range.y; batch_start += THREAD_COUNT) {
        // process gaussians in the current batch for this pixel
        let remaining = min(THREAD_COUNT, range.y - batch_start);
        let load_isect_id = batch_start + local_idx;
        let compact_gid = compact_gid_from_isect[load_isect_id];

        workgroupBarrier();
        if local_idx < remaining {
            load_gid[local_idx] = global_from_compact_gid[compact_gid];
            local_batch[local_idx] = projected[compact_gid];
        }
        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            var v_xy_thread = vec2f(0.0f);
            var v_conic_thread = vec3f(0.0f);
            var v_rgb_thread = vec3f(0.0f);
            var v_alpha_thread = 0.0f;
            var v_refine_thread = 0.0f;
            var hasGrad = false;

            let proj = local_batch[t];

            let xy = vec2f(proj.xy_x, proj.xy_y);
            let conic = vec3f(proj.conic_x, proj.conic_y, proj.conic_z);
            let color = vec4f(proj.color_r, proj.color_g, proj.color_b, proj.color_a);

            let clamped_rgb = max(color.rgb, vec3f(0.0f));

            for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
                if dones[i] { continue; }

                let pixel_coord = vec2f(pix_locs[i]) + 0.5f;
                let delta = xy - pixel_coord;
                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let gaussian = exp(-sigma);
                let alpha = min(0.999f, color.a * gaussian);

                if sigma < 0.0f || alpha < 1.0f / 255.0f {
                    continue;
                }

                let next_T = pix_outs[i].a * (1.0f - alpha);
                if next_T <= 1e-4f {
                    dones[i] = true;
                    continue;
                }

                let vis = alpha * pix_outs[i].a;

                // update v_colors for this gaussian
                let v_rgb_local = select(vec3f(0.0f), vis * v_outs[i].rgb, color.rgb >= vec3f(0.0f));
                v_rgb_thread += v_rgb_local;

                // add contribution of this gaussian to the pixel
                pix_outs[i] = vec4f(pix_outs[i].rgb + vis * clamped_rgb, pix_outs[i].a);

                let ra = 1.0f / (1.0f - alpha);
                let v_alpha = dot(pix_outs[i].a * clamped_rgb + (pix_outs[i].rgb - rgb_pixel_finals[i].rgb) * ra, v_outs[i].rgb) + v_outs[i].a * ra;
                let v_sigma = -alpha * v_alpha;
                let v_xy_local = v_sigma * vec2f(
                    conic.x * delta.x + conic.y * delta.y,
                    conic.y * delta.x + conic.z * delta.y
                );

                // Account for alpha being clamped.
                if (color.a * gaussian <= 0.999f) {
                    v_conic_thread += vec3f(
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    );

                    v_xy_thread += v_xy_local;
                    v_alpha_thread += v_alpha * gaussian;
                    let final_a = max(rgb_pixel_finals[i].a, 1e-5f);
                    // Divide as we don't have sum vis == 1, so reweight these gradients comparatively.
                    v_refine_thread += length(v_xy_local * vec2f(uniforms.img_size.xy)) / final_a;
                }

                hasGrad = true;
                pix_outs[i].a = next_T;
            }

            #ifdef WEBGPU
                let anyGrad = true;
                let doAdd = subgroupAny(hasGrad) && subgroup_invocation_id == 0u;
            #else
                let anyGrad = subgroupAny(hasGrad);
                let doAdd = subgroup_invocation_id == 0u;
            #endif

            // Now do subgroup reduction on thread-accumulated gradients (4x fewer atomics!)
            if anyGrad {
                let sum_xy = subgroupAdd(v_xy_thread);
                let sum_conic = subgroupAdd(v_conic_thread);
                let sum_rgb = subgroupAdd(v_rgb_thread);
                let sum_alpha = subgroupAdd(v_alpha_thread);
                let sum_refine = subgroupAdd(v_refine_thread);

                if doAdd {
                    let global_gid = load_gid[t];

                    write_grads_atomic(global_gid * 8 + 0, sum_xy.x);
                    write_grads_atomic(global_gid * 8 + 1, sum_xy.y);
                    write_grads_atomic(global_gid * 8 + 2, sum_conic.x);
                    write_grads_atomic(global_gid * 8 + 3, sum_conic.y);
                    write_grads_atomic(global_gid * 8 + 4, sum_conic.z);
                    write_grads_atomic(global_gid * 8 + 5, sum_rgb.x);
                    write_grads_atomic(global_gid * 8 + 6, sum_rgb.y);
                    write_grads_atomic(global_gid * 8 + 7, sum_rgb.z);
                    write_opac_atomic(global_gid, sum_alpha);
                    write_refine_atomic(global_gid, sum_refine);
                }
            }
        }
    }
}
