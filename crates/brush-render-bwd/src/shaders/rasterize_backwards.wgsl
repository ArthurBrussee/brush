#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> projected_splats: array<helpers::ProjectedSplat>;
@group(0) @binding(5) var<storage, read> output: array<vec4f>;
@group(0) @binding(6) var<storage, read> v_output: array<vec4f>;

#ifdef HARD_FLOAT
    @group(0) @binding(7) var<storage, read_write> v_splats: array<atomic<f32>>;
    @group(0) @binding(8) var<storage, read_write> v_opacs: array<atomic<f32>>;
    @group(0) @binding(9) var<storage, read_write> v_refines: array<atomic<f32>>;
#else
    @group(0) @binding(7) var<storage, read_write> v_splats: array<atomic<u32>>;
    @group(0) @binding(8) var<storage, read_write> v_opacs: array<atomic<u32>>;
    @group(0) @binding(9) var<storage, read_write> v_refines: array<atomic<u32>>;
#endif

const BATCH_SIZE = helpers::TILE_SIZE;

// Gaussians gathered in batch.
var<workgroup> local_batch: array<helpers::ProjectedSplat, BATCH_SIZE>;
var<workgroup> thread_gid: array<u32, BATCH_SIZE>;

fn add_bitcast(cur: u32, add: f32) -> u32 {
    return bitcast<u32>(bitcast<f32>(cur) + add);
}

fn write_grads_atomic(id: u32, grads: f32) {
    let p = &v_splats[id];
#ifdef HARD_FLOAT
    atomicAdd(p, grads);
#else
    var old_value = atomicLoad(p);
    loop {
        let cas = atomicCompareExchangeWeak(p, old_value, add_bitcast(old_value, grads));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
#endif
}

fn write_refine_atomic(id: u32, grads: f32) {
    let p = &v_refines[id];
#ifdef HARD_FLOAT
    atomicAdd(p, grads);
#else
    var old_value = atomicLoad(p);
    loop {
        let cas = atomicCompareExchangeWeak(p, old_value, add_bitcast(old_value, grads));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
#endif
}

fn write_opac_atomic(id: u32, grads: f32) {
    let p = &v_opacs[id];
#ifdef HARD_FLOAT
    atomicAdd(p, grads);
#else
    var old_value = atomicLoad(p);
    loop {
        let cas = atomicCompareExchangeWeak(p, old_value, add_bitcast(old_value, grads));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
#endif
}

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let img_size = uniforms.img_size;
    let tile_bounds = uniforms.tile_bounds;
    let tile_id = global_id.x / helpers::TILE_SIZE;
    let tile_loc = vec2u(tile_id % tile_bounds.x, tile_id / tile_bounds.x);
    let pixel_coordi = tile_loc * helpers::TILE_WIDTH + vec2u(
        local_idx % helpers::TILE_WIDTH,
        local_idx / helpers::TILE_WIDTH
    );
    let pix_id = pixel_coordi.x + pixel_coordi.y * img_size.x;
    let pixel_coord = vec2f(pixel_coordi) + 0.5f;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = pixel_coordi.x < img_size.x && pixel_coordi.y < img_size.y;

    // final values from forward pass before background blend
    let final_color = output[pix_id];
    let T_final = 1.0f - final_color.a;
    let rgb_pixel_final = vec3f(final_color.rgb) - T_final * uniforms.background.rgb;

    // df/d_out for this pixel
    var v_out = vec4f(0.0f);
    if inside {
        v_out = v_output[pix_id];
    }

    // precompute the gradient from the final alpha of the pixel as far as possible
    v_out.a = (v_out.a - dot(uniforms.background.rgb, v_out.rgb)) * T_final;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    let range = vec2u(tile_offsets[tile_id * 2], tile_offsets[tile_id * 2 + 1]);
    let num_batches = helpers::ceil_div(range.y - range.x, u32(helpers::TILE_SIZE));

    // current visibility left to render
    var T = 1.0f;
    var rgb_pixel = vec3f(0.0f);

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        let batch_start = range.x + b * helpers::TILE_SIZE;

        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        // Each thread first gathers one gaussian.
        if local_idx < remaining {
            let load_isect_id = batch_start + local_idx;
            let load_compact_gid = compact_gid_from_isect[load_isect_id];
            thread_gid[local_idx] = global_from_compact_gid[load_compact_gid];
            local_batch[local_idx] = projected_splats[load_compact_gid];
        }

        // Wait for all threads to finish loading.
        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            var v_xy_local = vec2f(0.0f);
            var v_conic_local = vec3f(0.0f);
            var v_rgb_local = vec3f(0.0f);
            var v_alpha_local = 0.0f;
            var v_refine_local = vec2f(0.0f);

            var hasGrad = false;

            if inside && T > 1e-4f {
                let projected = local_batch[t];
                let xy = vec2f(projected.xy_x, projected.xy_y);
                let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
                let color = vec4f(projected.color_r, projected.color_g, projected.color_b, projected.color_a);

                let delta = xy - pixel_coord;

                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let gaussian = exp(-sigma);
                let alpha = min(0.999f, color.a * gaussian);

                let next_T = T * (1.0f - alpha);

                if sigma >= 0.0f && alpha >= 1.0f / 255.0f && next_T > 1e-4f {
                    let vis = alpha * T;

                    // update v_colors for this gaussian
                    v_rgb_local = select(vec3f(0.0f), vis * v_out.rgb, color.rgb >= vec3f(0.0f));

                    // add contribution of this gaussian to the pixel
                    let clamped_rgb = max(color.rgb, vec3f(0.0f));
                    rgb_pixel += vis * clamped_rgb;

                    // Account for alpha being clamped.
                    if (color.a * gaussian <= 0.999f) {
                        let ra = 1.0f / (1.0f - alpha);

                        let v_alpha = dot(T * clamped_rgb + (rgb_pixel - rgb_pixel_final) * ra, v_out.rgb)
                                    + v_out.a * ra;

                        let v_sigma = -alpha * v_alpha;
                        v_conic_local = vec3f(
                            0.5f * v_sigma * delta.x * delta.x,
                            v_sigma * delta.x * delta.y,
                            0.5f * v_sigma * delta.y * delta.y
                        );
                        v_xy_local = v_sigma * vec2f(
                            conic.x * delta.x + conic.y * delta.y,
                            conic.y * delta.x + conic.z * delta.y
                        );
                        v_alpha_local = alpha * (1.0f - color.a) * v_alpha;
                        v_refine_local = abs(v_xy_local);
                    }

                    T = next_T;
                    hasGrad = true;
                }
            }

            // Note: This isn't uniform control flow according to the WebGPU spec. In practice we know it's fine - the control
            // flow is 100% uniform _for the subgroup_, but that isn't enough and Chrome validation chokes on it.
            if subgroupAny(hasGrad) {
                v_xy_local = subgroupAdd(v_xy_local);
                v_conic_local = subgroupAdd(v_conic_local);
                v_rgb_local = subgroupAdd(v_rgb_local);
                v_alpha_local = subgroupAdd(v_alpha_local);
                v_refine_local = subgroupAdd(v_refine_local);

                let global_gid = thread_gid[t];

                if subgroup_invocation_id == 0u {
                    // Spreading this over threads seems to make no difference.
                    write_grads_atomic(global_gid * 8 + 0, v_xy_local.x);
                    write_grads_atomic(global_gid * 8 + 1, v_xy_local.y);

                    write_grads_atomic(global_gid * 8 + 2, v_conic_local.x);
                    write_grads_atomic(global_gid * 8 + 3, v_conic_local.y);
                    write_grads_atomic(global_gid * 8 + 4, v_conic_local.z);

                    write_grads_atomic(global_gid * 8 + 5, v_rgb_local.x);
                    write_grads_atomic(global_gid * 8 + 6, v_rgb_local.y);
                    write_grads_atomic(global_gid * 8 + 7, v_rgb_local.z);

                    write_opac_atomic(global_gid, v_alpha_local);

                    write_refine_atomic(global_gid * 2 + 0, v_refine_local.x);
                    write_refine_atomic(global_gid * 2 + 1, v_refine_local.y);
                }
            }
        }

        // Wait for all gradients to be written.
        workgroupBarrier();
    }
}
