#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<i32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<i32>;

@group(0) @binding(3) var<storage, read> projected_splats: array<helpers::ProjectedSplat>;

@group(0) @binding(4) var<storage, read> output: array<vec4f>;
@group(0) @binding(5) var<storage, read> v_output: array<vec4f>;

#ifdef HARD_FLOAT
    @group(0) @binding(6) var<storage, read_write> v_splats: array<atomic<f32>>;
    @group(0) @binding(7) var<storage, read_write> v_refine_grad: array<atomic<f32>>;
#else
    @group(0) @binding(6) var<storage, read_write> v_splats: array<atomic<u32>>;
    @group(0) @binding(7) var<storage, read_write> v_refine_grad: array<atomic<u32>>;
#endif

const BATCH_SIZE = helpers::TILE_SIZE;

// Gaussians gathered in batch.
var<workgroup> local_batch: array<helpers::ProjectedSplat, BATCH_SIZE>;
var<workgroup> local_id: array<u32, BATCH_SIZE>;

var<workgroup> done_count: atomic<u32>;
var<workgroup> done_count_uniform: u32;

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
    let p = &v_refine_grad[id];
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
@workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let img_size = uniforms.img_size;
    let tile_bounds = uniforms.tile_bounds;

    let tile_id = workgroup_id.x;

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
    var done = !inside;

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
    let range = vec2u(
        u32(clamp(tile_offsets[tile_id], 0, i32(uniforms.max_intersects))),
        u32(clamp(tile_offsets[tile_id + 1], 0, i32(uniforms.max_intersects)))
    );
    let num_batches = helpers::ceil_div(range.y - range.x, u32(helpers::TILE_SIZE));

    // current visibility left to render
    var T = 1.0f;
    var rgb_pixel = vec3f(0.0f);

    atomicStore(&done_count, 0u);

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        let batch_start = range.x + b * helpers::TILE_SIZE;

        // Wait for all in flight threads and check whether we're all done.
        //
        // HACK: Annoyingly workgroupUniformLoad doesn't work for atomics...
        done_count_uniform = atomicLoad(&done_count);
        if workgroupUniformLoad(&done_count_uniform) >= helpers::TILE_SIZE {
            break;
        }

        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        // Each thread first gathers one gaussian.
        if local_idx < remaining {
            let load_isect_id = batch_start + local_idx;
            let load_compact_gid = u32(compact_gid_from_isect[load_isect_id]);
            local_id[local_idx] = load_compact_gid;
            local_batch[local_idx] = projected_splats[load_compact_gid];
        }

        // Wait for all threads to finish loading.
        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            var valid = inside && !done;

            var alpha: f32;
            var color: vec4f;
            var gaussian: f32;
            var delta: vec2f;
            var conic: vec3f;

            if valid {
                let projected = local_batch[t];
                let xy = vec2f(projected.xy_x, projected.xy_y);
                conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
                color = vec4f(projected.color_r, projected.color_g, projected.color_b, projected.color_a);

                delta = xy - pixel_coord;

                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                gaussian = exp(-sigma);
                alpha = min(0.999f, color.a * gaussian);

                if (sigma < 0.0f || alpha < 1.0f / 255.0f) {
                    valid = false;
                }
            }

            var next_T = T;
            if valid {
                next_T = T * (1.0f - alpha);
                if next_T <= 1e-4f {
                    atomicAdd(&done_count, 1u);
                    done = true;
                    valid = false;
                }
            }

            var v_xy_local = vec2f(0.0f);
            var v_conic_local = vec3f(0.0f);
            var v_rgb_local = vec3f(0.0f);
            var v_alpha_local = 0.0f;
            var v_refine_local = vec2f(0.0f);

            if valid {
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

                // update transmittance
                T = next_T;
            }

            let v_xy_sum = subgroupAdd(v_xy_local);
            let v_conic_sum = subgroupAdd(v_conic_local);
            let v_colors_sum = subgroupAdd(vec4f(v_rgb_local, v_alpha_local));
            let v_refine_sum = subgroupAdd(v_refine_local);

            // Note: We can't move this earlier in the file as the subgroupAdd has to be on
            // uniform control flow according to the WebGPU spec. In practice we know it's fine - the control
            // flow is 100% uniform _for the subgroup_, but that isn't enough and Chrome validation chokes on it.
            if subgroupAny(valid) {
                // Queue a new gradient if this subgroup has any.
                // The gradient is sum of all gradients in the subgroup.
                let compact_gid = local_id[t];
                switch subgroup_invocation_id {
                    case 0u:  { write_grads_atomic(compact_gid * 9 + 0, v_xy_sum.x); }
                    case 1u:  { write_grads_atomic(compact_gid * 9 + 1, v_xy_sum.y); }
                    case 2u:  { write_grads_atomic(compact_gid * 9 + 2, v_conic_sum.x); }
                    case 3u:  { write_grads_atomic(compact_gid * 9 + 3, v_conic_sum.y); }
                    case 4u:  { write_grads_atomic(compact_gid * 9 + 4, v_conic_sum.z); }
                    case 5u:  { write_grads_atomic(compact_gid * 9 + 5, v_colors_sum.x); }
                    case 6u:  { write_grads_atomic(compact_gid * 9 + 6, v_colors_sum.y); }
                    case 7u:  {
                        write_grads_atomic(compact_gid * 9 + 7, v_colors_sum.z);

                        // Subgroups of size 8 need to be handled separately as there's not enough threads to write
                        // all the gaussian fields. The next size (16) is fine.
                        if subgroup_size == 8u {
                            write_grads_atomic(compact_gid * 9 + 8, v_colors_sum.w);
                            write_refine_atomic(compact_gid * 2 + 0, v_refine_sum.x);
                            write_refine_atomic(compact_gid * 2 + 1, v_refine_sum.y);
                        }
                    }
                    case 8u:  { write_grads_atomic(compact_gid * 9 + 8, v_colors_sum.w); }
                    case 9u:  { write_refine_atomic(compact_gid * 2 + 0, v_refine_sum.x); }
                    case 10u: { write_refine_atomic(compact_gid * 2 + 1, v_refine_sum.y); }
                    default: {}
                }
            }
        }

        // Wait for all gradients to be written.
        workgroupBarrier();
    }
}
