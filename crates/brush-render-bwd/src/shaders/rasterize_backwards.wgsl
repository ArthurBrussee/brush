#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> projected: array<vec4f>;
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

const NUM_VECS: u32 = helpers::TILE_SIZE * 3u;
var<workgroup> local_batch: array<vec4f, NUM_VECS>;
var<workgroup> load_gid: array<u32, helpers::TILE_SIZE>;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::CHUNK_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let pix_loc = helpers::map_1d_to_2d(global_id.x, uniforms.tile_bounds.x);
    let pix_id = pix_loc.x + pix_loc.y * uniforms.img_size.x;
    let pixel_coord = vec2f(pix_loc) + 0.5f;
    let tile_loc = vec2u(pix_loc.x / helpers::TILE_WIDTH, pix_loc.y / helpers::TILE_WIDTH);
    let tile_id = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;
    let inside = pix_loc.x < uniforms.img_size.x && pix_loc.y < uniforms.img_size.y;

    let rect_min = subgroupMin(pixel_coord);
    let rect_max = subgroupMax(pixel_coord);
    let sub_rect = vec4f(rect_min.x, rect_min.y, rect_max.x, rect_max.y);

    // final values from forward pass before background blend
    let final_color = output[pix_id];
    let T_final = 1.0f - final_color.a;
    let rgb_pixel_final = final_color.rgb - T_final * uniforms.background.rgb;

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
        tile_offsets[tile_id * 2],
        tile_offsets[tile_id * 2 + 1]
    );
    let num_batches = helpers::ceil_div(range.y - range.x, helpers::CHUNK_SIZE);

    // current visibility left to render
    var T = 1.0f;
    var rgb_pixel = vec3f(0.0f);
    var done = !inside;

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        let batch_start = range.x + b * helpers::CHUNK_SIZE;

        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::CHUNK_SIZE, range.y - batch_start);
        let load_isect_id = batch_start + local_idx;
        let compact_gid = compact_gid_from_isect[load_isect_id];

        workgroupBarrier();
        if local_idx < remaining {
            local_batch[local_idx + helpers::CHUNK_SIZE * 0] = projected[compact_gid + uniforms.total_splats * 0];
            local_batch[local_idx + helpers::CHUNK_SIZE * 1] = projected[compact_gid + uniforms.total_splats * 1];
            local_batch[local_idx + helpers::CHUNK_SIZE * 2] = projected[compact_gid + uniforms.total_splats * 2];
            load_gid[local_idx] = global_from_compact_gid[compact_gid];
        }
        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            var v_xy_local = vec2f(0.0f, 0.0f);
            var v_conic_local = vec3f(0.0f, 0.0f, 0.0f);
            var v_rgb_local = vec3f(0.0f, 0.0f, 0.0f);
            var v_alpha_local = 0.0f;
            var v_refine_local = vec2f(0.0f, 0.0f);

            var hasGrad = false;

            if !done {
                let xy = local_batch[t].xy;
                let conic_alpha = local_batch[t + helpers::CHUNK_SIZE * 1];
                let conic = conic_alpha.xyz;
                let col_alpha = conic_alpha.w;

                let delta = xy - pixel_coord;

                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let gaussian = exp(-sigma);
                let alpha = min(0.999f, col_alpha * gaussian);

                let next_T = T * (1.0f - alpha);

                if next_T <= 1e-4f {
                    done = true;
                } else if sigma >= 0.0f && alpha >= 1.0f / 255.0f {
                    let color = local_batch[t + helpers::CHUNK_SIZE * 2].rgb;

                    let vis = alpha * T;

                    // update v_colors for this gaussian
                    v_rgb_local = select(vec3f(0.0f), vis * v_out.rgb, color >= vec3f(0.0f));

                    // add contribution of this gaussian to the pixel
                    let clamped_rgb = max(color.rgb, vec3f(0.0f));
                    rgb_pixel += vis * clamped_rgb;

                    // Account for alpha being clamped.
                    if (col_alpha * gaussian <= 0.999f) {
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
                        v_alpha_local = alpha * (1.0f - col_alpha) * v_alpha;
                        v_refine_local = abs(v_xy_local);
                    }

                    hasGrad = true;
                    T = next_T;
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

                if subgroup_invocation_id == 0u {
                    let global_gid = load_gid[t];

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
    }
}
