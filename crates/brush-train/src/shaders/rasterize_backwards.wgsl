#import helpers;

@group(0) @binding(0) var<uniform> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<i32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<i32>;

@group(0) @binding(3) var<storage, read> projected_splats: array<helpers::ProjectedSplat>;

@group(0) @binding(4) var<storage, read> final_index: array<i32>;
@group(0) @binding(5) var<storage, read> output: array<vec4f>;
@group(0) @binding(6) var<storage, read> v_output: array<vec4f>;

#ifdef HARD_FLOAT
    @group(0) @binding(7) var<storage, read_write> v_splats: array<atomic<f32>>;
    @group(0) @binding(8) var<storage, read_write> v_refine_grad: array<atomic<f32>>;
#else
    @group(0) @binding(7) var<storage, read_write> v_splats: array<atomic<u32>>;
    @group(0) @binding(8) var<storage, read_write> v_refine_grad: array<atomic<u32>>;
#endif

const BATCH_SIZE = helpers::TILE_SIZE;

// Gaussians gathered in batch.
var<workgroup> local_batch: array<helpers::ProjectedSplat, BATCH_SIZE>;
var<workgroup> local_id: array<i32, BATCH_SIZE>;

fn add_bitcast(cur: u32, add: f32) -> u32 {
    return bitcast<u32>(bitcast<f32>(cur) + add);
}

fn write_grads_atomic(id: i32, grads: f32) {
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

fn write_refine_atomic(id: i32, grads: f32) {
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
    let pixel_coord = vec2f(pixel_coordi) + 0.5;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = pixel_coordi.x < img_size.x && pixel_coordi.y < img_size.y;

    // this is the T AFTER the last gaussian in this pixel
    let T_final = 1.0 - output[pix_id].w;

    // Have all threads in tile process the same gaussians in batches
    // first collect gaussians between bin_start and bin_final in batches
    // which gaussians to look through in this tile
    let range = vec2u(u32(tile_offsets[tile_id]), u32(tile_offsets[tile_id + 1]));

    let num_batches = helpers::ceil_div(range.y - range.x, BATCH_SIZE);

    // current visibility left to render
    var T = T_final;

    var final_isect = 0;
    var buffer = vec3f(0.0);

    if inside {
        final_isect = final_index[pix_id];
    }

    // df/d_out for this pixel
    var v_out = vec4f(0.0);
    if inside {
        v_out = v_output[pix_id];
    }

    for (var b = 0u; b < num_batches; b++) {
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        let batch_end = range.y - b * BATCH_SIZE;
        let remaining = min(BATCH_SIZE, batch_end - range.x);

        // Each thread first gathers one gaussian.
        if local_idx < remaining {
            let load_isect_id = batch_end - 1 - local_idx;
            let load_compact_gid = compact_gid_from_isect[load_isect_id];
            local_id[local_idx] = load_compact_gid;
            local_batch[local_idx] = projected_splats[load_compact_gid];
        }

        // Wait for all threads to finish loading.
        workgroupBarrier();

        for (var t = 0u; t < remaining; t += 1) {
            let isect_id = batch_end - 1 - t;

            var v_xy = vec2f(0.0);
            var v_conic = vec3f(0.0);
            var v_colors = vec4f(0.0);
            var v_refine = vec2f(0.0);

            var splat_active = false;

            if inside && isect_id < u32(final_isect) {
                let projected = local_batch[t];

                let xy = vec2f(projected.xy_x, projected.xy_y);
                let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
                let color = vec4f(projected.color_r, projected.color_g, projected.color_b, projected.color_a);

                let delta = xy - pixel_coord;
                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let vis = exp(-sigma);
                let alpha = min(0.99f, color.w * vis);

                // Nb: Don't continue; here - local_idx == 0 always
                // needs to write out gradients.
                // compute the current T for this gaussian
                if (sigma >= 0.0 && alpha >= 1.0 / 255.0) {
                    splat_active = true;

                    let ra = 1.0 / (1.0 - alpha);
                    T *= ra;
                    // update v_colors for this gaussian
                    let fac = alpha * T;

                    // contribution from this pixel
                    let clamped_rgb = max(color.rgb, vec3f(0.0));
                    var v_alpha = dot(clamped_rgb * T - buffer * ra, v_out.rgb);
                    v_alpha += T_final * ra * v_out.a;

                    // update the running sum
                    buffer += clamped_rgb * fac;

                    let v_sigma = -color.a * vis * v_alpha;

                    v_xy = v_sigma * vec2f(
                        conic.x * delta.x + conic.y * delta.y,
                        conic.y * delta.x + conic.z * delta.y
                    );

                    v_conic = vec3f(0.5f * v_sigma * delta.x * delta.x,
                                            v_sigma * delta.x * delta.y,
                                    0.5f * v_sigma * delta.y * delta.y);

                    let v_rgb = select(vec3f(0.0), fac * v_out.rgb, color.rgb > vec3f(0.0));
                    v_colors = vec4f(v_rgb, vis * v_alpha);

                    v_refine = abs(v_xy);
                }
            }

            // Queue a new gradient if this subgroup has any.
            // The gradient is sum of all gradients in the subgroup.
            if subgroupAny(splat_active) {
                let compact_gid = local_id[t];

                let v_xy_sum = subgroupAdd(v_xy);
                let v_conic_sum = subgroupAdd(v_conic);
                let v_colors_sum = subgroupAdd(v_colors);
                let v_refine_sum = subgroupAdd(v_refine);

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
