#import helpers;

@group(0) @binding(0) var<uniform> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<i32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<i32>;

@group(0) @binding(3) var<storage, read> projected_splats: array<helpers::ProjectedSplat>;

@group(0) @binding(4) var<storage, read> final_index: array<i32>;
@group(0) @binding(5) var<storage, read> output: array<vec4f>;
@group(0) @binding(6) var<storage, read> v_output: array<vec4f>;

#ifdef HARD_FLOAT
    @group(0) @binding(7) var<storage, read_write> v_xy: array<atomic<f32>>;
    @group(0) @binding(8) var<storage, read_write> v_conics: array<atomic<f32>>;
    @group(0) @binding(9) var<storage, read_write> v_colors: array<atomic<f32>>;
#else
    @group(0) @binding(7) var<storage, read_write> v_xy: array<atomic<u32>>;
    @group(0) @binding(8) var<storage, read_write> v_conics: array<atomic<u32>>;
    @group(0) @binding(9) var<storage, read_write> v_colors: array<atomic<u32>>;
#endif


const BATCH_SIZE = helpers::TILE_SIZE;

// Gaussians gathered in batch.
var<workgroup> local_batch: array<helpers::ProjectedSplat, BATCH_SIZE>;
var<workgroup> local_id: array<i32, BATCH_SIZE>;

// This kernel use a new technique to reduce the overhead of atomic gradient accumulation, especially when
// using software CAS loops this helps performance a lot. Originally, each thread calculated
// a gradient, summed them together in a subgroup, and one thread of these subgroups would then atomically add
// this gradient to the global gradient. Instead, we push each subgroup gradient to a buffer
// until it has N threads gradients, which are then written to the global gradients all at once.

// Current queue of gradients to be flushed.
var<workgroup> grad_count: atomic<i32>;
var<workgroup> gather_grads: array<helpers::ProjectedSplat, BATCH_SIZE>;
var<workgroup> gather_grad_id: array<i32, BATCH_SIZE>;

fn add_bitcast(cur: u32, add: f32) -> u32 {
    return bitcast<u32>(bitcast<f32>(cur) + add);
}

fn write_grads_atomic(grads: helpers::ProjectedSplat, id: i32) {
#ifdef HARD_FLOAT
    atomicAdd(&v_xy[id * 2 + 0], grads.xy_x);
    atomicAdd(&v_xy[id * 2 + 1], grads.xy_y);

    atomicAdd(&v_conics[id * 3 + 0], grads.conic_x);
    atomicAdd(&v_conics[id * 3 + 1], grads.conic_y);
    atomicAdd(&v_conics[id * 3 + 2], grads.conic_z);

    atomicAdd(&v_colors[id * 4 + 0], grads.color_r);
    atomicAdd(&v_colors[id * 4 + 1], grads.color_g);
    atomicAdd(&v_colors[id * 4 + 2], grads.color_b);
    atomicAdd(&v_colors[id * 4 + 3], grads.color_a);
#else
    // Alternatively can run without any atomics and just race:
    // v_xy[id * 2 + 0] = add_bitcast(v_xy[id * 2 + 0], grads.xy.x);
    // v_xy[id * 2 + 1] = add_bitcast(v_xy[id * 2 + 1], grads.xy.y);
    // v_conics[id * 3 + 0] = add_bitcast(v_conics[id * 3 + 0], grads.conic.x);
    // v_conics[id * 3 + 1] = add_bitcast(v_conics[id * 3 + 1], grads.conic.y);
    // v_conics[id * 3 + 2] = add_bitcast(v_conics[id * 3 + 2], grads.conic.z);
    // v_colors[id * 4 + 0] = add_bitcast(v_colors[id * 4 + 0], grads.color.r);
    // v_colors[id * 4 + 1] = add_bitcast(v_colors[id * 4 + 1], grads.color.g);
    // v_colors[id * 4 + 2] = add_bitcast(v_colors[id * 4 + 2], grads.color.b);
    // v_colors[id * 4 + 3] = add_bitcast(v_colors[id * 4 + 3], grads.color.a);

    // Writing out all these CAS loops individually is terrible but wgsl doesn't have a mechanism to
    // turn this into a function as ptr<storage> can't be passed to a function...
    // v_xy.x
    let xy = vec2f(grads.xy_x, grads.xy_y);
    let conic = vec3f(grads.conic_x, grads.conic_y, grads.conic_z);
    let color = vec4f(grads.color_r, grads.color_g, grads.color_b, grads.color_a);

    var old_value = atomicLoad(&v_xy[id * 2 + 0]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_xy[id * 2 + 0], old_value, add_bitcast(old_value, xy.x));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_xy.y
    old_value = atomicLoad(&v_xy[id * 2 + 1]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_xy[id * 2 + 1], old_value, add_bitcast(old_value, xy.y));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }

    // v_conic.x
    old_value = atomicLoad(&v_conics[id * 3 + 0]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_conics[id * 3 + 0], old_value, add_bitcast(old_value, conic.x));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_conic.y
    old_value = atomicLoad(&v_conics[id * 3 + 1]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_conics[id * 3 + 1], old_value, add_bitcast(old_value, conic.y));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_conic.z
    old_value = atomicLoad(&v_conics[id * 3 + 2]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_conics[id * 3 + 2], old_value, add_bitcast(old_value, conic.z));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }

    // v_color.r
    old_value = atomicLoad(&v_colors[id * 4 + 0]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 0], old_value, add_bitcast(old_value, color.r));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_color.g
    old_value = atomicLoad(&v_colors[id * 4 + 1]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 1], old_value, add_bitcast(old_value, color.g));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_color.b
    old_value = atomicLoad(&v_colors[id * 4 + 2]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 2], old_value, add_bitcast(old_value, color.b));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_color.a
    old_value = atomicLoad(&v_colors[id * 4 + 3]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 3], old_value, add_bitcast(old_value, color.a));
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

    let tile_id = i32(workgroup_id.x);

    let tile_loc = vec2i(tile_id % tile_bounds.x, tile_id / tile_bounds.x);
    let pixel_coordi = tile_loc * i32(helpers::TILE_WIDTH) + vec2i(i32(local_idx % helpers::TILE_WIDTH), i32(local_idx / helpers::TILE_WIDTH));
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
    let range = vec2i(tile_offsets[tile_id], tile_offsets[tile_id + 1]);

    let num_batches = helpers::ceil_div(range.y - range.x, i32(BATCH_SIZE));

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

    // Make sure all groups start with empty gradient queue.
    atomicStore(&grad_count, 0);

    let sg_per_tile = helpers::ceil_div(i32(helpers::TILE_SIZE), i32(subgroup_size));
    let microbatch_size = i32(helpers::TILE_SIZE) / sg_per_tile;

    for (var b = 0; b < num_batches; b++) {
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        let batch_end = range.y - b * i32(BATCH_SIZE);
        let remaining = min(i32(BATCH_SIZE), batch_end - range.x);

        // Gather N gaussians.
        var load_compact_gid = 0;
        if i32(local_idx) < remaining {
            let load_isect_id = batch_end - 1 - i32(local_idx);
            load_compact_gid = compact_gid_from_isect[load_isect_id];
            local_id[local_idx] = load_compact_gid;
            local_batch[local_idx] = projected_splats[load_compact_gid];
        }

        for (var tb = 0; tb < remaining; tb += microbatch_size) {
            if local_idx == 0 {
                atomicStore(&grad_count, 0);
            }
            workgroupBarrier();

            for(var tt = 0; tt < microbatch_size; tt++) {
                let t = tb + tt;

                if t >= remaining {
                    break;
                }

                let isect_id = batch_end - 1 - t;

                var v_xy = vec2f(0.0);
                var v_conic = vec3f(0.0);
                var v_colors = vec4f(0.0);

                var splat_active = false;

                if inside && isect_id < final_isect {
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
                    }
                }

                // Queue a new gradient if this subgroup has any.
                // The gradient is sum of all gradients in the subgroup.
                if subgroupAny(splat_active) {
                    var v_xy_sum = subgroupAdd(v_xy);
                    var v_conic_sum = subgroupAdd(v_conic);
                    var v_colors_sum = subgroupAdd(v_colors);

                    // First thread of subgroup writes the gradient. This should be a
                    // subgroupBallot() when it's supported.
                    if subgroup_invocation_id == 0 {
                        let grad_idx = atomicAdd(&grad_count, 1);
                        gather_grads[grad_idx] = helpers::create_projected_splat(
                            v_xy_sum,
                            v_conic_sum,
                            v_colors_sum
                        );
                        gather_grad_id[grad_idx] = local_id[t];
                    }
                }
            }

            // Make sure all threads are done, and flush a batch of gradients.
            workgroupBarrier();
            if local_idx < u32(grad_count) {
                write_grads_atomic(gather_grads[local_idx], gather_grad_id[local_idx]);
            }
            workgroupBarrier();
        }
    }
}
