enable f16;
#import helpers

@group(0) @binding(0) var<storage, read> compact_gid_from_isect: array<u32>;
// In the bwd path the forward shrinks tile_offsets[tile*2+1] to "one past
// the last splat that any pixel actually consumed" so the backward kernel's
// outer loop can stop early. The forward already loaded its own copy of
// the range into `range_uniform` before any write, so the in-place mutation
// is safe within the kernel.
#ifdef BWD_INFO
    @group(0) @binding(1) var<storage, read_write> tile_offsets: array<u32>;
#else
    @group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
#endif
@group(0) @binding(2) var<storage, read> projected: array<helpers::ProjectedSplat>;

#ifdef BWD_INFO
    @group(0) @binding(3) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(4) var<storage, read> global_from_compact_gid: array<u32>;
    @group(0) @binding(5) var<storage, read_write> visible: array<f32>;
    @group(0) @binding(6) var<storage, read> uniforms: helpers::RasterizeUniforms;
#else
    @group(0) @binding(3) var<storage, read_write> out_img: array<u32>;
    @group(0) @binding(4) var<storage, read> uniforms: helpers::RasterizeUniforms;
#endif

var<workgroup> range_uniform: vec2u;

var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;

#ifdef BWD_INFO
    var<workgroup> load_gid: array<u32, helpers::TILE_SIZE>;
#endif

// Per-workgroup count of pixels that have terminated (alpha-saturated or
// off-image). When this reaches TILE_SIZE we can stop loading more splats:
// every remaining thread would no-op anyway, so skipping the load+barrier
// pair changes nothing about the output.
var<workgroup> num_done_atomic: atomic<u32>;

#ifdef BWD_INFO
// Tile-wide max of the (one-past) intersection index any pixel actually
// consumed. We feed this back into tile_offsets so the backward kernel's
// outer loop runs over a tighter range. atomicMax against this from each
// pixel gives the right tile-wide value with no additional barriers.
var<workgroup> max_useful_isect: atomic<u32>;
#endif

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let global_id = helpers::get_global_id(wg_id, num_wgs, local_idx, helpers::TILE_SIZE);
    let pix_loc = helpers::map_1d_to_2d(global_id, uniforms.tile_bounds.x);
    let pix_id = pix_loc.x + pix_loc.y * uniforms.img_size.x;
    let pixel_coord = vec2f(pix_loc) + 0.5f;
    let tile_loc = vec2u(pix_loc.x / helpers::TILE_WIDTH, pix_loc.y / helpers::TILE_WIDTH);

    let tile_id = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;
    let inside = pix_loc.x < uniforms.img_size.x && pix_loc.y < uniforms.img_size.y;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    range_uniform = vec2u(
        tile_offsets[tile_id * 2],
        tile_offsets[tile_id * 2 + 1],
    );

    // Stupid hack as Chrome isn't convinced the range variable is uniform, which it better be.
    let range = workgroupUniformLoad(&range_uniform);

    // current visibility left to render
    var T = 1.0;
    var pix_out = vec3f(0.0);
    var done = !inside;
    // Per-pixel "one past last consumed isect index". Updated whenever a
    // splat actually contributes to this pixel; at end-of-kernel reduced
    // workgroup-wide to shrink tile_offsets for the backward.
    var last_useful_isect: u32 = range.x;

    // Seed the workgroup-wide done counter with off-image pixels.
    if local_idx == 0u {
        atomicStore(&num_done_atomic, 0u);
        #ifdef BWD_INFO
            atomicStore(&max_useful_isect, range.x);
        #endif
    }
    workgroupBarrier();
    if done { atomicAdd(&num_done_atomic, 1u); }
    workgroupBarrier();

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var batch_start = range.x; batch_start < range.y; batch_start += helpers::TILE_SIZE) {
        // Bail when every pixel in this workgroup is alpha-saturated or
        // off-image. atomicLoad without an extra barrier may read a stale
        // (lower) count — that's fine, it just defers the early-exit by
        // one batch.
        if workgroupUniformLoad(&num_done_atomic) >= helpers::TILE_SIZE { break; }


        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        let load_isect_id = batch_start + local_idx;
        var compact_gid = 0u;
        if local_idx < remaining {
            compact_gid = compact_gid_from_isect[load_isect_id];
        }

        workgroupBarrier();
        if local_idx < remaining {
            local_batch[local_idx] = projected[compact_gid];
            #ifdef BWD_INFO
                load_gid[local_idx] = global_from_compact_gid[compact_gid];
            #endif
        }
        workgroupBarrier();

        let was_done = done;
        for (var t = 0u; !done && t < remaining; t++) {
            let proj = local_batch[t];

            let xy = vec2f(proj.xy_x, proj.xy_y);
            let conic = vec3f(proj.conic_x, proj.conic_y, proj.conic_z);
            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let alpha = min(0.999f, proj.color_a * exp(-sigma));

            if sigma >= 0.0f && alpha >= 1.0f / 255.0f {
                let next_T = T * (1.0 - alpha);

                if next_T <= 1e-4f {
                    done = true;
                    break;
                }

                #ifdef BWD_INFO
                    // Count visible if contribution is at least somewhat significant.
                    visible[load_gid[t]] = 1.0;
                #endif

                let vis = alpha * T;
                let color_rgb = max(vec3f(f32(proj.color_r), f32(proj.color_g), f32(proj.color_b)), vec3f(0.0));
                pix_out += color_rgb * vis;
                T = next_T;
                // (batch_start + t) is the isect index of the splat we
                // just consumed; +1 gives the exclusive end the backward
                // wants in tile_offsets[tile*2+1].
                last_useful_isect = batch_start + t + 1u;
            }
        }
        if !was_done && done {
            atomicAdd(&num_done_atomic, 1u);
        }
    }

    if inside {
        // Compose with background. Nb that color is already pre-multiplied
        // by definition.
        let final_color = vec4f(pix_out + T * uniforms.background.rgb, 1.0 - T);

        #ifdef BWD_INFO
            out_img[pix_id] = final_color;
        #else
            let colors_u = vec4u(clamp(final_color * 255.0, vec4f(0.0), vec4f(255.0)));
            let packed: u32 = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
            out_img[pix_id] = packed;
        #endif
    }

    // Reduce per-pixel `last_useful_isect` workgroup-wide and shrink
    // tile_offsets[tile*2+1] for the backward.
    #ifdef BWD_INFO
        atomicMax(&max_useful_isect, last_useful_isect);
        workgroupBarrier();
        if local_idx == 0u {
            tile_offsets[tile_id * 2u + 1u] = atomicLoad(&max_useful_isect);
        }
    #endif
}
