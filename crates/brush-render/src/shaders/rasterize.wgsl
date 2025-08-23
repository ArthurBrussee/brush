#import helpers

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected_splats: array<helpers::ProjectedSplat>;

#ifdef BWD_INFO
    @group(0) @binding(4) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(5) var<storage, read> global_from_compact_gid: array<u32>;
    @group(0) @binding(6) var<storage, read_write> visible: array<f32>;
#else
    @group(0) @binding(4) var<storage, read_write> out_img: array<u32>;
#endif

var<workgroup> range_uniform: vec2u;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::CHUNK_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,

    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let pix_loc = helpers::map_1d_to_2d(global_id.x, uniforms.tile_bounds.x);
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

    let num_batches = helpers::ceil_div(range.y - range.x, subgroup_size);

    // current visibility left to render
    var T = 1.0;
    var pix_out = vec3f(0.0);
    var done = !inside;

    let group_id = local_idx / subgroup_size;
    let rect_min = subgroupMin(pixel_coord);
    let rect_max = subgroupMax(pixel_coord);
    let sub_rect = vec4f(rect_min.x, rect_min.y, rect_max.x, rect_max.y);

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        let batch_start = range.x + b * subgroup_size;

        // process gaussians in the current batch for this pixel
        let remaining = min(subgroup_size, range.y - batch_start);

        let load_isect_id = batch_start + subgroup_invocation_id;
        let compact_gid = compact_gid_from_isect[load_isect_id];
        let projected = projected_splats[compact_gid];

        #ifdef BWD_INFO
            let load_gid = global_from_compact_gid[compact_gid];
        #endif

        let xy_load = vec2f(projected.xy_x, projected.xy_y);
        let conic_load = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
        let color_load = vec4f(projected.color_r, projected.color_g, projected.color_b, projected.color_a);

        let power_threshold = log(color_load.w * 255.0f);
        let chunk_visible_load = helpers::will_primitive_contribute(sub_rect, xy_load, conic_load, power_threshold);
        let vis_ballot = subgroupBallot(chunk_visible_load);

        for (var t = 0u; t < remaining; t++) {
            // TODO: Support subgroup size 64.
            if (vis_ballot.x & (1u << t)) == 0u {
                continue;
            }

            // Broadcast from right sg element.
            let xy = subgroupShuffle(xy_load, t);
            let conic = subgroupShuffle(conic_load, t);
            let color = subgroupShuffle(color_load, t);
            #ifdef BWD_INFO
                let gid = subgroupShuffle(load_gid, t);
            #endif

            if !done {
                let delta = xy - pixel_coord;
                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let alpha = min(0.999f, color.a * exp(-sigma));

                if sigma >= 0.0f && alpha >= 1.0f / 255.0f {
                    let next_T = T * (1.0 - alpha);

                    if next_T <= 1e-4f {
                        done = true;
                    } else {
                        #ifdef BWD_INFO
                            visible[gid] = 1.0;
                        #endif

                        let vis = alpha * T;
                        let clamped_rgb = max(color.rgb, vec3f(0.0));
                        pix_out += clamped_rgb * vis;
                        T = next_T;
                    }
                }
            }
        }

        // TODO: Not allowed on the web because control flow would be non uniform ugh.
        if subgroupAll(done) {
            // TODO: Write subgroup max.
            break;
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
}
