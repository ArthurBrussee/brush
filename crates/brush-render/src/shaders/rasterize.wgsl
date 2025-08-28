#import helpers

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected: array<vec4f>;

#ifdef BWD_INFO
    @group(0) @binding(4) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(5) var<storage, read> global_from_compact_gid: array<u32>;
    @group(0) @binding(6) var<storage, read_write> visible: array<f32>;
#else
    @group(0) @binding(4) var<storage, read_write> out_img: array<u32>;
#endif

var<workgroup> range_uniform: vec2u;

const NUM_VECS: u32 = helpers::TILE_SIZE * 3u;
var<workgroup> local_batch: array<vec4f, NUM_VECS>;
#ifdef BWD_INFO
    var<workgroup> load_gid: array<u32, helpers::TILE_SIZE>;
#endif

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::CHUNK_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
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
    let num_batches = helpers::ceil_div(range.y - range.x, helpers::CHUNK_SIZE);

    // current visibility left to render
    var T = 1.0;
    var pix_out = vec3f(0.0);
    var done = !inside;

    // let rect_min = subgroupMin(pixel_coord);
    // let rect_max = subgroupMax(pixel_coord);
    // let sub_rect = vec4f(rect_min.x, rect_min.y, rect_max.x, rect_max.y);

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

            #ifdef BWD_INFO
                load_gid[local_idx] = global_from_compact_gid[compact_gid];
            #endif
        }
        workgroupBarrier();

        for (var t = 0u; !done && t < remaining; t++) {
            let xy = local_batch[t].xy;
            let conic_alpha = local_batch[t + helpers::CHUNK_SIZE * 1];
            let conic = conic_alpha.xyz;
            let col_alpha = conic_alpha.w;

            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let alpha = min(0.999f, col_alpha * exp(-sigma));

            if sigma >= 0.0f && alpha >= 1.0f / 255.0f {
                let next_T = T * (1.0 - alpha);

                if next_T <= 1e-4f {
                    done = true;
                    break;
                }

                #ifdef BWD_INFO
                    visible[load_gid[t]] = 1.0;
                #endif

                let color = local_batch[t + helpers::CHUNK_SIZE * 2];
                let vis = alpha * T;
                pix_out += max(color.rgb, vec3f(0.0)) * vis;
                T = next_T;
            }
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
