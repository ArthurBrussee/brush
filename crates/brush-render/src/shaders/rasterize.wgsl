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

const LOAD_PER_ITER: u32 = 64u;

var<workgroup> local_batch: array<helpers::ProjectedSplat, LOAD_PER_ITER>;
var<workgroup> range_uniform: vec2u;

#ifdef BWD_INFO
    var<workgroup> load_gid: array<u32, LOAD_PER_ITER>;
#endif

const PIXELS_PER_THREAD_X = 2;
const PIXELS_PER_THREAD_Y = 2;
const PIXELS_PER_THREAD = PIXELS_PER_THREAD_X * PIXELS_PER_THREAD_Y;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let img_size = uniforms.img_size;

    // Get index of tile being drawn.
    let tile_id = group_id.x + group_id.y * uniforms.tile_bounds.x;

    let tile_origin = vec2u(
        group_id.x * helpers::TILE_WIDTH,
        group_id.y * helpers::TILE_WIDTH,
    );

    var pixel_coords: array<vec2u, PIXELS_PER_THREAD>;
    var done: array<u32, PIXELS_PER_THREAD>;
    var pix_out: array<vec4f, PIXELS_PER_THREAD>;

    for (var i = 0u; i < PIXELS_PER_THREAD_X; i++) {
        for (var j = 0u; j < PIXELS_PER_THREAD_Y; j++) {
            let pixel_offset = vec2u(
                local_id.x * PIXELS_PER_THREAD_X + i,
                local_id.y * PIXELS_PER_THREAD_Y + j,
            );
            let global_pixel = tile_origin + pixel_offset;
            let pix_idx = i + j * PIXELS_PER_THREAD_X;
            pixel_coords[pix_idx] = global_pixel;

            let inside = global_pixel.x < img_size.x && global_pixel.y < img_size.y;
            done[pix_idx] = select(1u, 0u, inside);
            pix_out[pix_idx] = vec4f(0.0, 0.0, 0.0, 1.0);
        }
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    range_uniform = vec2u(
        tile_offsets[tile_id * 2],
        tile_offsets[tile_id * 2 + 1],
    );
    // Stupid hack as Chrome isn't convinced the range variable is uniform, which it better be.
    let range = workgroupUniformLoad(&range_uniform);

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var batch_start = range.x; batch_start < range.y; batch_start += LOAD_PER_ITER) {
        // process gaussians in the current batch for this pixel
        let remaining = min(LOAD_PER_ITER, range.y - batch_start);

        // Wait for all writes to complete.
        workgroupBarrier();

        if local_idx < remaining {
            let load_isect_id = batch_start + local_idx;
            let compact_gid = compact_gid_from_isect[load_isect_id];
            local_batch[local_idx] = projected_splats[compact_gid];

            // Visibility is written to global ID's.
            #ifdef BWD_INFO
                load_gid[local_idx] = global_from_compact_gid[compact_gid];
            #endif
        }

        // Wait for all writes to complete.
        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            let projected = local_batch[t];

            let xy = vec2f(projected.xy_x, projected.xy_y);
            let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
            let color = vec4f(projected.color_r, projected.color_g, projected.color_b, projected.color_a);

            for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
                if done[i] == 1u {
                    continue;
                }

                let delta = xy - (vec2f(pixel_coords[i]) + 0.5f);
                let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                let alpha = min(0.999f, color.a * exp(-sigma));

                if (sigma < 0.0f || alpha < 1.0f / 255.0f) {
                    continue;
                }

                let cur_pix = pix_out[i];
                let next_T = cur_pix.a * (1.0 - alpha);

                if (next_T <= 1e-4f) {
                    done[i] = 1u;
                    continue;
                }

                let vis = alpha * cur_pix.a;
                let clamped_rgb = max(color.rgb, vec3f(0.0));
                pix_out[i] = vec4f(cur_pix.rgb + clamped_rgb * vis, next_T);

                #ifdef BWD_INFO
                    let gid = load_gid[t];
                    visible[gid] = 1.0;
                #endif
            }
        }
    }

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let coords = pixel_coords[i];
        let inside = all(coords < img_size);

        if inside {
            // Compose with background. Nb that color is already pre-multiplied
            // by definition.
            let cur_pix = pix_out[i];
            let final_color = vec4f(cur_pix.rgb + cur_pix.a * uniforms.background.rgb, 1.0 - cur_pix.a);
            let pix_id = coords.x + coords.y * img_size.x;

            #ifdef BWD_INFO
                out_img[pix_id] = final_color;
            #else
                let colors_u = vec4u(clamp(final_color * 255.0, vec4f(0.0), vec4f(255.0)));
                let packed: u32 = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
                out_img[pix_id] = packed;
            #endif
        }
    }
}
