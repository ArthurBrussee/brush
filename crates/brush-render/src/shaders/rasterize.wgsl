#import helpers

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<i32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<i32>;
@group(0) @binding(3) var<storage, read> projected_splats: array<helpers::ProjectedSplat>;

#ifdef RASTER_U32
    @group(0) @binding(4) var<storage, read_write> out_img: array<u32>;
#else
    @group(0) @binding(4) var<storage, read_write> out_img: array<vec4f>;
#endif

@group(0) @binding(5) var<storage, read_write> final_index : array<i32>;

var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_WIDTH, helpers::TILE_WIDTH, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let img_size = uniforms.img_size;

    // Get index of tile being drawn.
    let pix_id = i32(global_id.x) + i32(global_id.y) * img_size.x;
    let tile_id = i32(workgroup_id.x) + i32(workgroup_id.y) * uniforms.tile_bounds.x;
    let pixel_coord = vec2f(global_id.xy) + 0.5;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = i32(global_id.x) < img_size.x && i32(global_id.y) < img_size.y;
    var done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    let range = vec2i(tile_offsets[tile_id], tile_offsets[tile_id + 1]);

    let num_batches = helpers::ceil_div(range.y - range.x, i32(helpers::TILE_SIZE));
    // current visibility left to render
    var T = 1.0;

    var pix_out = vec3f(0.0);

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    var t = 0;
    var final_idx = 0;

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0; b < num_batches; b++) {
        let batch_start = range.x + b * i32(helpers::TILE_SIZE);

        // Wait for all in flight threads.
        workgroupBarrier();

        // process gaussians in the current batch for this pixel
        let remaining = min(i32(helpers::TILE_SIZE), range.y - batch_start);

        if i32(local_idx) < remaining {
            let load_isect_id = batch_start + i32(local_idx);
            local_batch[local_idx] = projected_splats[compact_gid_from_isect[load_isect_id]];
        }
        // Wait for all writes to complete.
        workgroupBarrier();

        for (var t = 0; t < remaining && !done; t++) {
            let projected = local_batch[t];

            let xy = vec2f(projected.xy_x, projected.xy_y);
            let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
            let color = vec4f(projected.color_r, projected.color_g, projected.color_b, projected.color_a);

            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let vis = exp(-sigma);
            let alpha = min(0.999f, color.a * vis);

            if sigma >= 0.0 && alpha >= 1.0 / 255.0 {
                let next_T = T * (1.0 - alpha);

                if next_T <= 1e-4f {
                    done = true;
                    break;
                }

                let fac = alpha * T;
                let clamped_rgb = max(color.rgb, vec3f(0.0));
                pix_out += clamped_rgb * fac;
                T = next_T;

                let isect_id = batch_start + t;
                final_idx = isect_id + 1;
            }
        }
    }

    if inside {
        let img_alpha = (1.0 - T);
        let final_color = vec4f(pix_out, img_alpha);
        #ifdef RASTER_U32
            let colors_u = vec4u(clamp(final_color * 255.0, vec4f(0.0), vec4f(255.0)));
            let packed: u32 = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
            out_img[pix_id] = packed;
            final_index[pix_id] = final_idx;
        #else
            out_img[pix_id] = final_color;
            final_index[pix_id] = final_idx;
        #endif
    }
}
