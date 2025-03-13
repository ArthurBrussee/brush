#import helpers

@group(0) @binding(0) var<uniform> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<i32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<i32>;
@group(0) @binding(3) var<storage, read> projected_splats: array<helpers::ProjectedSplat>;

@group(0) @binding(4) var<storage, read> global_from_compact_gid: array<i32>;

#ifdef RASTER_U32
    @group(0) @binding(5) var<storage, read_write> out_img: array<u32>;
#else
    @group(0) @binding(5) var<storage, read_write> out_img: array<vec4f>;

    @group(0) @binding(6) var<storage, read_write> final_index: array<i32>;
    @group(0) @binding(7) var<storage, read_write> visible: array<f32>;
#endif

var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;
var<workgroup> load_gid: array<u32, helpers::TILE_SIZE>;

var<workgroup> done_count: atomic<u32>;

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
    let pix_id = global_id.x + global_id.y * img_size.x;
    let tile_id = workgroup_id.x + workgroup_id.y * uniforms.tile_bounds.x;
    let pixel_coord = vec2f(global_id.xy) + 0.5;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = global_id.x < img_size.x && global_id.y < img_size.y;
    var done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    let range = vec2u(u32(tile_offsets[tile_id]), u32(tile_offsets[tile_id + 1]));

    let num_batches = helpers::ceil_div(range.y - range.x, u32(helpers::TILE_SIZE));
    // current visibility left to render
    var T = 1.0;
    var pix_out = vec3f(0.0);

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    var t = 0;
    var final_idx = 0u;

    atomicStore(&done_count, 0u);

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        let batch_start = range.x + b * helpers::TILE_SIZE;

        // Wait for all in flight threads.
        workgroupBarrier();

        if atomicLoad(&done_count) >= helpers::TILE_SIZE {
            break;
        }

        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        if local_idx < remaining {
            let load_isect_id = batch_start + local_idx;
            let compact_gid = compact_gid_from_isect[load_isect_id];
            load_gid[local_idx] = u32(global_from_compact_gid[compact_gid]);
            local_batch[local_idx] = projected_splats[compact_gid];
        }
        // Wait for all writes to complete.
        workgroupBarrier();

        for (var t = 0u; t < remaining && !done; t++) {
            let projected = local_batch[t];

            let xy = vec2f(projected.xy_x, projected.xy_y);
            let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
            let color = vec4f(projected.color_r, projected.color_g, projected.color_b, projected.color_a);

            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let alpha = min(0.999f, color.a * exp(-sigma));

            if (sigma < 0.0f || alpha < 1.0f / 255.0f) {
                continue;
            }

            let next_T = T * (1.0 - alpha);

            if next_T <= 1e-4f {
                atomicAdd(&done_count, 1u);
                done = true;
                break;
            }

            #ifndef RASTER_U32
                let gid = load_gid[t];
                visible[gid] = 1.0;
            #endif

            let vis = alpha * T;
            let clamped_rgb = max(color.rgb, vec3f(0.0));
            pix_out += clamped_rgb * vis;
            T = next_T;

            let isect_id = batch_start + t;
            final_idx = isect_id + 1;
        }
    }

    if inside {
        let img_alpha = (1.0 - T);
        let final_color = vec4f(pix_out, img_alpha);
        #ifdef RASTER_U32
            let colors_u = vec4u(clamp(final_color * 255.0, vec4f(0.0), vec4f(255.0)));
            let packed: u32 = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
            out_img[pix_id] = packed;
        #else
            out_img[pix_id] = final_color;
            final_index[pix_id] = i32(final_idx);
        #endif
    }
}
