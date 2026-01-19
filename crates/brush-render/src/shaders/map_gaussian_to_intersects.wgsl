#import helpers;

@group(0) @binding(0) var<storage, read> num_visible: u32;
@group(0) @binding(1) var<storage, read> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(2) var<storage, read> splat_cum_hit_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_id_from_isect: array<u32>;
@group(0) @binding(4) var<storage, read_write> compact_gid_from_isect: array<u32>;

// Uniforms passed via with_metadata (always last binding)
struct Uniforms {
    tile_bounds: vec2u,
}
@group(0) @binding(5) var<storage, read> uniforms: Uniforms;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let compact_gid = helpers::get_global_id(wid, num_wgs, lid, WG_SIZE);

    if compact_gid >= num_visible {
        return;
    }

    let projected = projected[compact_gid];
    let mean2d = vec2f(projected.xy_x, projected.xy_y);
    let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
    let opac = projected.color_a;

    let power_threshold = log(opac * 255.0);
    let cov2d = helpers::inverse(mat2x2f(conic.x, conic.y, conic.y, conic.z));
    let extent = helpers::compute_bbox_extent(cov2d, power_threshold);
    let tile_bbox = helpers::get_tile_bbox(mean2d, extent, uniforms.tile_bounds);
    let tile_bbox_min = tile_bbox.xy;
    let tile_bbox_max = tile_bbox.zw;

    // With inclusive prefix sum, use cum[compact_gid - 1] as base (or 0 for first element)
    let base_isect_id = select(splat_cum_hit_counts[compact_gid - 1u], 0u, compact_gid == 0u);

    var num_tiles_hit = 0u;

    // Nb: It's really really important here the two dispatches
    // of this kernel arrive at the exact same num_tiles_hit count. Otherwise
    // we might not be writing some intersection data.
    // This is a bit scary given potential optimizations that might happen depending
    // on which version is being ran.
    let tile_bbox_width = tile_bbox_max.x - tile_bbox_min.x;
    let num_tiles_bbox = (tile_bbox_max.y - tile_bbox_min.y) * tile_bbox_width;

    for (var tile_idx = 0u; tile_idx < num_tiles_bbox; tile_idx++) {
        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min.x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min.y;

        let rect = helpers::tile_rect(vec2u(tx, ty));
        if helpers::will_primitive_contribute(rect, mean2d, conic, power_threshold) {
            let tile_id = tx + ty * uniforms.tile_bounds.x;

            let isect_id = base_isect_id + num_tiles_hit;
            // Nb: isect_id MIGHT be out of bounds here for degenerate cases.
            // These kernels should be launched with bounds checking, so that these
            // writes are ignored. This will skip these intersections.
            tile_id_from_isect[isect_id] = tile_id;
            compact_gid_from_isect[isect_id] = compact_gid;

            num_tiles_hit += 1u;
        }
    }
}
