#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> projected: array<helpers::ProjectedSplat>;

#ifdef PREPASS
    @group(0) @binding(2) var<storage, read_write> splat_intersect_counts: array<atomic<i32>>;
    @group(0) @binding(3) var<storage, read_write> tile_intersect_counts: array<atomic<i32>>;
#else
    @group(0) @binding(2) var<storage, read> splat_cum_hit_counts: array<i32>;
    @group(0) @binding(3) var<storage, read_write> tile_id_from_isect: array<i32>;
    @group(0) @binding(4) var<storage, read_write> compact_gid_from_isect: array<i32>;
#endif


@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let compact_gid = gid.x;

    if i32(compact_gid) >= uniforms.num_visible {
        return;
    }

    let projected = projected[compact_gid];
    let mean2d = vec2f(projected.xy_x, projected.xy_y);
    let opac = 1.0;

    // Reconstruct conic matrix.
    let conic = mat2x2f(projected.conic_x, projected.conic_y, projected.conic_y, projected.conic_z);
    let cov_from_conic = helpers::inverse(conic);
    let radius = helpers::radius_from_cov(cov_from_conic, opac);
    let tile_minmax = helpers::get_tile_bbox(mean2d, radius, uniforms.tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    var num_tiles_hit = 0;

    #ifdef PREPASS
        var base_isect_id = 0;
    #else
        var base_isect_id = splat_cum_hit_counts[compact_gid];
    #endif

    // Nb: It's really really important here the two dispatches
    // of this kernel arrive at the exact same num_tiles_hit count. Otherwise
    // we might not be writing some intersection data.
    // This is a bit scary given potential optimizations that might happen depending
    // on which version is being ran.
    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            if helpers::can_be_visible(vec2u(tx, ty), mean2d, conic, opac) {
                let tile_id = tx + ty * uniforms.tile_bounds.x;

            #ifdef PREPASS
                // TODO: Want to bail here if the tile is saturated with gaussians, but not clear
                // how the prepass and final pass would agree.
                atomicAdd(&tile_intersect_counts[tile_id + 1u], 1);
            #else
                let isect_id = base_isect_id + num_tiles_hit;
                // Nb: isect_id MIGHT be out of bounds here for degenerate cases.
                // These kernels should be launched with bounds checking, so that these
                // writes are ignored. This will skip these intersections.
                tile_id_from_isect[isect_id] = i32(tile_id);
                compact_gid_from_isect[isect_id] = i32(compact_gid);
            #endif

                num_tiles_hit += 1;
            }
        }
    }

    #ifdef PREPASS
        splat_intersect_counts[compact_gid + 1u] = num_tiles_hit;
    #endif
}
