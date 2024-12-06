#import helpers;

struct IsectInfo {
    compact_gid: u32,
    tile_id: u32,
}

@group(0) @binding(0) var<storage, read> num_intersections: u32;
@group(0) @binding(1) var<storage, read> isect_info: array<IsectInfo>;

@group(0) @binding(2) var<storage, read_write> cum_hits: array<atomic<u32>>;

@group(0) @binding(3) var<storage, read_write> tile_id_from_isect: array<u32>;
@group(0) @binding(4) var<storage, read_write> compact_gid_from_isect: array<u32>;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= num_intersections {
        return;
    }

    let isect_info = isect_info[gid.x];

    // Keep track of how many hits each tile has.

    // Find position in buffer.
    let tile_id = isect_info.tile_id;
    let compact_gid = isect_info.compact_gid;

    // Find base offset in the cumulative ghits.
    // var isect_id = cum_hits[compact_gid];
    var isect_id = atomicAdd(&cum_hits[compact_gid], 1u);

    // Write to the intersection buffers which are now sorted by depth.
    tile_id_from_isect[isect_id] = tile_id;
    compact_gid_from_isect[isect_id] = compact_gid;
}
