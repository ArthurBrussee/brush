@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

const THREADS_PER_GROUP: u32 = 512u;

// Maximum workgroups per dimension (WebGPU limit)
const MAX_WG_PER_DIM: u32 = 65535u;

// Compute linear workgroup ID from 2D dispatch
fn get_workgroup_id(wid: vec3u) -> u32 {
    return wid.x + wid.y * MAX_WG_PER_DIM;
}

// Compute linear global invocation ID from 2D dispatch
fn get_global_id(gid: vec3u, wid: vec3u, lid: u32) -> u32 {
    return get_workgroup_id(wid) * THREADS_PER_GROUP + lid;
}

var<workgroup> bucket: array<u32, THREADS_PER_GROUP>;

fn groupScan(id: u32, gi: u32, x: u32) {
    bucket[gi] = x;
    for (var t = 1u; t < THREADS_PER_GROUP; t = t * 2u) {
        workgroupBarrier();
        var temp = bucket[gi];
        if (gi >= t) {
            temp += bucket[gi - t];
        }
        workgroupBarrier();
        bucket[gi] = temp;
    }
    if id < arrayLength(&output) {
        output[id] = bucket[gi];
    }
}
