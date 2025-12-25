@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> length: array<u32>;

const THREADS_PER_GROUP: u32 = 512u;

fn get_length() -> u32 {
    return length[0];
}

// Compute linear workgroup ID from 2D dispatch
fn get_workgroup_id(wid: vec3u, num_wgs: vec3u) -> u32 {
    return wid.x + wid.y * num_wgs.x;
}

// Compute linear global invocation ID from 2D dispatch
fn get_global_id(wid: vec3u, num_wgs: vec3u, lid: u32) -> u32 {
    return get_workgroup_id(wid, num_wgs) * THREADS_PER_GROUP + lid;
}

var<workgroup> bucket: array<u32, THREADS_PER_GROUP>;

fn groupScan(id: u32, gi: u32, x: u32, len: u32) {
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
    if id < len {
        output[id] = bucket[gi];
    }
}
