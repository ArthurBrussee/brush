struct Uniforms {
    wg_size_x: i32,
    wg_size_y: i32,
    wg_size_z: i32,
}

@group(0) @binding(0) var<storage, read> thread_counts: array<i32>;
@group(0) @binding(1) var<storage, read_write> wg_count: array<i32>;

@group(0) @binding(2) var<storage, read> uniforms: Uniforms;

fn ceil_div(a: i32, b: i32) -> i32 {
    return (a + b - 1) / b;
}

// Maximum workgroups per dimension (WebGPU limit)
const MAX_WG_PER_DIM: i32 = 65535;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    if global_id.x > 0 {
        return;
    }

    var cx = 1;
    if arrayLength(&thread_counts) >= 1u {
        cx = thread_counts[0];
    }

    var cy = 1;
    if arrayLength(&thread_counts) >= 2u {
        cy = thread_counts[1];
    }

    var cz = 1;
    if arrayLength(&thread_counts) >= 3u {
        cz = thread_counts[2];
    }

    var wg_x = ceil_div(cx, uniforms.wg_size_x);
    var wg_y = ceil_div(cy, uniforms.wg_size_y);
    let wg_z = ceil_div(cz, uniforms.wg_size_z);

    // If wg_x exceeds the limit, split into 2D dispatch.
    if wg_x > MAX_WG_PER_DIM && wg_y == 1 {
        wg_y = ceil_div(wg_x, MAX_WG_PER_DIM);
        wg_x = MAX_WG_PER_DIM;
    }

    wg_count[0] = wg_x;
    wg_count[1] = wg_y;
    wg_count[2] = wg_z;
}
