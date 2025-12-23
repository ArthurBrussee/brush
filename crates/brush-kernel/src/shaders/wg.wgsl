// Dynamic dispatch buffer creation for 1D dispatches.
// Takes a single thread count and workgroup size, outputs (wg_x, wg_y, 1).
// Tiles into 2D if workgroup count exceeds 65535.

struct Uniforms {
    wg_size: u32,
}

@group(0) @binding(0) var<storage, read> thread_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> wg_count: array<u32>;
@group(0) @binding(2) var<storage, read> uniforms: Uniforms;

fn ceil_div(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    if global_id.x > 0 {
        return;
    }

    let num_threads = thread_count[0];
    let total_wgs = ceil_div(num_threads, uniforms.wg_size);

    // WebGPU limit is 65535 workgroups per dimension.
    var wg_x: u32 = total_wgs;
    var wg_y: u32 = 1u;

    if total_wgs > 65535 {
        wg_y = u32(ceil(sqrt(f32(total_wgs))));
        wg_x = ceil_div(total_wgs, wg_y);
    }

    wg_count[0] = wg_x;
    wg_count[1] = wg_y;
    wg_count[2] = 1;
}
