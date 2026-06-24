struct Uniforms {
    img_width: u32,
    img_height: u32,
    mode: u32,
    _pad: u32,
    depth_min: f32,
    depth_max: f32,
    _pad2: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> image_data: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle using oversized triangle technique
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);  // 0, 2, 0 for indices 0, 1, 2
    let y = f32(vertex_index & 2u);           // 0, 0, 2 for indices 0, 1, 2
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn turbo(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.1140890109226559, 0.06288340699912215, 0.2248337216805064);
    let c1 = vec3<f32>(6.716419496985708, 3.182286745507602, 7.571581586103393);
    let c2 = vec3<f32>(-66.09402360453038, -4.9279827041226, -10.09439367561635);
    let c3 = vec3<f32>(228.7660791526501, 25.04986699771574, -91.54105330182436);
    let c4 = vec3<f32>(-334.8351565777451, -69.31749712757485, 288.5858850615712);
    let c5 = vec3<f32>(218.7637218434795, 67.52150567819112, -305.2045772184957);
    let c6 = vec3<f32>(-52.88903478218835, -21.54527364654712, 110.5174647748972);
    let rgb = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6)))));
    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel_x = u32(in.uv.x * f32(uniforms.img_width));
    let pixel_y = u32(in.uv.y * f32(uniforms.img_height));

    if (pixel_x >= uniforms.img_width || pixel_y >= uniforms.img_height) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let idx = pixel_y * uniforms.img_width + pixel_x;
    let raw = image_data[idx];

    if (uniforms.mode == 1u) {
        let depth = bitcast<f32>(raw);
        if (depth <= 0.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        let range = max(uniforms.depth_max - uniforms.depth_min, 1e-6);
        let t = (depth - uniforms.depth_min) / range;
        return vec4<f32>(turbo(t), 1.0);
    }

    // Unpack RGBA8: R|(G<<8)|(B<<16)|(A<<24)
    let r = f32(raw & 0xFFu) / 255.0;
    let g = f32((raw >> 8u) & 0xFFu) / 255.0;
    let b = f32((raw >> 16u) & 0xFFu) / 255.0;
    let a = f32((raw >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}
