struct Uniforms {
    img_width: u32,
    img_height: u32,
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel_x = u32(in.uv.x * f32(uniforms.img_width));
    let pixel_y = u32(in.uv.y * f32(uniforms.img_height));

    if (pixel_x >= uniforms.img_width || pixel_y >= uniforms.img_height) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let idx = pixel_y * uniforms.img_width + pixel_x;
    let packed = image_data[idx];

    // Unpack RGBA8: R|(G<<8)|(B<<16)|(A<<24)
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}
