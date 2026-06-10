struct Uniforms {
    mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = u.mvp * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(v: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(v.color, 1.0);
}
