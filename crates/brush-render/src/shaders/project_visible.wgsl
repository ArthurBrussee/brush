enable f16;
#import helpers;
#import sh;

@group(0) @binding(0) var<storage, read> transforms: array<f32>;
@group(0) @binding(1) var<storage, read> coeffs_dc: array<helpers::PackedVec3>;
@group(0) @binding(2) var<storage, read> coeffs_rest: array<helpers::PackedVec3>;
@group(0) @binding(3) var<storage, read> raw_opacities: array<f32>;
@group(0) @binding(4) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(5) var<storage, read_write> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(6) var<storage, read> uniforms: helpers::ProjectUniforms;

fn read_dc(gid: u32) -> vec3f {
    let c = coeffs_dc[gid];
    return vec3f(c.x, c.y, c.z);
}

fn read_rest(base_id: ptr<function, u32>) -> vec3f {
    let c = coeffs_rest[*base_id];
    *base_id += 1u;
    return vec3f(c.x, c.y, c.z);
}

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let compact_gid = helpers::get_global_id(wid, num_wgs, lid, WG_SIZE);

    if compact_gid >= uniforms.num_visible {
        return;
    }

    let global_gid = global_from_compact_gid[compact_gid];

    // Read transform data: means(3) + quats(4) + log_scales(3)
    let base = global_gid * 10u;
    let mean = vec3f(transforms[base], transforms[base + 1u], transforms[base + 2u]);
    let scale = exp(vec3f(transforms[base + 7u], transforms[base + 8u], transforms[base + 9u]));

    // Safe to normalize, splats with length(quat) == 0 are invisible.
    let quat = normalize(vec4f(transforms[base + 3u], transforms[base + 4u], transforms[base + 5u], transforms[base + 6u]));
    var opac = helpers::sigmoid(raw_opacities[global_gid]);

    let viewmat = uniforms.viewmat;
    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let mean_c = R * mean + viewmat[3].xyz;

    var cov2d = helpers::calc_cov2d(scale, quat, mean_c, uniforms.focal, uniforms.img_size, uniforms.pixel_center, viewmat);
    opac *= helpers::compensate_cov2d(&cov2d);

    let conic = helpers::inverse(cov2d);

    // compute the projected mean
    let rz = 1.0 / mean_c.z;
    let mean2d = uniforms.focal * mean_c.xy * rz + uniforms.pixel_center;

    let sh_degree = uniforms.sh_degree;
    // DC is read from separate f32 buffer, rest from f16 buffer.
    let num_rest = sh::num_sh_coeffs(sh_degree) - 1u;
    var rest_id = u32(global_gid) * num_rest;

    var coeffs = sh::ShCoeffs();
    coeffs.dc = read_dc(global_gid);

    if sh_degree >= 1 {
        coeffs.ba = read_rest(&rest_id);
        coeffs.bb = read_rest(&rest_id);
        coeffs.bc = read_rest(&rest_id);

        if sh_degree >= 2 {
            coeffs.ca = read_rest(&rest_id);
            coeffs.cb = read_rest(&rest_id);
            coeffs.cc = read_rest(&rest_id);
            coeffs.cd = read_rest(&rest_id);
            coeffs.ce = read_rest(&rest_id);

            if sh_degree >= 3 {
                coeffs.da = read_rest(&rest_id);
                coeffs.db = read_rest(&rest_id);
                coeffs.dd = read_rest(&rest_id);
                coeffs.de = read_rest(&rest_id);
                coeffs.df = read_rest(&rest_id);
                coeffs.dg = read_rest(&rest_id);
                coeffs.dh = read_rest(&rest_id);

                if sh_degree >= 4 {
                    coeffs.ea = read_rest(&rest_id);
                    coeffs.eb = read_rest(&rest_id);
                    coeffs.ec = read_rest(&rest_id);
                    coeffs.ed = read_rest(&rest_id);
                    coeffs.ee = read_rest(&rest_id);
                    coeffs.ef = read_rest(&rest_id);
                    coeffs.eg = read_rest(&rest_id);
                    coeffs.eh = read_rest(&rest_id);
                    coeffs.ei = read_rest(&rest_id);
                }
            }
        }
    }

    // Write projected splat information.
    let viewdir = normalize(mean - uniforms.camera_position.xyz);
    var color = sh::sh_coeffs_to_color(sh_degree, viewdir, coeffs) + vec3f(0.5);

    let conic_packed = vec3f(conic[0][0], conic[0][1], conic[1][1]);

    projected[compact_gid] = helpers::create_projected_splat(
        mean2d,
        conic_packed,
        vec4f(color, opac)
    );
}
