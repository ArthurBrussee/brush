#import helpers;

struct IsectInfo {
    compact_gid: u32,
    tile_id: u32,
}

@group(0) @binding(0) var<storage, read> num_visible: u32;
@group(0) @binding(1) var<storage, read> means: array<helpers::PackedVec3>;
@group(0) @binding(2) var<storage, read> log_scales: array<helpers::PackedVec3>;
@group(0) @binding(3) var<storage, read> quats: array<vec4f>;
@group(0) @binding(4) var<storage, read> coeffs: array<helpers::PackedVec3>;
@group(0) @binding(5) var<storage, read> raw_opacities: array<f32>;
@group(0) @binding(6) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(7) var<storage, read_write> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(8) var<storage, read_write> splat_intersect_counts: array<u32>;
// Uniforms via with_metadata (always last binding)
@group(0) @binding(9) var<storage, read> uniforms: helpers::ProjectUniforms;

struct ShCoeffs {
    b0_c0: vec3f,

    b1_c0: vec3f,
    b1_c1: vec3f,
    b1_c2: vec3f,

    b2_c0: vec3f,
    b2_c1: vec3f,
    b2_c2: vec3f,
    b2_c3: vec3f,
    b2_c4: vec3f,

    b3_c0: vec3f,
    b3_c1: vec3f,
    b3_c2: vec3f,
    b3_c3: vec3f,
    b3_c4: vec3f,
    b3_c5: vec3f,
    b3_c6: vec3f,

    b4_c0: vec3f,
    b4_c1: vec3f,
    b4_c2: vec3f,
    b4_c3: vec3f,
    b4_c4: vec3f,
    b4_c5: vec3f,
    b4_c6: vec3f,
    b4_c7: vec3f,
    b4_c8: vec3f,
}

const SH_C0: f32 = 0.2820947917738781f;

// Evaluate spherical harmonics bases at unit direction for high orders using approach described by
// Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
// See https://jcgt.org/published/0002/02/06/ for reference implementation
fn sh_coeffs_to_color(
    degree: u32,
    viewdir: vec3f,
    sh: ShCoeffs,
) -> vec3f {
    var colors = SH_C0 * sh.b0_c0;

    if (degree == 0) {
        return colors;
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292f;
    colors += fTmp0A *
                    (-y * sh.b1_c0 +
                    z * sh.b1_c1 -
                    x * sh.b1_c2);

    if (degree == 1) {
        return colors;
    }
    let z2 = z * z;

    let fTmp0B = -1.092548430592079f * z;
    let fTmp1A = 0.5462742152960395f;
    let fC1 = x * x - y * y;
    let fS1 = 2.f * x * y;
    let pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;

    colors +=
        pSH4 * sh.b2_c0 +
        pSH5 * sh.b2_c1 +
        pSH6 * sh.b2_c2 +
        pSH7 * sh.b2_c3 +
        pSH8 * sh.b2_c4;

    if (degree == 2) {
        return colors;
    }

    let fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    let fTmp1B = 1.445305721320277f * z;
    let fTmp2A = -0.5900435899266435f;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    let pSH13 = fTmp0C * x;
    let pSH11 = fTmp0C * y;
    let pSH14 = fTmp1B * fC1;
    let pSH10 = fTmp1B * fS1;
    let pSH15 = fTmp2A * fC2;
    let pSH9  = fTmp2A * fS2;
    colors +=   pSH9  * sh.b3_c0 +
                pSH10 * sh.b3_c1 +
                pSH11 * sh.b3_c2 +
                pSH12 * sh.b3_c3 +
                pSH13 * sh.b3_c4 +
                pSH14 * sh.b3_c5 +
                pSH15 * sh.b3_c6;

    if (degree == 3) {
        return colors;
    }

    let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    let fTmp2B = -1.770130769779931f * z;
    let fTmp3A = 0.6258357354491763f;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
    let pSH21 = fTmp0D * x;
    let pSH19 = fTmp0D * y;
    let pSH22 = fTmp1C * fC1;
    let pSH18 = fTmp1C * fS1;
    let pSH23 = fTmp2B * fC2;
    let pSH17 = fTmp2B * fS2;
    let pSH24 = fTmp3A * fC3;
    let pSH16 = fTmp3A * fS3;
    colors += pSH16 * sh.b4_c0 +
                pSH17 * sh.b4_c1 +
                pSH18 * sh.b4_c2 +
                pSH19 * sh.b4_c3 +
                pSH20 * sh.b4_c4 +
                pSH21 * sh.b4_c5 +
                pSH22 * sh.b4_c6 +
                pSH23 * sh.b4_c7 +
                pSH24 * sh.b4_c8;
    return colors;
}

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1) * (degree + 1);
}

fn read_coeffs(base_id: ptr<function, u32>) -> vec3f {
    let ret = helpers::as_vec(coeffs[*base_id]);
    *base_id += 1u;
    return ret;
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

    if compact_gid >= num_visible {
        return;
    }

    let global_gid = global_from_compact_gid[compact_gid];

    // Project world space to camera space.
    let mean = helpers::as_vec(means[global_gid]);
    let scale = exp(helpers::as_vec(log_scales[global_gid]));

    // Safe to normalize, splats with length(quat) == 0 are invisible.
    let quat = normalize(quats[global_gid]);
    var opac = helpers::sigmoid(raw_opacities[global_gid]);

    let viewmat = uniforms.viewmat;
    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let mean_c = R * mean + viewmat[3].xyz;

    let covar = helpers::calc_cov3d(scale, quat);
    var cov2d = helpers::calc_cov2d(covar, mean_c, uniforms.focal, uniforms.img_size, uniforms.pixel_center, viewmat);
    opac *= helpers::compensate_cov2d(&cov2d);

    let conic = helpers::inverse(cov2d);

    // compute the projected mean
    let rz = 1.0 / mean_c.z;
    let mean2d = uniforms.focal * mean_c.xy * rz + uniforms.pixel_center;

    let sh_degree = uniforms.sh_degree;
    let num_coeffs = num_sh_coeffs(sh_degree);
    var base_id = u32(global_gid) * num_coeffs;

    var sh = ShCoeffs();
    sh.b0_c0 = read_coeffs(&base_id);

    if sh_degree >= 1 {
        sh.b1_c0 = read_coeffs(&base_id);
        sh.b1_c1 = read_coeffs(&base_id);
        sh.b1_c2 = read_coeffs(&base_id);

        if sh_degree >= 2 {
            sh.b2_c0 = read_coeffs(&base_id);
            sh.b2_c1 = read_coeffs(&base_id);
            sh.b2_c2 = read_coeffs(&base_id);
            sh.b2_c3 = read_coeffs(&base_id);
            sh.b2_c4 = read_coeffs(&base_id);

            if sh_degree >= 3 {
                sh.b3_c0 = read_coeffs(&base_id);
                sh.b3_c1 = read_coeffs(&base_id);
                sh.b3_c2 = read_coeffs(&base_id);
                sh.b3_c3 = read_coeffs(&base_id);
                sh.b3_c4 = read_coeffs(&base_id);
                sh.b3_c5 = read_coeffs(&base_id);
                sh.b3_c6 = read_coeffs(&base_id);

                if sh_degree >= 4 {
                    sh.b4_c0 = read_coeffs(&base_id);
                    sh.b4_c1 = read_coeffs(&base_id);
                    sh.b4_c2 = read_coeffs(&base_id);
                    sh.b4_c3 = read_coeffs(&base_id);
                    sh.b4_c4 = read_coeffs(&base_id);
                    sh.b4_c5 = read_coeffs(&base_id);
                    sh.b4_c6 = read_coeffs(&base_id);
                    sh.b4_c7 = read_coeffs(&base_id);
                    sh.b4_c8 = read_coeffs(&base_id);
                }
            }
        }
    }

    // Write projected splat information.
    let viewdir = normalize(mean - uniforms.camera_position.xyz);
    var color = sh_coeffs_to_color(sh_degree, viewdir, sh) + vec3f(0.5);

    let conic_packed = vec3f(conic[0][0], conic[0][1], conic[1][1]);

    projected[compact_gid] = helpers::create_projected_splat(
        mean2d,
        conic_packed,
        vec4f(color, opac)
    );

    // Count intersections for this splat (merged from map_gaussian_to_intersects prepass)
    let power_threshold = log(opac * 255.0);
    let extent = helpers::compute_bbox_extent(cov2d, power_threshold);
    let tile_bbox = helpers::get_tile_bbox(mean2d, extent, uniforms.tile_bounds);
    let tile_bbox_min = tile_bbox.xy;
    let tile_bbox_max = tile_bbox.zw;

    var num_tiles_hit = 0u;
    let tile_bbox_width = tile_bbox_max.x - tile_bbox_min.x;
    let num_tiles_bbox = (tile_bbox_max.y - tile_bbox_min.y) * tile_bbox_width;

    for (var tile_idx = 0u; tile_idx < num_tiles_bbox; tile_idx++) {
        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min.x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min.y;

        let rect = helpers::tile_rect(vec2u(tx, ty));
        if helpers::will_primitive_contribute(rect, mean2d, conic_packed, power_threshold) {
            num_tiles_hit += 1u;
        }
    }

    splat_intersect_counts[compact_gid] = num_tiles_hit;
}
