#import helpers;
#import sh;

@group(0) @binding(0) var<storage, read> transforms: array<f32>;
@group(0) @binding(1) var<storage, read> raw_opac: array<f32>;
@group(0) @binding(2) var<storage, read> global_from_compact_gid: array<u32>;
// Sparse rasterize grads (indexed by compact_gid, stride 10): projected_splat_grads(8) + opac_grad(1) + refine_weight(1)
@group(0) @binding(3) var<storage, read_write> v_rasterize_grads: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_transforms: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_coeffs: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_raw_opac: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_refine_weight: array<f32>;
@group(0) @binding(8) var<storage, read> uniforms: helpers::ProjectUniforms;

fn write_coeffs(base_id: ptr<function, u32>, val: vec3f) {
    v_coeffs[*base_id + 0] = val.x;
    v_coeffs[*base_id + 1] = val.y;
    v_coeffs[*base_id + 2] = val.z;
    *base_id += 3;
}

fn normalize_vjp(quat: vec4f) -> mat4x4f {
    let quat_sqr = quat * quat;
    let quat_len_sqr = dot(quat, quat);
    let quat_len = sqrt(quat_len_sqr);

    let cross_complex = -quat.xyz * quat.yzx;
    let cross_scalar = -quat.xyz * quat.w;

    return mat4x4f(
        vec4f(quat_len_sqr - quat_sqr.x, cross_complex.x, cross_complex.z, cross_scalar.x),
        vec4f(cross_complex.x, quat_len_sqr - quat_sqr.y, cross_complex.y, cross_scalar.y),
        vec4f(cross_complex.z, cross_complex.y, quat_len_sqr - quat_sqr.z, cross_scalar.z),
        vec4f(cross_scalar.x, cross_scalar.y, cross_scalar.z, quat_len_sqr - quat_sqr.w),
    ) * (1.0 / (quat_len * quat_len_sqr));
}

fn quat_to_mat_vjp(quat: vec4f, v_R: mat3x3f) -> vec4f {
    let w = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    return vec4f(
        // w element stored in x field
        (
            x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
            z * (v_R[0][1] - v_R[1][0])
        ),
        // x element in y field
        (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        ),
        // y element in z field
        (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        ),
        // z element in w field
        (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        )
    ) * 2.0;
}

fn inverse_vjp(Minv: mat2x2f, v_Minv: mat2x2f) -> mat2x2f {
    // P = M^-1
    // b3_c4_/dM = -P * b3_c4_/dP * P
    return mat2x2f(-Minv[0], -Minv[1]) * v_Minv * Minv;
}

fn outer_product(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3f(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z
    );
}

fn persp_proj_vjp(
    J: mat3x2f,
    // fwd inputs
    mean3d: vec3f,
    cov3d: mat3x3f,
    focal: vec2f,
    pixel_center: vec2f,
    img_size: vec2u,
    // grad outputs
    v_cov2d: mat2x2f,
    v_mean2d: vec2f,
) -> vec3f {
    let x = mean3d.x;
    let y = mean3d.y;
    let z = mean3d.z;

    let rz = 1.0 / mean3d.z;
    let rz2 = rz * rz;

    // b3_c4_/dx = fx * rz * b3_c4_/dpixx
    // b3_c4_/dy = fy * rz * b3_c4_/dpixy
    // b3_c4_/dz = - fx * mean.x * rz2 * b3_c4_/dpixx - fy * mean.y * rz2 * b3_c4_/dpixy
    var v_mean3d = vec3f(
        focal.x * rz * v_mean2d[0],
        focal.y * rz * v_mean2d[1],
        -(focal.x * x * v_mean2d[0] + focal.y * y * v_mean2d[1]) * rz2
    );

    // b3_c4_/dx = -fx * rz2 * b3_c4_/dJ_02
    // b3_c4_/dy = -fy * rz2 * b3_c4_/dJ_12
    // b3_c4_/dz = -fx * rz2 * b3_c4_/dJ_00 - fy * rz2 * b3_c4_/dJ_11
    //         + 2 * fx * tx * rz3 * b3_c4_/dJ_02 + 2 * fy * ty * rz3
    let rz3 = rz2 * rz;
    let v_J = v_cov2d * J * transpose(cov3d) + transpose(v_cov2d) * J * cov3d;

    let tan_fov = 0.5 * vec2f(img_size.xy) / focal;

    let lims_pos = (vec2f(img_size.xy) - pixel_center) / focal + 0.3f * tan_fov;
    let lims_neg = pixel_center / focal + 0.3f * tan_fov;
    // Get ndc coords +- clipped to the frustum.
    let t = mean3d.z * clamp(mean3d.xy * rz, -lims_neg, lims_pos);

    let lim_x_pos = lims_pos.x;
    let lim_x_neg = lims_neg.x;
    let lim_y_pos = lims_pos.y;
    let lim_y_neg = lims_neg.y;
    let tx = t.x;
    let ty = t.y;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -focal.x * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -focal.x * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -focal.y * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -focal.y * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -focal.x * rz2 * v_J[0][0] - focal.y * rz2 * v_J[1][1] +
                  2.f * focal.x * tx * rz3 * v_J[2][0] +
                  2.f * focal.y * ty * rz3 * v_J[2][1];

    // add contribution from v_depths
    // Disabled as there is no depth supervision currently.
    // v_mean3d.z += v_depths[0];

    return v_mean3d;
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

    // Read upstream rasterize grads first. rasterize_bwd only writes for
    // splats that contributed to a pixel; non-contributing splats leave
    // v_rasterize_grads at its zero-init value. For those, every output
    // gradient (v_transforms, v_coeffs, v_raw_opac, v_refine_weight) is
    // also zero — and since the dense output buffers are zero-init, we
    // can return without writing anything at all.
    let rg_base = compact_gid * 10u;
    let v_mean2d = vec2f(v_rasterize_grads[rg_base], v_rasterize_grads[rg_base + 1u]);
    let v_conics = vec3f(v_rasterize_grads[rg_base + 2u], v_rasterize_grads[rg_base + 3u], v_rasterize_grads[rg_base + 4u]);
    let v_color = vec3f(v_rasterize_grads[rg_base + 5u], v_rasterize_grads[rg_base + 6u], v_rasterize_grads[rg_base + 7u]);
    let v_alpha_in = v_rasterize_grads[rg_base + 8u];
    let v_refine_in = v_rasterize_grads[rg_base + 9u];

    let any_grad =
        v_mean2d.x != 0.0f || v_mean2d.y != 0.0f ||
        v_conics.x != 0.0f || v_conics.y != 0.0f || v_conics.z != 0.0f ||
        v_color.x != 0.0f || v_color.y != 0.0f || v_color.z != 0.0f ||
        v_alpha_in != 0.0f || v_refine_in != 0.0f;

    if !any_grad {
        return;
    }

    let viewmat = uniforms.viewmat;
    let focal = uniforms.focal;
    let img_size = uniforms.img_size;
    let pixel_center = uniforms.pixel_center;

    // Read transform data: means(3) + quats(4) + log_scales(3)
    let tbase = global_gid * 10u;
    let mean = vec3f(transforms[tbase], transforms[tbase + 1u], transforms[tbase + 2u]);
    let scale = exp(vec3f(transforms[tbase + 7u], transforms[tbase + 8u], transforms[tbase + 9u]));
    let quat_unorm = vec4f(transforms[tbase + 3u], transforms[tbase + 4u], transforms[tbase + 5u], transforms[tbase + 6u]);
    // Safe to normalize, quats with norm 0 are invisible.
    let quat = normalize(quat_unorm);

    let viewdir = normalize(mean - uniforms.camera_position.xyz);

    let sh_degree = uniforms.sh_degree;
    let v_coeff = sh::sh_coeffs_to_color_vjp(sh_degree, viewdir, v_color);
    let num_coeffs = sh::num_sh_coeffs(sh_degree);
    var base_id = global_gid * num_coeffs * 3;

    write_coeffs(&base_id, v_coeff.b0_c0_);
    if sh_degree > 0 {
        write_coeffs(&base_id, v_coeff.b1_c0_);
        write_coeffs(&base_id, v_coeff.b1_c1_);
        write_coeffs(&base_id, v_coeff.b1_c2_);
        if sh_degree > 1 {
            write_coeffs(&base_id, v_coeff.b2_c0_);
            write_coeffs(&base_id, v_coeff.b2_c1_);
            write_coeffs(&base_id, v_coeff.b2_c2_);
            write_coeffs(&base_id, v_coeff.b2_c3_);
            write_coeffs(&base_id, v_coeff.b2_c4_);
            if sh_degree > 2 {
                write_coeffs(&base_id, v_coeff.b3_c0_);
                write_coeffs(&base_id, v_coeff.b3_c1_);
                write_coeffs(&base_id, v_coeff.b3_c2_);
                write_coeffs(&base_id, v_coeff.b3_c3_);
                write_coeffs(&base_id, v_coeff.b3_c4_);
                write_coeffs(&base_id, v_coeff.b3_c5_);
                write_coeffs(&base_id, v_coeff.b3_c6_);
                if sh_degree > 3 {
                    write_coeffs(&base_id, v_coeff.b4_c0_);
                    write_coeffs(&base_id, v_coeff.b4_c1_);
                    write_coeffs(&base_id, v_coeff.b4_c2_);
                    write_coeffs(&base_id, v_coeff.b4_c3_);
                    write_coeffs(&base_id, v_coeff.b4_c4_);
                    write_coeffs(&base_id, v_coeff.b4_c5_);
                    write_coeffs(&base_id, v_coeff.b4_c6_);
                    write_coeffs(&base_id, v_coeff.b4_c7_);
                    write_coeffs(&base_id, v_coeff.b4_c8_);
                }
            }
        }
    }

    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let mean_c = R * mean + viewmat[3].xyz;

    let rz = 1.0 / mean_c.z;
    let rz2 = rz * rz;

    let rotmat = helpers::quat_to_mat(quat);
    let S = helpers::scale_to_mat(scale);
    let M = rotmat * S;

    var cov2d = helpers::calc_cov2d(scale, quat, mean_c, focal, img_size, pixel_center, viewmat);

    let filter_comp = helpers::compensate_cov2d(&cov2d);
    let opac = helpers::sigmoid(raw_opac[global_gid]);
    // Write opacity gradient to dense output buffer (scatter compact→global happens here).
    v_raw_opac[global_gid] = filter_comp * v_alpha_in * opac * (1.0 - opac);
    // Write refine weight to dense output buffer (scatter compact→global).
    v_refine_weight[global_gid] = v_refine_in;

    let covar2d_inv = helpers::inverse(cov2d);
    let v_covar2d_inv = mat2x2f(vec2f(v_conics.x, v_conics.y * 0.5f), vec2f(v_conics.y * 0.5f, v_conics.z));
    let v_covar2d = inverse_vjp(covar2d_inv, v_covar2d_inv);

    // covar_world_to_cam
    let covar = M * transpose(M);
    let covar_c = R * covar * transpose(R);

    // persp_proj_vjp
    let J = helpers::calc_cam_J(mean_c, focal, img_size, pixel_center);
    let v_mean_c = persp_proj_vjp(J, mean_c, covar_c, focal, pixel_center, img_size, v_covar2d, v_mean2d);
    // cov = J * V * Jt; G = b3_c4_/dcov = v_cov
    // -> b3_c4_/dV = Jt * G * J
    // -> b3_c4_/dJ = G * J * Vt + Gt * J * V
    let v_covar_c = transpose(J) * v_covar2d * J;

    // b3_c4_/dx = -fx * rz2 * b3_c4_/dJ_02
    // b3_c4_/dy = -fy * rz2 * b3_c4_/dJ_12
    // b3_c4_/dz = -fx * rz2 * b3_c4_/dJ_00 - fy * rz2 * b3_c4_/dJ_11
    //         + 2 * fx * tx * rz3 * b3_c4_/dJ_02 + 2 * fy * ty * rz3
    // for D = W * X, G = b3_c4_/dD
    // b3_c4_/dW = G * XT, b3_c4_/dX = WT * G

    // TODO: Camera gradient is not done yet.
    // var v_R = outer_product(v_mean_c, mean);
    let v_mean = transpose(R) * v_mean_c;

    // covar_world_to_cam_vjp
    // TODO: Camera gradient is not done yet.
    // v_R += v_covar_c * R * transpose(covar) +
    //        transpose(v_covar_c) * R * covar;

    let v_covar = transpose(R) * v_covar_c * R;

    // quat_scale_to_covar_vjp
    // TODO: Merge with cov calculation.

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = b3_c4_/dD
    // b3_c4_/dW = G * XT, b3_c4_/dX = WT * G
    // so
    // for D = M * Mt,
    // b3_c4_/dM = b3_c4_/dM + b3_c4_/dMt = G * M + (Mt * G)t = G * M + Gt * M
    let v_M = (v_covar + transpose(v_covar)) * M;

    let v_scale = vec3f(
        dot(rotmat[0], v_M[0]),
        dot(rotmat[1], v_M[1]),
        dot(rotmat[2], v_M[2]),
    );
    let v_scale_exp = v_scale * scale;

    // grad for (quat, scale) from covar
    let v_quat = normalize_vjp(quat_unorm) * quat_to_mat_vjp(quat, v_M * S);

    // Write gradients to dense v_transforms: means(3) + quats(4) + log_scales(3)
    let vbase = global_gid * 10u;
    v_transforms[vbase]      = v_mean.x;
    v_transforms[vbase + 1u] = v_mean.y;
    v_transforms[vbase + 2u] = v_mean.z;
    v_transforms[vbase + 3u] = v_quat.x;
    v_transforms[vbase + 4u] = v_quat.y;
    v_transforms[vbase + 5u] = v_quat.z;
    v_transforms[vbase + 6u] = v_quat.w;
    v_transforms[vbase + 7u] = v_scale_exp.x;
    v_transforms[vbase + 8u] = v_scale_exp.y;
    v_transforms[vbase + 9u] = v_scale_exp.z;
}
