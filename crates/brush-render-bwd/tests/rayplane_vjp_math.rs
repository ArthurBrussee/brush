//! Pure-Rust finite-difference check of the RaDe-GS per-Gaussian ray-plane
//! VJP (`rayplane_vjp` in `kernels::project_backwards`). The forward and the
//! analytic VJP are transcribed verbatim from the cube kernels; the forward's
//! finite difference is the ground truth, so any VJP bug shows up here without
//! the rasterizer / autodiff coupling getting in the way.

use glam::{Mat3, Vec3};

// cube `Quat::to_mat3` ((w, x, y, z), column-major / standard rotation).
fn rmat(q: [f32; 4]) -> Mat3 {
    let [w, x, y, z] = q;
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    Mat3::from_cols(
        Vec3::new(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        Vec3::new(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
        Vec3::new(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)),
    )
}

fn normalize_q(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn mul_diag(m: Mat3, s: Vec3) -> Mat3 {
    Mat3::from_cols(m.x_axis * s.x, m.y_axis * s.y, m.z_axis * s.z)
}

// `M M^T` applied as M (M^T a).
fn cov_mul(m: Mat3, a: Vec3) -> Vec3 {
    m * (m.transpose() * a)
}

/// Forward: returns (grad_depth, normal). Mirrors `splat_view_rayplane_core`.
fn forward(
    scale: Vec3,
    quat: [f32; 4],
    mean_c: Vec3,
    view_rot: Mat3,
    fx: f32,
    fy: f32,
) -> (Vec3, Vec3) {
    let (tx, ty, tz) = (mean_c.x, mean_c.y, mean_c.z);
    let l = mean_c.length();
    let uu = tx / tz;
    let vv = ty / tz;
    let uvh = Vec3::new(uu, vv, 1.0);

    let r_c = view_rot * rmat(quat);
    let inv_s = Vec3::new(
        (1.0 / scale.x).min(1e6),
        (1.0 / scale.y).min(1e6),
        (1.0 / scale.z).min(1e6),
    );
    let m_geo = mul_diag(r_c, inv_s);
    let uvh_m = cov_mul(m_geo, uvh);

    if !uvh_m.is_finite() || uvh_m.length() <= 1e-20 {
        return (Vec3::new(0.0, 0.0, l), Vec3::new(0.0, 0.0, -1.0));
    }
    let uvh_mn = uvh_m.normalize();
    let vbn = uvh_mn.dot(uvh);
    let (u2, v2, uv) = (uu * uu, vv * vv, uu * vv);
    let ray_len2 = u2 + v2 + 1.0;
    let fn_ = l / ray_len2;
    let q = uvh_mn / vbn.max(1e-7);
    let n_j_inv = Mat3::from_cols(
        Vec3::new(v2 + 1.0, -uv, 0.0),
        Vec3::new(-uv, u2 + 1.0, 0.0),
        Vec3::new(-uu, -vv, 0.0),
    );
    let plane = n_j_inv * q;
    let grad_x = plane.x * fn_ / fx;
    let grad_y = plane.y * fn_ / fy;
    let gd_ok = grad_x.is_finite() && grad_y.is_finite();
    let grad_depth = Vec3::new(
        if gd_ok { grad_x } else { 0.0 },
        if gd_ok { grad_y } else { 0.0 },
        l,
    );

    let ray_n = Vec3::new(-plane.x * fn_, -plane.y * fn_, -1.0);
    let n_j = Mat3::from_cols(
        Vec3::new(1.0 / tz, 0.0, -tx / (tz * tz)),
        Vec3::new(0.0, 1.0 / tz, -ty / (tz * tz)),
        Vec3::new(tx / l, ty / l, tz / l),
    );
    let cn = n_j * ray_n;
    let cn_len = cn.length();
    let normal = if cn.is_finite() && cn_len > 1e-12 {
        cn / cn_len
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    (grad_depth, normal)
}

fn vec_normalize_vjp(x: Vec3, v_y: Vec3) -> Vec3 {
    let len = x.length().max(1e-12);
    let y = x / len;
    (v_y - y * y.dot(v_y)) / len
}

/// Analytic VJP. Mirrors `rayplane_vjp`. Returns (v_mean_c, v_r_world, v_scale_log).
fn vjp(
    view_rot: Mat3,
    mean_c: Vec3,
    quat: [f32; 4],
    scale: Vec3,
    fx: f32,
    fy: f32,
    v_gd: Vec3,
    v_nrm: Vec3,
) -> (Vec3, Mat3, Vec3) {
    let (tx, ty, tz) = (mean_c.x, mean_c.y, mean_c.z);
    let l = mean_c.length();
    let uu = tx / tz;
    let vv = ty / tz;
    let a = Vec3::new(uu, vv, 1.0);

    let r_c = view_rot * rmat(quat);
    let inv_s = Vec3::new(
        (1.0 / scale.x).min(1e6),
        (1.0 / scale.y).min(1e6),
        (1.0 / scale.z).min(1e6),
    );
    let m_geo = mul_diag(r_c, inv_s);
    let b = m_geo.transpose() * a;
    let w = m_geo * b;

    let wlen = w.length();
    let ok = w.is_finite() && wlen > 1e-20;
    let mask = if ok { 1.0 } else { 0.0 };
    let wn = w / wlen.max(1e-20);

    let vbn = wn.dot(a);
    let (u2, v2, uv) = (uu * uu, vv * vv, uu * vv);
    let ray_len2 = u2 + v2 + 1.0;
    let fn_ = l / ray_len2;
    let denom = vbn.max(1e-7);
    let q = wn / denom;
    let px = (v2 + 1.0) * q.x - uv * q.y - uu * q.z;
    let py = -uv * q.x + (u2 + 1.0) * q.y - vv * q.z;
    let rn = Vec3::new(-px * fn_, -py * fn_, -1.0);
    let n_j = Mat3::from_cols(
        Vec3::new(1.0 / tz, 0.0, -tx / (tz * tz)),
        Vec3::new(0.0, 1.0 / tz, -ty / (tz * tz)),
        Vec3::new(tx / l, ty / l, tz / l),
    );
    let cn = n_j * rn;

    let (v_gx, v_gy, v_dc) = (v_gd.x, v_gd.y, v_gd.z);
    let cn_mask = if cn.is_finite() && cn.length() > 1e-12 {
        1.0
    } else {
        0.0
    };
    let v_cn = vec_normalize_vjp(cn, v_nrm) * cn_mask;
    let v_rn = n_j.transpose() * v_cn;

    let mut v_px = v_rn.x * (-fn_);
    let mut v_py = v_rn.y * (-fn_);
    let mut v_fn = v_rn.x * (-px) + v_rn.y * (-py);
    v_px += v_gx * fn_ / fx;
    v_py += v_gy * fn_ / fy;
    v_fn += v_gx * px / fx + v_gy * py / fy;
    let mut v_l = v_dc;

    v_l += v_fn / ray_len2;
    let v_ray_len2 = v_fn * (-l / (ray_len2 * ray_len2));
    let mut v_u = v_ray_len2 * 2.0 * uu;
    let mut v_v = v_ray_len2 * 2.0 * vv;

    let v_q = Vec3::new(
        v_px * (v2 + 1.0) + v_py * (-uv),
        v_px * (-uv) + v_py * (u2 + 1.0),
        v_px * (-uu) + v_py * (-vv),
    );
    v_u += v_px * (-vv * q.y - q.z) + v_py * (-vv * q.x + 2.0 * uu * q.y);
    v_v += v_px * (2.0 * vv * q.x - uu * q.y) + v_py * (-uu * q.x - q.z);

    let mut v_wn = v_q / denom;
    let v_denom = -(v_q.dot(wn)) / (denom * denom);
    let v_vbn = if vbn > 1e-7 { v_denom } else { 0.0 };
    v_wn += a * v_vbn;
    let mut v_a = wn * v_vbn;

    let v_w = vec_normalize_vjp(w, v_wn);
    let c = m_geo.transpose() * v_w;
    v_a += m_geo * c;
    v_u += v_a.x;
    v_v += v_a.y;

    let v_m0 = v_w * b.x + a * c.x;
    let v_m1 = v_w * b.y + a * c.y;
    let v_m2 = v_w * b.z + a * c.z;
    let v_r_c = Mat3::from_cols(v_m0 * inv_s.x, v_m1 * inv_s.y, v_m2 * inv_s.z);
    let v_inv_sx = r_c.x_axis.dot(v_m0);
    let v_inv_sy = r_c.y_axis.dot(v_m1);
    let v_inv_sz = r_c.z_axis.dot(v_m2);
    let v_scale_log_full = Vec3::new(
        if scale.x > 1e-6 {
            -v_inv_sx / scale.x
        } else {
            0.0
        },
        if scale.y > 1e-6 {
            -v_inv_sy / scale.y
        } else {
            0.0
        },
        if scale.z > 1e-6 {
            -v_inv_sz / scale.z
        } else {
            0.0
        },
    );
    let v_r_world_full = Mat3::from_cols(
        view_rot.transpose() * v_r_c.x_axis,
        view_rot.transpose() * v_r_c.y_axis,
        view_rot.transpose() * v_r_c.z_axis,
    );

    let inv_tz2 = 1.0 / (tz * tz);
    let inv_tz3 = inv_tz2 / tz;
    let inv_l = 1.0 / l;
    let inv_l2 = inv_l * inv_l;
    let v_nj_00 = v_cn.x * rn.x;
    let v_nj_20 = v_cn.z * rn.x;
    let v_nj_11 = v_cn.y * rn.y;
    let v_nj_21 = v_cn.z * rn.y;
    let v_nj_02 = v_cn.x * rn.z;
    let v_nj_12 = v_cn.y * rn.z;
    let v_nj_22 = v_cn.z * rn.z;
    let v_tx_nj = v_nj_20 * (-inv_tz2) + v_nj_02 * inv_l;
    let v_ty_nj = v_nj_21 * (-inv_tz2) + v_nj_12 * inv_l;
    let v_tz_nj = v_nj_00 * (-inv_tz2)
        + v_nj_20 * (2.0 * tx * inv_tz3)
        + v_nj_11 * (-inv_tz2)
        + v_nj_21 * (2.0 * ty * inv_tz3)
        + v_nj_22 * inv_l;
    v_l += v_nj_02 * (-tx * inv_l2) + v_nj_12 * (-ty * inv_l2) + v_nj_22 * (-tz * inv_l2);

    let v_tx = v_u / tz + v_tx_nj + (tx * inv_l) * v_l;
    let v_ty = v_v / tz + v_ty_nj + (ty * inv_l) * v_l;
    let v_tz = v_u * (-tx * inv_tz2) + v_v * (-ty * inv_tz2) + v_tz_nj + (tz * inv_l) * v_l;
    let v_mean_c_full = Vec3::new(v_tx, v_ty, v_tz);

    let v_mean_c_deg = mean_c * (v_gd.z * inv_l);
    let v_mean_c = v_mean_c_full * mask + v_mean_c_deg * (1.0 - mask);
    let v_scale_log = v_scale_log_full * mask;
    let v_r_world = Mat3::from_cols(
        v_r_world_full.x_axis * mask,
        v_r_world_full.y_axis * mask,
        v_r_world_full.z_axis * mask,
    );
    (v_mean_c, v_r_world, v_scale_log)
}

/// Degenerate / extreme inputs must never yield NaN or Inf in the forward or
/// the VJP (collapsed scale, near edge-on, tiny depth, behind-camera-ish).
#[test]
fn rayplane_stays_finite_in_degenerate_cases() {
    let view_rot = rmat(normalize_q([0.92, 0.1, -0.2, 0.05]));
    let v_gd = Vec3::new(1.3, -0.7, 0.9);
    let v_nrm = Vec3::new(0.6, 0.4, -0.8);
    // (scale, quat, mean_c)
    let cases: &[(Vec3, [f32; 4], Vec3)] = &[
        // Flatten-collapsed axis (one scale ~ 0).
        (
            Vec3::new(0.3, 0.25, 1e-9),
            [0.9, 0.1, 0.05, 0.03],
            Vec3::new(0.2, -0.3, 1.0),
        ),
        (
            Vec3::new(1e-12, 1e-10, 0.4),
            [0.7, 0.2, 0.3, 0.1],
            Vec3::new(-0.4, 0.5, 2.0),
        ),
        // Near edge-on (thin slab aligned with the view ray).
        (
            Vec3::new(0.5, 0.5, 1e-7),
            [1.0, 0.0, 0.0, 0.0],
            Vec3::new(0.0, 0.0, 1.0),
        ),
        // Tiny / grazing depth.
        (
            Vec3::new(0.2, 0.2, 0.2),
            [0.8, 0.1, 0.1, 0.2],
            Vec3::new(2.0, 1.5, 1e-4),
        ),
        // Huge scale + off-axis.
        (
            Vec3::new(1e5, 1e4, 1e5),
            [0.5, 0.4, 0.3, 0.2],
            Vec3::new(-3.0, 2.0, 0.5),
        ),
    ];
    for (i, &(scale, quat, mean_c)) in cases.iter().enumerate() {
        let qn = normalize_q(quat);
        let (gd, n) = forward(scale, qn, mean_c, view_rot, 50.0, 47.0);
        assert!(
            gd.is_finite(),
            "forward grad_depth non-finite, case {i}: {gd:?}"
        );
        assert!(n.is_finite(), "forward normal non-finite, case {i}: {n:?}");
        let (v_mean_c, v_r, v_scale) = vjp(view_rot, mean_c, qn, scale, 50.0, 47.0, v_gd, v_nrm);
        assert!(
            v_mean_c.is_finite(),
            "v_mean_c non-finite, case {i}: {v_mean_c:?}"
        );
        assert!(
            v_scale.is_finite(),
            "v_scale non-finite, case {i}: {v_scale:?}"
        );
        assert!(
            v_r.x_axis.is_finite() && v_r.y_axis.is_finite() && v_r.z_axis.is_finite(),
            "v_r_world non-finite, case {i}",
        );
    }
}

// Loss = v_gd·grad_depth + v_nrm·normal for fixed weights.
fn loss(
    scale: Vec3,
    quat: [f32; 4],
    mean_c: Vec3,
    view_rot: Mat3,
    fx: f32,
    fy: f32,
    v_gd: Vec3,
    v_nrm: Vec3,
) -> f32 {
    let (gd, n) = forward(scale, normalize_q(quat), mean_c, view_rot, fx, fy);
    v_gd.dot(gd) + v_nrm.dot(n)
}

#[test]
fn rayplane_vjp_matches_finite_diff() {
    // A non-axis-aligned view rotation so the cov path is exercised.
    let view_rot = rmat(normalize_q([0.92, 0.1, -0.2, 0.05]));
    let fx = 50.0;
    let fy = 47.0;
    let v_gd = Vec3::new(1.3, -0.7, 0.9);
    let v_nrm = Vec3::new(0.6, 0.4, -0.8);

    let cases: &[(Vec3, [f32; 4], Vec3)] = &[
        (
            Vec3::new(0.2, -0.3, 1.0),
            [0.9, 0.1, 0.05, 0.03],
            Vec3::new(0.25, 0.18, 0.3),
        ),
        (
            Vec3::new(-0.4, 0.5, 1.0),
            [0.7, 0.2, 0.3, 0.1],
            Vec3::new(0.4, 0.2, 0.15),
        ),
        (
            Vec3::new(0.1, 0.2, 1.0),
            [0.5, 0.4, 0.3, 0.2],
            Vec3::new(0.12, 0.4, 0.22),
        ),
    ];

    let eps = 1e-3_f32;
    let mut fails = Vec::new();
    for (ci, &(mean_c, quat, log_scale)) in cases.iter().enumerate() {
        let scale = Vec3::new(log_scale.x.exp(), log_scale.y.exp(), log_scale.z.exp());
        let qn = normalize_q(quat);
        let (v_mean_c, v_r_world, v_scale_log) =
            vjp(view_rot, mean_c, qn, scale, fx, fy, v_gd, v_nrm);
        // quat grad: chain v_r_world through quat_to_mat + normalize.
        let v_q = apply_normalize_vjp(quat, quat_to_mat_vjp(qn, v_r_world));

        let fd = |f: &dyn Fn(f32) -> f32, x: f32| {
            let (a, b) = (f(x + eps), f(x - eps));
            (a - b) / (2.0 * eps)
        };

        // mean_c components.
        for i in 0..3 {
            let num = fd(
                &|d| {
                    let mut mm = mean_c;
                    mm[i] = mean_c[i] + d;
                    loss(scale, quat, mm, view_rot, fx, fy, v_gd, v_nrm)
                },
                0.0,
            );
            check(&mut fails, format!("mean_c[{ci},{i}]"), num, v_mean_c[i]);
        }
        // log_scale components.
        for i in 0..3 {
            let num = fd(
                &|d| {
                    let mut ls = log_scale;
                    ls[i] = log_scale[i] + d;
                    let s = Vec3::new(ls.x.exp(), ls.y.exp(), ls.z.exp());
                    loss(s, quat, mean_c, view_rot, fx, fy, v_gd, v_nrm)
                },
                0.0,
            );
            check(
                &mut fails,
                format!("log_scale[{ci},{i}]"),
                num,
                v_scale_log[i],
            );
        }
        // quat (unnormalized) components.
        for i in 0..4 {
            let num = fd(
                &|d| {
                    let mut qq = quat;
                    qq[i] = quat[i] + d;
                    loss(scale, qq, mean_c, view_rot, fx, fy, v_gd, v_nrm)
                },
                0.0,
            );
            check(&mut fails, format!("quat[{ci},{i}]"), num, v_q[i]);
        }
    }
    assert!(
        fails.is_empty(),
        "rayplane VJP mismatches:\n{}",
        fails.join("\n")
    );
}

fn check(fails: &mut Vec<String>, name: String, num: f32, an: f32) {
    let scale = num.abs().max(an.abs()).max(1e-6);
    if (num - an).abs() > 1e-2 * scale + 1e-5 {
        fails.push(format!(
            "  {name}: numerical {num:.6} vs analytical {an:.6}"
        ));
    }
}

fn quat_to_mat_vjp(q: [f32; 4], v_r: Mat3) -> [f32; 4] {
    let [qw, qx, qy, qz] = q;
    // v_r columns: x_axis=(c0_x,c0_y,c0_z), etc. cube uses c{col}_{row}.
    let (c0x, c0y, c0z) = (v_r.x_axis.x, v_r.x_axis.y, v_r.x_axis.z);
    let (c1x, c1y, c1z) = (v_r.y_axis.x, v_r.y_axis.y, v_r.y_axis.z);
    let (c2x, c2y, c2z) = (v_r.z_axis.x, v_r.z_axis.y, v_r.z_axis.z);
    let w_grad = qx * (c1z - c2y) + qy * (c2x - c0z) + qz * (c0y - c1x);
    let x_grad = -2.0 * qx * (c1y + c2z) + qy * (c0y + c1x) + qz * (c0z + c2x) + qw * (c1z - c2y);
    let y_grad = qx * (c0y + c1x) - 2.0 * qy * (c0x + c2z) + qz * (c1z + c2y) + qw * (c2x - c0z);
    let z_grad = qx * (c0z + c2x) + qy * (c1z + c2y) - 2.0 * qz * (c0x + c1y) + qw * (c0y - c1x);
    [2.0 * w_grad, 2.0 * x_grad, 2.0 * y_grad, 2.0 * z_grad]
}

fn apply_normalize_vjp(q: [f32; 4], g: [f32; 4]) -> [f32; 4] {
    let [qw, qx, qy, qz] = q;
    let lsq = qw * qw + qx * qx + qy * qy + qz * qz;
    let l = lsq.sqrt();
    let inv = 1.0 / (l * lsq);
    let [gw, gx, gy, gz] = g;
    let cc0 = -qw * qx;
    let cc1 = -qx * qy;
    let cc2 = -qy * qw;
    let cs0 = -qw * qz;
    let cs1 = -qx * qz;
    let cs2 = -qy * qz;
    [
        ((lsq - qw * qw) * gw + cc0 * gx + cc2 * gy + cs0 * gz) * inv,
        (cc0 * gw + (lsq - qx * qx) * gx + cc1 * gy + cs1 * gz) * inv,
        (cc2 * gw + cc1 * gx + (lsq - qy * qy) * gy + cs2 * gz) * inv,
        (cs0 * gw + cs1 * gx + cs2 * gy + (lsq - qz * qz) * gz) * inv,
    ]
}
