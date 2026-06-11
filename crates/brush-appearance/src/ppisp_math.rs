//! Shared PPISP math as `#[cube]` functions: the homography construction,
//! CRF tone curve and vignetting falloff with their hand-derived backwards.
//! Used by both the per-frame/per-camera PPISP kernels (`ppisp_kernels`) and
//! the hybrid PPISP-grid kernels (`ppisp_grid_kernels`).

use brush_cube::Vec3A;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub(crate) const LN2: f32 = core::f32::consts::LN_2;

// ZCA pinv blocks mapping latent color offsets to real chromaticity
// offsets, row-major 2x2 (`m00 m01 m10 m11`) per control point
// [Blue, Red, Green, Neutral]. Plain consts so they inline into cube code.
const ZCA_B00: f32 = 0.048_054_2;
const ZCA_B01: f32 = -0.004_363_1;
const ZCA_B10: f32 = -0.004_363_1;
const ZCA_B11: f32 = 0.048_128_3;
const ZCA_R00: f32 = 0.058_057;
const ZCA_R01: f32 = -0.017_987_2;
const ZCA_R10: f32 = -0.017_987_2;
const ZCA_R11: f32 = 0.043_106_1;
const ZCA_G00: f32 = 0.043_333_6;
const ZCA_G01: f32 = -0.018_053_7;
const ZCA_G10: f32 = -0.018_053_7;
const ZCA_G11: f32 = 0.058_05;
const ZCA_N00: f32 = 0.012_836_9;
const ZCA_N01: f32 = -0.003_465_4;
const ZCA_N10: f32 = -0.003_465_4;
const ZCA_N11: f32 = 0.012_815_8;

// ---------------------------------------------------------------------------
// Small row-major 3x3 matrix (the reference CUDA code is row-major; keeping
// the same layout makes the port 1:1 auditable).
// ---------------------------------------------------------------------------

#[derive(CubeType, CubeTypeMut, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct M3 {
    pub m00: f32,
    pub m01: f32,
    pub m02: f32,
    pub m10: f32,
    pub m11: f32,
    pub m12: f32,
    pub m20: f32,
    pub m21: f32,
    pub m22: f32,
}

#[cube]
impl M3 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        m00: f32,
        m01: f32,
        m02: f32,
        m10: f32,
        m11: f32,
        m12: f32,
        m20: f32,
        m21: f32,
        m22: f32,
    ) -> M3 {
        M3 {
            m00,
            m01,
            m02,
            m10,
            m11,
            m12,
            m20,
            m21,
            m22,
        }
    }

    fn row0(self) -> Vec3A {
        Vec3A::new(self.m00, self.m01, self.m02)
    }
    fn row1(self) -> Vec3A {
        Vec3A::new(self.m10, self.m11, self.m12)
    }
    fn row2(self) -> Vec3A {
        Vec3A::new(self.m20, self.m21, self.m22)
    }

    fn from_rows(r0: Vec3A, r1: Vec3A, r2: Vec3A) -> M3 {
        M3::new(
            r0.x(),
            r0.y(),
            r0.z(),
            r1.x(),
            r1.y(),
            r1.z(),
            r2.x(),
            r2.y(),
            r2.z(),
        )
    }

    fn transpose(self) -> M3 {
        M3::new(
            self.m00, self.m10, self.m20, self.m01, self.m11, self.m21, self.m02, self.m12,
            self.m22,
        )
    }

    /// `self * v`.
    fn mul_vec(self, v: Vec3A) -> Vec3A {
        Vec3A::new(self.row0().dot(v), self.row1().dot(v), self.row2().dot(v))
    }

    /// `self^T * v`.
    fn tmul_vec(self, v: Vec3A) -> Vec3A {
        Vec3A::new(
            self.m00 * v.x() + self.m10 * v.y() + self.m20 * v.z(),
            self.m01 * v.x() + self.m11 * v.y() + self.m21 * v.z(),
            self.m02 * v.x() + self.m12 * v.y() + self.m22 * v.z(),
        )
    }

    /// `self * other`.
    fn mul(self, other: M3) -> M3 {
        let t = other.transpose();
        M3::new(
            self.row0().dot(t.row0()),
            self.row0().dot(t.row1()),
            self.row0().dot(t.row2()),
            self.row1().dot(t.row0()),
            self.row1().dot(t.row1()),
            self.row1().dot(t.row2()),
            self.row2().dot(t.row0()),
            self.row2().dot(t.row1()),
            self.row2().dot(t.row2()),
        )
    }

    fn scale(self, s: f32) -> M3 {
        M3::new(
            self.m00 * s,
            self.m01 * s,
            self.m02 * s,
            self.m10 * s,
            self.m11 * s,
            self.m12 * s,
            self.m20 * s,
            self.m21 * s,
            self.m22 * s,
        )
    }

    fn add(self, other: M3) -> M3 {
        M3::new(
            self.m00 + other.m00,
            self.m01 + other.m01,
            self.m02 + other.m02,
            self.m10 + other.m10,
            self.m11 + other.m11,
            self.m12 + other.m12,
            self.m20 + other.m20,
            self.m21 + other.m21,
            self.m22 + other.m22,
        )
    }
}

#[cube]
pub(crate) fn cross3(a: Vec3A, b: Vec3A) -> Vec3A {
    Vec3A::new(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )
}

/// Outer product `a * b^T`.
#[cube]
pub(crate) fn outer3(a: Vec3A, b: Vec3A) -> M3 {
    M3::from_rows(b.scale(a.x()), b.scale(a.y()), b.scale(a.z()))
}

#[cube]
pub(crate) fn softplus(x: f32) -> f32 {
    f32::ln(1.0f32 + f32::exp(x))
}

#[cube]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + f32::exp(-x))
}

// ---------------------------------------------------------------------------
// Homography from latent color params
// ---------------------------------------------------------------------------

/// Targets of the four control chromaticities `(t_b, t_r, t_g, t_gray)`
/// after applying the ZCA-mapped latent offsets.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn color_targets(
    b0: f32,
    b1: f32,
    r0: f32,
    r1: f32,
    g0: f32,
    g1: f32,
    n0: f32,
    n1: f32,
) -> (Vec3A, Vec3A, Vec3A, Vec3A) {
    let bdx = ZCA_B00 * b0 + ZCA_B01 * b1;
    let bdy = ZCA_B10 * b0 + ZCA_B11 * b1;
    let rdx = ZCA_R00 * r0 + ZCA_R01 * r1;
    let rdy = ZCA_R10 * r0 + ZCA_R11 * r1;
    let gdx = ZCA_G00 * g0 + ZCA_G01 * g1;
    let gdy = ZCA_G10 * g0 + ZCA_G11 * g1;
    let ndx = ZCA_N00 * n0 + ZCA_N01 * n1;
    let ndy = ZCA_N10 * n0 + ZCA_N11 * n1;

    let t_b = Vec3A::new(bdx, bdy, 1.0f32);
    let t_r = Vec3A::new(1.0f32 + rdx, rdy, 1.0f32);
    let t_g = Vec3A::new(gdx, 1.0f32 + gdy, 1.0f32);
    let t_gray = Vec3A::new(1.0f32 / 3.0f32 + ndx, 1.0f32 / 3.0f32 + ndy, 1.0f32);
    (t_b, t_r, t_g, t_gray)
}

/// `T` has columns `[t_b, t_r, t_g]`.
#[cube]
pub(crate) fn matrix_t(t_b: Vec3A, t_r: Vec3A, t_g: Vec3A) -> M3 {
    M3::new(
        t_b.x(),
        t_r.x(),
        t_g.x(),
        t_b.y(),
        t_r.y(),
        t_g.y(),
        t_b.z(),
        t_r.z(),
        t_g.z(),
    )
}

/// Skew-symmetric `[t_gray]_x`.
#[cube]
pub(crate) fn skew_of(t: Vec3A) -> M3 {
    M3::new(
        0.0f32,
        -t.z(),
        t.y(),
        t.z(),
        0.0f32,
        -t.x(),
        -t.y(),
        t.x(),
        0.0f32,
    )
}

/// `S^-1` for fixed sources `[B, R, G]` (constant).
#[cube]
pub(crate) fn s_inv() -> M3 {
    M3::new(
        -1.0f32, -1.0f32, 1.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32,
    )
}

/// Nullspace vector of `M` via cross products, with the reference's
/// degeneracy fallbacks.
#[cube]
pub(crate) fn nullspace(m: M3) -> Vec3A {
    let r0 = m.row0();
    let r1 = m.row1();
    let r2 = m.row2();
    let mut lam = cross3(r0, r1);
    if lam.dot(lam) < 1.0e-20f32 {
        lam = cross3(r0, r2);
        if lam.dot(lam) < 1.0e-20f32 {
            lam = cross3(r1, r2);
        }
    }
    lam
}

/// Forward homography (the unnormalised matrix is also needed by the
/// backward, so return both).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn homography(
    b0: f32,
    b1: f32,
    r0: f32,
    r1: f32,
    g0: f32,
    g1: f32,
    n0: f32,
    n1: f32,
) -> (M3, M3) {
    let (t_b, t_r, t_g, t_gray) = color_targets(b0, b1, r0, r1, g0, g1, n0, n1);
    let t = matrix_t(t_b, t_r, t_g);
    let skew = skew_of(t_gray);
    let m = skew.mul(t);
    let lam = nullspace(m);
    let d = M3::new(
        lam.x(),
        0.0f32,
        0.0f32,
        0.0f32,
        lam.y(),
        0.0f32,
        0.0f32,
        0.0f32,
        lam.z(),
    );
    let h_unnorm = t.mul(d).mul(s_inv());
    let s = h_unnorm.m22;
    let mut h = h_unnorm;
    if f32::abs(s) > 1.0e-20f32 {
        h = h_unnorm.scale(1.0f32 / s);
    }
    (h, h_unnorm)
}

// ---------------------------------------------------------------------------
// Forward stage functions
// ---------------------------------------------------------------------------

/// Unclamped vignetting falloff polynomial and the radius terms.
#[cube]
pub(crate) fn vig_falloff_raw(
    uvx: f32,
    uvy: f32,
    cx: f32,
    cy: f32,
    a0: f32,
    a1: f32,
    a2: f32,
) -> f32 {
    let dx = uvx - cx;
    let dy = uvy - cy;
    let r2 = dx * dx + dy * dy;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    1.0f32 + a0 * r2 + a1 * r4 + a2 * r6
}

/// Color correction: RGB → RGI, homography, intensity renorm, back to RGB.
#[cube]
pub(crate) fn color_correct_fwd(rgb: Vec3A, h: M3) -> Vec3A {
    let intensity = rgb.x() + rgb.y() + rgb.z();
    let rgi_in = Vec3A::new(rgb.x(), rgb.y(), intensity);
    let rgi_out = h.mul_vec(rgi_in);
    let norm = intensity / (rgi_out.z() + 1.0e-5f32);
    let o = rgi_out.scale(norm);
    Vec3A::new(o.x(), o.y(), o.z() - o.x() - o.y())
}

/// Single-channel CRF toe-shoulder curve on `x` (already clamped to `[0, 1]`).
#[cube]
pub(crate) fn crf_channel_fwd(
    x: f32,
    toe_raw: f32,
    shoulder_raw: f32,
    gamma_raw: f32,
    center_raw: f32,
) -> f32 {
    let toe = 0.3f32 + softplus(toe_raw);
    let shoulder = 0.3f32 + softplus(shoulder_raw);
    let gamma = 0.1f32 + softplus(gamma_raw);
    let center = sigmoid(center_raw);

    let lerp_val = toe + center * (shoulder - toe);
    let a = shoulder * center / lerp_val;
    let b = 1.0f32 - a;

    let y_low = a * f32::powf(x / center, toe);
    let y_high = 1.0f32 - b * f32::powf((1.0f32 - x) / (1.0f32 - center), shoulder);
    let y = select(x <= center, y_low, y_high);
    f32::powf(f32::max(y, 0.0f32), gamma)
}

/// Normalised vignetting UV for a pixel center.
#[cube]
pub(crate) fn vig_uv(wi: u32, hi: u32, img_w: u32, img_h: u32) -> (f32, f32) {
    let wf = f32::cast_from(img_w);
    let hf = f32::cast_from(img_h);
    let max_res = f32::max(wf, hf);
    let px = f32::cast_from(wi) + 0.5f32;
    let py = f32::cast_from(hi) + 0.5f32;
    ((px - wf * 0.5f32) / max_res, (py - hf * 0.5f32) / max_res)
}

// ---------------------------------------------------------------------------
// Backward helpers (1:1 ports of ppisp_math_bwd.cuh)
// ---------------------------------------------------------------------------

/// Backward of the homography construction. Adds the latent-color gradients
/// into `pg[16..24]`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn homography_bwd(
    b0: f32,
    b1: f32,
    r0p: f32,
    r1p: f32,
    g0: f32,
    g1: f32,
    n0: f32,
    n1: f32,
    grad_h: M3,
    pg: &mut Array<f32>,
    #[comptime] grad_base: u32,
) {
    // Recompute forward intermediates.
    let (t_b, t_r, t_g, t_gray) = color_targets(b0, b1, r0p, r1p, g0, g1, n0, n1);
    let t = matrix_t(t_b, t_r, t_g);
    let skew = skew_of(t_gray);
    let m = skew.mul(t);
    let lam = nullspace(m);
    let d = M3::new(
        lam.x(),
        0.0f32,
        0.0f32,
        0.0f32,
        lam.y(),
        0.0f32,
        0.0f32,
        0.0f32,
        lam.z(),
    );
    let td = t.mul(d);
    let h_unnorm = td.mul(s_inv());

    // Normalisation backward: H = H_unnorm / H_unnorm[2][2].
    let s = h_unnorm.m22;
    let mut grad_hu = grad_h;
    if f32::abs(s) > 1.0e-20f32 {
        let inv_s = 1.0f32 / s;
        let grad_s = -(grad_h.m00 * h_unnorm.m00
            + grad_h.m01 * h_unnorm.m01
            + grad_h.m02 * h_unnorm.m02
            + grad_h.m10 * h_unnorm.m10
            + grad_h.m11 * h_unnorm.m11
            + grad_h.m12 * h_unnorm.m12
            + grad_h.m20 * h_unnorm.m20
            + grad_h.m21 * h_unnorm.m21
            + grad_h.m22 * h_unnorm.m22)
            * inv_s
            * inv_s;
        grad_hu = grad_h.scale(inv_s);
        grad_hu = M3::new(
            grad_hu.m00,
            grad_hu.m01,
            grad_hu.m02,
            grad_hu.m10,
            grad_hu.m11,
            grad_hu.m12,
            grad_hu.m20,
            grad_hu.m21,
            grad_hu.m22 + grad_s,
        );
    }

    // H_unnorm = TD * S_inv → grad_TD = grad_Hu * S_inv^T.
    let grad_td = grad_hu.mul(s_inv().transpose());

    // TD = T * D → grad_T = grad_TD * D^T, grad_D = T^T * grad_TD.
    let mut grad_t = grad_td.mul(d.transpose());
    let grad_d = t.transpose().mul(grad_td);

    // D = diag(lam) → grad_lam = diag(grad_D).
    let grad_lam = Vec3A::new(grad_d.m00, grad_d.m11, grad_d.m22);

    // lam = nullspace(M) via cross products (replay the forward branch).
    let r0 = m.row0();
    let r1 = m.row1();
    let r2 = m.row2();
    let mut grad_r0 = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    let mut grad_r1 = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    let mut grad_r2 = Vec3A::new(0.0f32, 0.0f32, 0.0f32);
    // cross_bwd(g, a, b): grad_a = cross(b, g), grad_b = cross(g, a).
    let lam01 = cross3(r0, r1);
    if lam01.dot(lam01) < 1.0e-20f32 {
        let lam02 = cross3(r0, r2);
        if lam02.dot(lam02) < 1.0e-20f32 {
            grad_r1 = cross3(r2, grad_lam);
            grad_r2 = cross3(grad_lam, r1);
        } else {
            grad_r0 = cross3(r2, grad_lam);
            grad_r2 = cross3(grad_lam, r0);
        }
    } else {
        grad_r0 = cross3(r1, grad_lam);
        grad_r1 = cross3(grad_lam, r0);
    }
    let grad_m = M3::from_rows(grad_r0, grad_r1, grad_r2);

    // M = skew * T → grad_skew = grad_M * T^T, grad_T += skew^T * grad_M.
    let grad_skew = grad_m.mul(t.transpose());
    grad_t = grad_t.add(skew.transpose().mul(grad_m));

    // Skew construction → grad_t_gray.
    let grad_t_gray = Vec3A::new(
        -grad_skew.m12 + grad_skew.m21,
        grad_skew.m02 - grad_skew.m20,
        -grad_skew.m01 + grad_skew.m10,
    );

    // T columns are [t_b, t_r, t_g]; only x/y components flow to the latent
    // offsets (z is the constant 1).
    let grad_bd_x = grad_t.m00;
    let grad_bd_y = grad_t.m10;
    let grad_rd_x = grad_t.m01;
    let grad_rd_y = grad_t.m11;
    let grad_gd_x = grad_t.m02;
    let grad_gd_y = grad_t.m12;
    let grad_nd_x = grad_t_gray.x();
    let grad_nd_y = grad_t_gray.y();

    // ZCA backward: grad_latent = zca^T * grad_offset. `grad_base` selects
    // where the 8 latent gradients land in the caller's accumulator.
    pg[comptime!(grad_base as usize)] += ZCA_B00 * grad_bd_x + ZCA_B10 * grad_bd_y;
    pg[comptime!((grad_base + 1) as usize)] += ZCA_B01 * grad_bd_x + ZCA_B11 * grad_bd_y;
    pg[comptime!((grad_base + 2) as usize)] += ZCA_R00 * grad_rd_x + ZCA_R10 * grad_rd_y;
    pg[comptime!((grad_base + 3) as usize)] += ZCA_R01 * grad_rd_x + ZCA_R11 * grad_rd_y;
    pg[comptime!((grad_base + 4) as usize)] += ZCA_G00 * grad_gd_x + ZCA_G10 * grad_gd_y;
    pg[comptime!((grad_base + 5) as usize)] += ZCA_G01 * grad_gd_x + ZCA_G11 * grad_gd_y;
    pg[comptime!((grad_base + 6) as usize)] += ZCA_N00 * grad_nd_x + ZCA_N10 * grad_nd_y;
    pg[comptime!((grad_base + 7) as usize)] += ZCA_N01 * grad_nd_x + ZCA_N11 * grad_nd_y;
}

/// Backward through the color correction. Returns `dL/drgb_in` and adds the
/// latent color-param gradients into `pg[16..24]`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn color_correct_bwd(
    rgb_in: Vec3A,
    b0: f32,
    b1: f32,
    r0: f32,
    r1: f32,
    g0: f32,
    g1: f32,
    n0: f32,
    n1: f32,
    grad_out: Vec3A,
    pg: &mut Array<f32>,
    #[comptime] grad_base: u32,
) -> Vec3A {
    let (h, _) = homography(b0, b1, r0, r1, g0, g1, n0, n1);

    let intensity = rgb_in.x() + rgb_in.y() + rgb_in.z();
    let rgi_in = Vec3A::new(rgb_in.x(), rgb_in.y(), intensity);
    let rgi_out = h.mul_vec(rgi_in);
    let norm = intensity / (rgi_out.z() + 1.0e-5f32);

    // rgb = [o.x, o.y, o.z - o.x - o.y] where o = rgi_out * norm.
    let grad_o = Vec3A::new(
        grad_out.x() - grad_out.z(),
        grad_out.y() - grad_out.z(),
        grad_out.z(),
    );

    let mut grad_rgi_out = grad_o.scale(norm);
    let grad_norm = grad_o.dot(rgi_out);
    grad_rgi_out = Vec3A::new(
        grad_rgi_out.x(),
        grad_rgi_out.y(),
        grad_rgi_out.z() - grad_norm * norm / (rgi_out.z() + 1.0e-5f32),
    );

    // rgi_out = H * rgi_in.
    let grad_h = outer3(grad_rgi_out, rgi_in);
    let grad_rgi_in = h.tmul_vec(grad_rgi_out);

    let mut grad_intensity = 0.0f32;
    if intensity > 1.0e-8f32 {
        grad_intensity = grad_norm * norm / intensity;
    }

    let grad_in = Vec3A::new(
        grad_rgi_in.x() + grad_rgi_in.z() + grad_intensity,
        grad_rgi_in.y() + grad_rgi_in.z() + grad_intensity,
        grad_rgi_in.z() + grad_intensity,
    );

    homography_bwd(b0, b1, r0, r1, g0, g1, n0, n1, grad_h, pg, grad_base);
    grad_in
}

/// Single-channel CRF backward. Returns `dL/dx` and the four raw-parameter
/// gradients `(toe, shoulder, gamma, center)`.
#[cube]
pub(crate) fn crf_channel_bwd(
    x: f32,
    toe_raw: f32,
    shoulder_raw: f32,
    gamma_raw: f32,
    center_raw: f32,
    grad_out: f32,
) -> (f32, f32, f32, f32, f32) {
    let toe = 0.3f32 + softplus(toe_raw);
    let shoulder = 0.3f32 + softplus(shoulder_raw);
    let gamma = 0.1f32 + softplus(gamma_raw);
    let center = sigmoid(center_raw);

    let lerp_val = toe + center * (shoulder - toe);
    let a = shoulder * center / lerp_val;
    let b = 1.0f32 - a;

    let low = x <= center;
    let y_low = a * f32::powf(x / center, toe);
    let y_high = 1.0f32 - b * f32::powf((1.0f32 - x) / (1.0f32 - center), shoulder);
    let y = select(low, y_low, y_high);
    let y_clamped = f32::max(y, 0.0f32);
    let output = f32::powf(y_clamped, gamma);

    // d(output)/dy through the gamma power.
    let mut grad_y = 0.0f32;
    if y_clamped > 0.0f32 {
        grad_y = grad_out * gamma * f32::powf(y_clamped, gamma - 1.0f32);
    }

    let mut grad_x = 0.0f32;
    let mut grad_toe = 0.0f32;
    let mut grad_shoulder = 0.0f32;
    let mut grad_center = 0.0f32;
    let mut grad_a = 0.0f32;
    let mut grad_b = 0.0f32;

    if low && center > 0.0f32 {
        let base = x / center;
        if base > 0.0f32 {
            let powered = f32::powf(base, toe);
            grad_x = grad_y * a * toe * f32::powf(base, toe - 1.0f32) / center;
            grad_a += grad_y * powered;
            grad_toe += grad_y * a * powered * f32::ln(base + 1.0e-8f32);
            let grad_base = grad_y * a * toe * f32::powf(base, toe - 1.0f32);
            grad_center += grad_base * (-x / (center * center));
        }
    } else if !low && center < 1.0f32 {
        let base = (1.0f32 - x) / (1.0f32 - center);
        if base > 0.0f32 {
            let powered = f32::powf(base, shoulder);
            grad_x = grad_y * b * shoulder * f32::powf(base, shoulder - 1.0f32) / (1.0f32 - center);
            grad_b += -grad_y * powered;
            grad_shoulder += -grad_y * b * powered * f32::ln(base + 1.0e-8f32);
            let grad_base = grad_y * (-b * shoulder * f32::powf(base, shoulder - 1.0f32));
            let dbase_dcenter = (1.0f32 - x) / ((1.0f32 - center) * (1.0f32 - center));
            grad_center += grad_base * dbase_dcenter;
        }
    }

    // b = 1 - a.
    grad_a += -grad_b;

    if f32::abs(lerp_val) > 1.0e-8f32 {
        let a_over_lerp = shoulder * center / lerp_val;
        grad_shoulder += grad_a * center / lerp_val;
        grad_center += grad_a * shoulder / lerp_val;
        let grad_lerp_val = -grad_a * a_over_lerp / lerp_val;
        grad_shoulder += grad_lerp_val * center;
        grad_toe += grad_lerp_val * (1.0f32 - center);
        grad_center += grad_lerp_val * (shoulder - toe);
    }

    // Gradient to gamma from output = y_clamped^gamma.
    let mut grad_gamma = 0.0f32;
    if y_clamped > 0.0f32 {
        grad_gamma = grad_out * output * f32::ln(y_clamped + 1.0e-8f32);
    }

    // Raw-parameter transforms: softplus'(x) = sigmoid(x); sigmoid' = s(1-s).
    let grad_toe_raw = grad_toe * sigmoid(toe_raw);
    let grad_shoulder_raw = grad_shoulder * sigmoid(shoulder_raw);
    let grad_gamma_raw = grad_gamma * sigmoid(gamma_raw);
    let grad_center_raw = grad_center * center * (1.0f32 - center);

    (
        grad_x,
        grad_toe_raw,
        grad_shoulder_raw,
        grad_gamma_raw,
        grad_center_raw,
    )
}
