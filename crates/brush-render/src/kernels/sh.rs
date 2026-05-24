//! Spherical harmonics (forward + VJP).
//!
//! Coefficients are stored 3-per-coefficient in a flat `Tensor<f32>`,
//! packed without padding. Each splat occupies `num_sh_coeffs(degree) *
//! 3` consecutive f32s starting at `global_gid * num_sh_coeffs(degree)
//! * 3`.
//!
//! Bases & weight constants follow Sloan, "Efficient Spherical Harmonic
//! Evaluation" (JCGT 2013) — see <https://jcgt.org/published/0002/02/06/>.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::types::Vec3A;

pub const SH_C0: f32 = 0.282_094_8;

#[cube]
pub fn num_sh_coeffs(#[comptime] degree: u32) -> comptime_type!(u32) {
    comptime![(degree + 1u32) * (degree + 1u32)]
}

/// Read one coefficient (3 f32s) at the given f32-offset.
#[cube]
fn read_coeff(coeffs: &Tensor<f32>, base: u32) -> Vec3A {
    let b = base as usize;
    Vec3A::new(coeffs[b], coeffs[b + 1], coeffs[b + 2])
}

/// Write one coefficient gradient (3 f32s) at the given f32-offset.
#[cube]
fn write_coeff(v: &mut Tensor<f32>, base: u32, val: Vec3A) {
    let b = base as usize;
    v[b] = val.x();
    v[b + 1] = val.y();
    v[b + 2] = val.z();
}

/// Evaluate SH coefficients to color given a unit `viewdir`. Returns
/// the resulting `vec3` color (without the `+0.5` SH-to-color offset —
/// the caller adds that).
///
/// `degree` is `#[comptime]` so the band branches DCE away — each
/// kernel variant carries only the work for its actual SH degree.
#[cube]
pub fn sh_coeffs_to_color(
    coeffs: &Tensor<f32>,
    coeff_base: u32,
    #[comptime] degree: u32,
    v: Vec3A,
) -> Vec3A {
    let mut color = read_coeff(coeffs, coeff_base).scale(SH_C0);

    if comptime![degree >= 1u32] {
        let b1_0 = read_coeff(coeffs, coeff_base + 3u32);
        let b1_1 = read_coeff(coeffs, coeff_base + 6u32);
        let b1_2 = read_coeff(coeffs, coeff_base + 9u32);
        let f0a = 0.488_602_5f32;
        color = color.add(b1_0.scale(-f0a * v.y()));
        color = color.add(b1_1.scale(f0a * v.z()));
        color = color.add(b1_2.scale(-f0a * v.x()));

        if comptime![degree >= 2u32] {
            let z2 = v.z() * v.z();
            let f0b = -1.092_548_5f32 * v.z();
            let f1a = 0.546_274_24f32;
            let fc1 = v.x() * v.x() - v.y() * v.y();
            let fs1 = 2.0f32 * v.x() * v.y();
            let p_sh4 = f1a * fs1;
            let p_sh5 = f0b * v.y();
            let p_sh6 = 0.946_174_7f32 * z2 - 0.315_391_57f32;
            let p_sh7 = f0b * v.x();
            let p_sh8 = f1a * fc1;

            color = color.add(read_coeff(coeffs, coeff_base + 12u32).scale(p_sh4));
            color = color.add(read_coeff(coeffs, coeff_base + 15u32).scale(p_sh5));
            color = color.add(read_coeff(coeffs, coeff_base + 18u32).scale(p_sh6));
            color = color.add(read_coeff(coeffs, coeff_base + 21u32).scale(p_sh7));
            color = color.add(read_coeff(coeffs, coeff_base + 24u32).scale(p_sh8));

            if comptime![degree >= 3u32] {
                let f0c = -2.285_229f32 * z2 + 0.457_045_8f32;
                let f1b = 1.445_305_7f32 * v.z();
                let f2a = -0.590_043_6f32;
                let fc2 = v.x() * fc1 - v.y() * fs1;
                let fs2 = v.x() * fs1 + v.y() * fc1;
                let p_sh12 = v.z() * (1.865_881_7f32 * z2 - 1.119_529f32);
                let p_sh9 = f2a * fs2;
                let p_sh10 = f1b * fs1;
                let p_sh11 = f0c * v.y();
                let p_sh13 = f0c * v.x();
                let p_sh14 = f1b * fc1;
                let p_sh15 = f2a * fc2;

                color = color.add(read_coeff(coeffs, coeff_base + 27u32).scale(p_sh9));
                color = color.add(read_coeff(coeffs, coeff_base + 30u32).scale(p_sh10));
                color = color.add(read_coeff(coeffs, coeff_base + 33u32).scale(p_sh11));
                color = color.add(read_coeff(coeffs, coeff_base + 36u32).scale(p_sh12));
                color = color.add(read_coeff(coeffs, coeff_base + 39u32).scale(p_sh13));
                color = color.add(read_coeff(coeffs, coeff_base + 42u32).scale(p_sh14));
                color = color.add(read_coeff(coeffs, coeff_base + 45u32).scale(p_sh15));

                if comptime![degree >= 4u32] {
                    let f0d = v.z() * (-4.683_326f32 * z2 + 2.007_139_6f32);
                    let f1c = 3.311_611_4f32 * z2 - 0.473_087_35f32;
                    let f2b = -1.770_130_8f32 * v.z();
                    let f3a = 0.625_835_75f32;
                    let fc3 = v.x() * fc2 - v.y() * fs2;
                    let fs3 = v.x() * fs2 + v.y() * fc2;
                    let p_sh20 = 1.984_313_5f32 * v.z() * p_sh12 - 1.006_230_6f32 * p_sh6;
                    let p_sh16 = f3a * fs3;
                    let p_sh17 = f2b * fs2;
                    let p_sh18 = f1c * fs1;
                    let p_sh19 = f0d * v.y();
                    let p_sh21 = f0d * v.x();
                    let p_sh22 = f1c * fc1;
                    let p_sh23 = f2b * fc2;
                    let p_sh24 = f3a * fc3;

                    color = color.add(read_coeff(coeffs, coeff_base + 48u32).scale(p_sh16));
                    color = color.add(read_coeff(coeffs, coeff_base + 51u32).scale(p_sh17));
                    color = color.add(read_coeff(coeffs, coeff_base + 54u32).scale(p_sh18));
                    color = color.add(read_coeff(coeffs, coeff_base + 57u32).scale(p_sh19));
                    color = color.add(read_coeff(coeffs, coeff_base + 60u32).scale(p_sh20));
                    color = color.add(read_coeff(coeffs, coeff_base + 63u32).scale(p_sh21));
                    color = color.add(read_coeff(coeffs, coeff_base + 66u32).scale(p_sh22));
                    color = color.add(read_coeff(coeffs, coeff_base + 69u32).scale(p_sh23));
                    color = color.add(read_coeff(coeffs, coeff_base + 72u32).scale(p_sh24));
                }
            }
        }
    }

    color
}

/// VJP w.r.t. the view direction `v`: given upstream `vc = ∂L/∂color`,
/// returns `∂L/∂v` from the SH basis polynomials' `v`-dependence. The DC
/// term contributes nothing. Symbolic derivatives mirror
/// `sh_coeffs_to_color`'s structure.
#[cube]
pub fn sh_color_viewdir_vjp(
    coeffs: &Tensor<f32>,
    coeff_base: u32,
    #[comptime] degree: u32,
    v: Vec3A,
    vc: Vec3A,
) -> Vec3A {
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    let mut gz = 0.0f32;

    if comptime![degree >= 1u32] {
        let f0a = 0.488_602_5f32;
        let s_n1 = read_coeff(coeffs, coeff_base + 3u32).dot(vc);
        let s_z0 = read_coeff(coeffs, coeff_base + 6u32).dot(vc);
        let s_p1 = read_coeff(coeffs, coeff_base + 9u32).dot(vc);
        gx += -f0a * s_p1;
        gy += -f0a * s_n1;
        gz += f0a * s_z0;

        if comptime![degree >= 2u32] {
            let z = v.z();
            let x = v.x();
            let y = v.y();
            let c2 = -1.092_548_5f32;
            let f1a = 0.546_274_24f32;
            let s_n2 = read_coeff(coeffs, coeff_base + 12u32).dot(vc);
            let s_n1 = read_coeff(coeffs, coeff_base + 15u32).dot(vc);
            let s_z0 = read_coeff(coeffs, coeff_base + 18u32).dot(vc);
            let s_p1 = read_coeff(coeffs, coeff_base + 21u32).dot(vc);
            let s_p2 = read_coeff(coeffs, coeff_base + 24u32).dot(vc);
            gx += 2.0f32 * f1a * y * s_n2 + c2 * z * s_p1 + 2.0f32 * f1a * x * s_p2;
            gy += 2.0f32 * f1a * x * s_n2 + c2 * z * s_n1 - 2.0f32 * f1a * y * s_p2;
            gz += c2 * y * s_n1 + 2.0f32 * 0.946_174_7f32 * z * s_z0 + c2 * x * s_p1;

            if comptime![degree >= 3u32] {
                let z2 = z * z;
                let x2 = x * x;
                let y2 = y * y;
                let f2a = -0.590_043_6f32;
                let c1b = 1.445_305_7f32;
                let f1b = c1b * z;
                let c0c = -2.285_229f32;
                let f0c = c0c * z2 + 0.457_045_8f32;
                let f0c_dz = 2.0f32 * c0c * z;
                let s_n3 = read_coeff(coeffs, coeff_base + 27u32).dot(vc);
                let s_n2 = read_coeff(coeffs, coeff_base + 30u32).dot(vc);
                let s_n1 = read_coeff(coeffs, coeff_base + 33u32).dot(vc);
                let s_z0 = read_coeff(coeffs, coeff_base + 36u32).dot(vc);
                let s_p1 = read_coeff(coeffs, coeff_base + 39u32).dot(vc);
                let s_p2 = read_coeff(coeffs, coeff_base + 42u32).dot(vc);
                let s_p3 = read_coeff(coeffs, coeff_base + 45u32).dot(vc);
                let d12_z = 3.0f32 * 1.865_881_7f32 * z2 - 1.119_529f32;
                gx += f2a * 6.0f32 * x * y * s_n3
                    + 2.0f32 * f1b * y * s_n2
                    + f0c * s_p1
                    + 2.0f32 * f1b * x * s_p2
                    + f2a * 3.0f32 * (x2 - y2) * s_p3;
                gy += f2a * 3.0f32 * (x2 - y2) * s_n3
                    + 2.0f32 * f1b * x * s_n2
                    + f0c * s_n1
                    + (-2.0f32) * f1b * y * s_p2
                    + f2a * (-6.0f32) * x * y * s_p3;
                gz += 2.0f32 * c1b * x * y * s_n2
                    + f0c_dz * y * s_n1
                    + d12_z * s_z0
                    + f0c_dz * x * s_p1
                    + c1b * (x2 - y2) * s_p2;

                if comptime![degree >= 4u32] {
                    // fc_k / fs_k are Re/Im of (x+iy)^k. Partials follow from
                    // d fc_k/dx = k*fc_{k-1}, d fc_k/dy = -k*fs_{k-1},
                    // d fs_k/dx = k*fs_{k-1}, d fs_k/dy =  k*fc_{k-1}.
                    let fc1 = x2 - y2;
                    let fs1 = 2.0f32 * x * y;
                    let fc2 = x * fc1 - y * fs1;
                    let fs2 = x * fs1 + y * fc1;
                    let f0d = z * (-4.683_326f32 * z2 + 2.007_139_6f32);
                    let f0d_dz = -14.049_978f32 * z2 + 2.007_139_6f32;
                    let f1c = 3.311_611_4f32 * z2 - 0.473_087_35f32;
                    let f1c_dz = 2.0f32 * 3.311_611_4f32 * z;
                    let f2b_dz_const = -1.770_130_8f32;
                    let f2b = f2b_dz_const * z;
                    let f3a = 0.625_835_75f32;
                    // p_sh20 (m=0) = 1.984... z * p_sh12 - 1.006... p_sh6,
                    // both pure functions of z; pull the z-derivative.
                    let p_sh12 = z * (1.865_881_7f32 * z2 - 1.119_529f32);
                    let dp_sh12_dz = 3.0f32 * 1.865_881_7f32 * z2 - 1.119_529f32;
                    let dp_sh6_dz = 2.0f32 * 0.946_174_7f32 * z;
                    let dp_sh20_dz =
                        1.984_313_5f32 * (p_sh12 + z * dp_sh12_dz) - 1.006_230_6f32 * dp_sh6_dz;
                    let s_n4 = read_coeff(coeffs, coeff_base + 48u32).dot(vc);
                    let s_n3 = read_coeff(coeffs, coeff_base + 51u32).dot(vc);
                    let s_n2 = read_coeff(coeffs, coeff_base + 54u32).dot(vc);
                    let s_n1 = read_coeff(coeffs, coeff_base + 57u32).dot(vc);
                    let s_z0 = read_coeff(coeffs, coeff_base + 60u32).dot(vc);
                    let s_p1 = read_coeff(coeffs, coeff_base + 63u32).dot(vc);
                    let s_p2 = read_coeff(coeffs, coeff_base + 66u32).dot(vc);
                    let s_p3 = read_coeff(coeffs, coeff_base + 69u32).dot(vc);
                    let s_p4 = read_coeff(coeffs, coeff_base + 72u32).dot(vc);
                    gx += f3a * 4.0f32 * fs2 * s_n4
                        + f2b * 3.0f32 * fs1 * s_n3
                        + f1c * 2.0f32 * y * s_n2
                        + f0d * s_p1
                        + f1c * 2.0f32 * x * s_p2
                        + f2b * 3.0f32 * fc1 * s_p3
                        + f3a * 4.0f32 * fc2 * s_p4;
                    gy += f3a * 4.0f32 * fc2 * s_n4
                        + f2b * 3.0f32 * fc1 * s_n3
                        + f1c * 2.0f32 * x * s_n2
                        + f0d * s_n1
                        + f1c * (-2.0f32) * y * s_p2
                        + f2b * (-3.0f32) * fs1 * s_p3
                        + f3a * (-4.0f32) * fs2 * s_p4;
                    gz += f2b_dz_const * fs2 * s_n3
                        + f1c_dz * fs1 * s_n2
                        + f0d_dz * y * s_n1
                        + dp_sh20_dz * s_z0
                        + f0d_dz * x * s_p1
                        + f1c_dz * fc1 * s_p2
                        + f2b_dz_const * fc2 * s_p3;
                }
            }
        }
    }

    Vec3A::new(gx, gy, gz)
}

/// VJP of `sh_coeffs_to_color`. Writes the gradient w.r.t. each
/// coefficient to `v_coeffs` at offset `coeff_base`. Higher-degree slots
/// are left untouched (the host pre-zeroes the buffer). `degree` is
/// `#[comptime]`; see `sh_coeffs_to_color` for the rationale.
#[cube]
pub fn sh_coeffs_to_color_vjp(
    v_coeffs: &mut Tensor<f32>,
    coeff_base: u32,
    #[comptime] degree: u32,
    v: Vec3A,
    vc: Vec3A,
) {
    write_coeff(v_coeffs, coeff_base, vc.scale(SH_C0));
    if comptime![degree >= 1u32] {
        let f0a = 0.488_602_5f32;
        write_coeff(v_coeffs, coeff_base + 3u32, vc.scale(-f0a * v.y()));
        write_coeff(v_coeffs, coeff_base + 6u32, vc.scale(f0a * v.z()));
        write_coeff(v_coeffs, coeff_base + 9u32, vc.scale(-f0a * v.x()));
        if comptime![degree >= 2u32] {
            let z2 = v.z() * v.z();
            let f0b = -1.092_548_5f32 * v.z();
            let f1a = 0.546_274_24f32;
            let fc1 = v.x() * v.x() - v.y() * v.y();
            let fs1 = 2.0f32 * v.x() * v.y();
            let p_sh4 = f1a * fs1;
            let p_sh5 = f0b * v.y();
            let p_sh6 = 0.946_174_7f32 * z2 - 0.315_391_57f32;
            let p_sh7 = f0b * v.x();
            let p_sh8 = f1a * fc1;
            write_coeff(v_coeffs, coeff_base + 12u32, vc.scale(p_sh4));
            write_coeff(v_coeffs, coeff_base + 15u32, vc.scale(p_sh5));
            write_coeff(v_coeffs, coeff_base + 18u32, vc.scale(p_sh6));
            write_coeff(v_coeffs, coeff_base + 21u32, vc.scale(p_sh7));
            write_coeff(v_coeffs, coeff_base + 24u32, vc.scale(p_sh8));
            if comptime![degree >= 3u32] {
                let f0c = -2.285_229f32 * z2 + 0.457_045_8f32;
                let f1b = 1.445_305_7f32 * v.z();
                let f2a = -0.590_043_6f32;
                let fc2 = v.x() * fc1 - v.y() * fs1;
                let fs2 = v.x() * fs1 + v.y() * fc1;
                let p_sh12 = v.z() * (1.865_881_7f32 * z2 - 1.119_529f32);
                let p_sh9 = f2a * fs2;
                let p_sh10 = f1b * fs1;
                let p_sh11 = f0c * v.y();
                let p_sh13 = f0c * v.x();
                let p_sh14 = f1b * fc1;
                let p_sh15 = f2a * fc2;
                write_coeff(v_coeffs, coeff_base + 27u32, vc.scale(p_sh9));
                write_coeff(v_coeffs, coeff_base + 30u32, vc.scale(p_sh10));
                write_coeff(v_coeffs, coeff_base + 33u32, vc.scale(p_sh11));
                write_coeff(v_coeffs, coeff_base + 36u32, vc.scale(p_sh12));
                write_coeff(v_coeffs, coeff_base + 39u32, vc.scale(p_sh13));
                write_coeff(v_coeffs, coeff_base + 42u32, vc.scale(p_sh14));
                write_coeff(v_coeffs, coeff_base + 45u32, vc.scale(p_sh15));
                if comptime![degree >= 4u32] {
                    let f0d = v.z() * (-4.683_326f32 * z2 + 2.007_139_6f32);
                    let f1c = 3.311_611_4f32 * z2 - 0.473_087_35f32;
                    let f2b = -1.770_130_8f32 * v.z();
                    let f3a = 0.625_835_75f32;
                    let fc3 = v.x() * fc2 - v.y() * fs2;
                    let fs3 = v.x() * fs2 + v.y() * fc2;
                    let p_sh20 = 1.984_313_5f32 * v.z() * p_sh12 + -1.006_230_6f32 * p_sh6;
                    let p_sh16 = f3a * fs3;
                    let p_sh17 = f2b * fs2;
                    let p_sh18 = f1c * fs1;
                    let p_sh19 = f0d * v.y();
                    let p_sh21 = f0d * v.x();
                    let p_sh22 = f1c * fc1;
                    let p_sh23 = f2b * fc2;
                    let p_sh24 = f3a * fc3;
                    write_coeff(v_coeffs, coeff_base + 48u32, vc.scale(p_sh16));
                    write_coeff(v_coeffs, coeff_base + 51u32, vc.scale(p_sh17));
                    write_coeff(v_coeffs, coeff_base + 54u32, vc.scale(p_sh18));
                    write_coeff(v_coeffs, coeff_base + 57u32, vc.scale(p_sh19));
                    write_coeff(v_coeffs, coeff_base + 60u32, vc.scale(p_sh20));
                    write_coeff(v_coeffs, coeff_base + 63u32, vc.scale(p_sh21));
                    write_coeff(v_coeffs, coeff_base + 66u32, vc.scale(p_sh22));
                    write_coeff(v_coeffs, coeff_base + 69u32, vc.scale(p_sh23));
                    write_coeff(v_coeffs, coeff_base + 72u32, vc.scale(p_sh24));
                }
            }
        }
    }
}
