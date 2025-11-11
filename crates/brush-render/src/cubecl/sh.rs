use burn::cubecl;
use burn_cubecl::cubecl::prelude::*;

// Spherical harmonics constant
pub const SH_C0: f32 = 0.2820947917738781;

/// Calculate number of SH coefficients for a given degree
#[cube]
pub fn num_sh_coeffs(degree: u32) -> u32 {
    (degree + 1) * (degree + 1)
}

/// Evaluate spherical harmonics at a view direction to get color
/// Based on: Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
///
/// Returns color as individual components to avoid vec3→vec4 cast issues in CubeCL WGSL backend
#[cube]
pub fn sh_coeffs_to_color(
    degree: u32,
    viewdir_x: f32,
    viewdir_y: f32,
    viewdir_z: f32,
    coeffs: &Tensor<f32>,
    base_idx: u32,
    out_r: &mut f32,
    out_g: &mut f32,
    out_b: &mut f32,
) {
    // Degree 0: Just the constant term
    let mut color_r = SH_C0 * coeffs[base_idx];
    let mut color_g = SH_C0 * coeffs[base_idx + 1];
    let mut color_b = SH_C0 * coeffs[base_idx + 2];

    if degree >= 1 {
        // Degree 1: Linear terms
        let x = viewdir_x;
        let y = viewdir_y;
        let z = viewdir_z;

        let ftmp0a = 0.48860251190292;

        let idx = base_idx + 3;
        let sh1_c0_r = coeffs[idx];
        let sh1_c0_g = coeffs[idx + 1];
        let sh1_c0_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh1_c1_r = coeffs[idx];
        let sh1_c1_g = coeffs[idx + 1];
        let sh1_c1_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh1_c2_r = coeffs[idx];
        let sh1_c2_g = coeffs[idx + 1];
        let sh1_c2_b = coeffs[idx + 2];

        color_r = color_r + ftmp0a * (-y * sh1_c0_r + z * sh1_c1_r - x * sh1_c2_r);
        color_g = color_g + ftmp0a * (-y * sh1_c0_g + z * sh1_c1_g - x * sh1_c2_g);
        color_b = color_b + ftmp0a * (-y * sh1_c0_b + z * sh1_c1_b - x * sh1_c2_b);
    }

    if degree >= 2 {
        let x = viewdir_x;
        let y = viewdir_y;
        let z = viewdir_z;

        // Degree 2: Quadratic terms
        let z2 = z * z;
        let ftmp0b = -1.092548430592079 * z;
        let ftmp1a = 0.5462742152960395;
        let fc1 = x * x - y * y;
        let fs1 = 2.0 * x * y;

        let psh6 = 0.9461746957575601 * z2 - 0.3153915652525201;
        let psh7 = ftmp0b * x;
        let psh5 = ftmp0b * y;
        let psh8 = ftmp1a * fc1;
        let psh4 = ftmp1a * fs1;

        let idx = base_idx + 12;
        let sh2_c0_r = coeffs[idx];
        let sh2_c0_g = coeffs[idx + 1];
        let sh2_c0_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh2_c1_r = coeffs[idx];
        let sh2_c1_g = coeffs[idx + 1];
        let sh2_c1_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh2_c2_r = coeffs[idx];
        let sh2_c2_g = coeffs[idx + 1];
        let sh2_c2_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh2_c3_r = coeffs[idx];
        let sh2_c3_g = coeffs[idx + 1];
        let sh2_c3_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh2_c4_r = coeffs[idx];
        let sh2_c4_g = coeffs[idx + 1];
        let sh2_c4_b = coeffs[idx + 2];

        color_r = color_r
            + psh4 * sh2_c0_r
            + psh5 * sh2_c1_r
            + psh6 * sh2_c2_r
            + psh7 * sh2_c3_r
            + psh8 * sh2_c4_r;
        color_g = color_g
            + psh4 * sh2_c0_g
            + psh5 * sh2_c1_g
            + psh6 * sh2_c2_g
            + psh7 * sh2_c3_g
            + psh8 * sh2_c4_g;
        color_b = color_b
            + psh4 * sh2_c0_b
            + psh5 * sh2_c1_b
            + psh6 * sh2_c2_b
            + psh7 * sh2_c3_b
            + psh8 * sh2_c4_b;
    }

    if degree >= 3 {
        let x = viewdir_x;
        let y = viewdir_y;
        let z = viewdir_z;

        let z2 = z * z;
        let fc1 = x * x - y * y;
        let fs1 = 2.0 * x * y;

        // Degree 3: Cubic terms
        let ftmp0c = -2.285228997322329 * z2 + 0.4570457994644658;
        let ftmp1b = 1.445305721320277 * z;
        let ftmp2a = -0.5900435899266435;
        let fc2 = x * fc1 - y * fs1;
        let fs2 = x * fs1 + y * fc1;

        let psh12 = z * (1.865881662950577 * z2 - 1.119528997770346);
        let psh13 = ftmp0c * x;
        let psh11 = ftmp0c * y;
        let psh14 = ftmp1b * fc1;
        let psh10 = ftmp1b * fs1;
        let psh15 = ftmp2a * fc2;
        let psh9 = ftmp2a * fs2;

        let idx = base_idx + 27;
        let sh3_c0_r = coeffs[idx];
        let sh3_c0_g = coeffs[idx + 1];
        let sh3_c0_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh3_c1_r = coeffs[idx];
        let sh3_c1_g = coeffs[idx + 1];
        let sh3_c1_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh3_c2_r = coeffs[idx];
        let sh3_c2_g = coeffs[idx + 1];
        let sh3_c2_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh3_c3_r = coeffs[idx];
        let sh3_c3_g = coeffs[idx + 1];
        let sh3_c3_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh3_c4_r = coeffs[idx];
        let sh3_c4_g = coeffs[idx + 1];
        let sh3_c4_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh3_c5_r = coeffs[idx];
        let sh3_c5_g = coeffs[idx + 1];
        let sh3_c5_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh3_c6_r = coeffs[idx];
        let sh3_c6_g = coeffs[idx + 1];
        let sh3_c6_b = coeffs[idx + 2];

        color_r = color_r
            + psh9 * sh3_c0_r
            + psh10 * sh3_c1_r
            + psh11 * sh3_c2_r
            + psh12 * sh3_c3_r
            + psh13 * sh3_c4_r
            + psh14 * sh3_c5_r
            + psh15 * sh3_c6_r;
        color_g = color_g
            + psh9 * sh3_c0_g
            + psh10 * sh3_c1_g
            + psh11 * sh3_c2_g
            + psh12 * sh3_c3_g
            + psh13 * sh3_c4_g
            + psh14 * sh3_c5_g
            + psh15 * sh3_c6_g;
        color_b = color_b
            + psh9 * sh3_c0_b
            + psh10 * sh3_c1_b
            + psh11 * sh3_c2_b
            + psh12 * sh3_c3_b
            + psh13 * sh3_c4_b
            + psh14 * sh3_c5_b
            + psh15 * sh3_c6_b;
    }

    if degree >= 4 {
        let x = viewdir_x;
        let y = viewdir_y;
        let z = viewdir_z;

        let z2 = z * z;
        let fc1 = x * x - y * y;
        let fs1 = 2.0 * x * y;
        let fc2 = x * fc1 - y * fs1;
        let fs2 = x * fs1 + y * fc1;

        // Degree 4: Quartic terms
        let ftmp0d = z * (-4.683325804901025 * z2 + 2.007139630671868);
        let ftmp1c = 3.31161143515146 * z2 - 0.47308734787878;
        let ftmp2b = -1.770130769779931 * z;
        let ftmp3a = 0.6258357354491763;
        let fc3 = x * fc2 - y * fs2;
        let fs3 = x * fs2 + y * fc2;

        let psh6 = 0.9461746957575601 * z2 - 0.3153915652525201;
        let psh12 = z * (1.865881662950577 * z2 - 1.119528997770346);

        let psh20 = 1.984313483298443 * z * psh12 - 1.006230589874905 * psh6;
        let psh21 = ftmp0d * x;
        let psh19 = ftmp0d * y;
        let psh22 = ftmp1c * fc1;
        let psh18 = ftmp1c * fs1;
        let psh23 = ftmp2b * fc2;
        let psh17 = ftmp2b * fs2;
        let psh24 = ftmp3a * fc3;
        let psh16 = ftmp3a * fs3;

        let idx = base_idx + 48;
        let sh4_c0_r = coeffs[idx];
        let sh4_c0_g = coeffs[idx + 1];
        let sh4_c0_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c1_r = coeffs[idx];
        let sh4_c1_g = coeffs[idx + 1];
        let sh4_c1_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c2_r = coeffs[idx];
        let sh4_c2_g = coeffs[idx + 1];
        let sh4_c2_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c3_r = coeffs[idx];
        let sh4_c3_g = coeffs[idx + 1];
        let sh4_c3_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c4_r = coeffs[idx];
        let sh4_c4_g = coeffs[idx + 1];
        let sh4_c4_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c5_r = coeffs[idx];
        let sh4_c5_g = coeffs[idx + 1];
        let sh4_c5_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c6_r = coeffs[idx];
        let sh4_c6_g = coeffs[idx + 1];
        let sh4_c6_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c7_r = coeffs[idx];
        let sh4_c7_g = coeffs[idx + 1];
        let sh4_c7_b = coeffs[idx + 2];

        let idx = idx + 3;
        let sh4_c8_r = coeffs[idx];
        let sh4_c8_g = coeffs[idx + 1];
        let sh4_c8_b = coeffs[idx + 2];

        color_r = color_r
            + psh16 * sh4_c0_r
            + psh17 * sh4_c1_r
            + psh18 * sh4_c2_r
            + psh19 * sh4_c3_r
            + psh20 * sh4_c4_r
            + psh21 * sh4_c5_r
            + psh22 * sh4_c6_r
            + psh23 * sh4_c7_r
            + psh24 * sh4_c8_r;
        color_g = color_g
            + psh16 * sh4_c0_g
            + psh17 * sh4_c1_g
            + psh18 * sh4_c2_g
            + psh19 * sh4_c3_g
            + psh20 * sh4_c4_g
            + psh21 * sh4_c5_g
            + psh22 * sh4_c6_g
            + psh23 * sh4_c7_g
            + psh24 * sh4_c8_g;
        color_b = color_b
            + psh16 * sh4_c0_b
            + psh17 * sh4_c1_b
            + psh18 * sh4_c2_b
            + psh19 * sh4_c3_b
            + psh20 * sh4_c4_b
            + psh21 * sh4_c5_b
            + psh22 * sh4_c6_b
            + psh23 * sh4_c7_b
            + psh24 * sh4_c8_b;
    }

    *out_r = color_r;
    *out_g = color_g;
    *out_b = color_b;
}
