use crate::shaders;

use glam::Vec3;
const SH_C0: f32 = shaders::SH_C0;

pub const fn sh_coeffs_for_degree(degree: u32) -> u32 {
    (degree + 1).pow(2)
}

pub fn sh_degree_from_coeffs(coeffs_per_channel: u32) -> u32 {
    match coeffs_per_channel {
        1 => 0,
        4 => 1,
        9 => 2,
        16 => 3,
        25 => 4,
        _ => panic!("Invalid nr. of sh bases {coeffs_per_channel}"),
    }
}

pub fn channel_to_sh(rgb: f32) -> f32 {
    (rgb - 0.5) / SH_C0
}

pub fn rgb_to_sh(rgb: Vec3) -> Vec3 {
    glam::vec3(
        channel_to_sh(rgb.x),
        channel_to_sh(rgb.y),
        channel_to_sh(rgb.z),
    )
}

// For linear RGB input, the SH coefficients work the same way
// The centering at 0.5 is appropriate for both sRGB [0,1] and linear RGB [0,1]
// The SH basis is color-space independent; it's just a linear representation

/// Convert linear RGB to sRGB for display
/// Used when rendering linear RGB values for display on standard monitors
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert a linear RGB color to sRGB
pub fn linear_color_to_srgb(rgb: Vec3) -> Vec3 {
    glam::vec3(
        linear_to_srgb(rgb.x),
        linear_to_srgb(rgb.y),
        linear_to_srgb(rgb.z),
    )
}
