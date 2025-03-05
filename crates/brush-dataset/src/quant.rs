use std::f32;

/// Unpacks a value from an n-bit normalized integer representation back to a float in [0, 1]
fn unpack_unorm(packed: u32, bits: u32) -> f32 {
    let max_value = (1 << bits) - 1;
    packed as f32 / max_value as f32
}

pub(crate) fn decode_vec_11_10_11(value: u32) -> glam::Vec3 {
    let first = (value >> 21) & 0x7FF; // First 11 bits
    let second = (value >> 11) & 0x3FF; // Next 10 bits
    let third = value & 0x7FF; // Last 11 bits
    glam::vec3(
        unpack_unorm(first, 11),
        unpack_unorm(second, 10),
        unpack_unorm(third, 11),
    )
}

pub(crate) fn decode_vec_8_8_8_8(value: u32) -> glam::Vec4 {
    // Create Vec4 from a u32, each component gets 8 bits
    // Extract each byte
    let x = (value >> 24) & 0xFF;
    let y = (value >> 16) & 0xFF;
    let z = (value >> 8) & 0xFF;
    let w = value & 0xFF;

    // Normalize to 0.0-1.0 range
    glam::vec4(
        unpack_unorm(x, 8),
        unpack_unorm(y, 8),
        unpack_unorm(z, 8),
        unpack_unorm(w, 8),
    )
}

pub(crate) fn decode_quat(value: u32) -> glam::Quat {
    let largest = ((value >> 30) & 0x3) as usize; // First 2 bits

    let a = (value >> 20) & 0x3FF; // Next 10 bits
    let b = (value >> 10) & 0x3FF; // Next 10 bits
    let c = value & 0x3FF; // Last 10 bits

    let norm = 0.5 * f32::consts::SQRT_2;

    let a = (unpack_unorm(a, 10) - 0.5) / norm;
    let b = (unpack_unorm(b, 10) - 0.5) / norm;
    let c = (unpack_unorm(c, 10) - 0.5) / norm;

    let vals = [a, b, c];

    let mut quat = [0.0; 4];
    quat[largest] = (1.0 - glam::vec3(a, b, c).length_squared()).sqrt();

    let mut ind = 0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        if i != largest {
            quat[i] = vals[ind];
            ind += 1;
        }
    }
    let w = quat[0];
    let x = quat[1];
    let y = quat[2];
    let z = quat[3];
    glam::Quat::from_xyzw(x, y, z, w)
}
