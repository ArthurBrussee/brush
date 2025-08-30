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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vector_decode() {
        let result = decode_vec_11_10_11(0);
        assert_eq!(result, glam::Vec3::ZERO);

        let max_val = (0x7FF << 21) | (0x3FF << 11) | 0x7FF;
        let result = decode_vec_11_10_11(max_val);
        assert!((result.x - 1.0).abs() < 1e-6);
        assert!((result.y - 1.0).abs() < 1e-6);
        assert!((result.z - 1.0).abs() < 1e-6);
        let result = decode_vec_8_8_8_8(0);
        assert_eq!(result, glam::Vec4::ZERO);
        let result = decode_vec_8_8_8_8(0xFFFFFFFF);
        assert!((result - glam::Vec4::ONE).length() < 1e-6);

        for i in 0..100 {
            let test_val = i * 42949673;
            let vec3 = decode_vec_11_10_11(test_val);
            let vec4 = decode_vec_8_8_8_8(test_val);
            assert!(vec3.min_element() >= 0.0 && vec3.max_element() <= 1.0);
            assert!(vec4.min_element() >= 0.0 && vec4.max_element() <= 1.0);
        }
    }

    #[test]
    fn test_decode_quat() {
        let test_val = (512 << 20) | (512 << 10) | 512;
        let result = decode_quat(test_val);
        assert!(
            (result.length() - 1.0).abs() < 1e-5,
            "Quaternion should be normalized"
        );
        assert!(result.x.is_finite());
        assert!(result.y.is_finite());
        assert!(result.z.is_finite());
        assert!(result.w.is_finite());
    }
}
