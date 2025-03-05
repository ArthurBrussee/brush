use brush_render::render::channel_to_sh;
use glam::{Quat, Vec3};
use ply_rs::ply::{Property, PropertyAccess};

use crate::quant::{decode_quat, decode_vec_8_8_8_8, decode_vec_11_10_11};

/// A gaussian as it parsed in the ply.
///
/// Nb that this is somewhat abused and the values are mostly what _directly_ comes out of the ply,
/// which might need activation/de-activation/dequantization before the values are usable.
#[derive(Default)]
pub(crate) struct ParsedGaussian {
    pub(crate) mean: Vec3,
    pub(crate) log_scale: Vec3,
    pub(crate) opacity: f32,
    pub(crate) rotation: Quat,
    pub(crate) sh_dc: Vec3,
    // NB: This is in the inria format, aka [channels, coeffs]
    // not [coeffs, channels].
    pub(crate) sh_coeffs_rest: Vec<f32>,
}

impl PropertyAccess for ParsedGaussian {
    fn new() -> Self {
        Self::default()
    }

    fn set_property(&mut self, key: &str, property: Property) {
        let ascii = key.as_bytes();

        match ascii {
            // Custom parsed values.
            b"packed_position" => {
                // Atm not doing anything if this is not a uint (which is invalid).
                if let Property::UInt(value) = property {
                    // Parse to (normalized) means.
                    self.mean = decode_vec_11_10_11(value);
                }
            }
            b"packed_rotation" => {
                // Atm not doing anything if this is not a uint (which is invalid).
                if let Property::UInt(value) = property {
                    // Parse to (normalized) quaternion.
                    self.rotation = decode_quat(value);
                }
            }

            b"packed_scale" => {
                // Atm not doing anything if this is not a uint (which is invalid).
                if let Property::UInt(value) = property {
                    // Parse to (normalized) scale.
                    self.log_scale = decode_vec_11_10_11(value);
                }
            }

            b"packed_color" => {
                // Atm not doing anything if this is not a uint (which is invalid).
                if let Property::UInt(value) = property {
                    // Parse to (normalized) color.
                    let vec = decode_vec_8_8_8_8(value);
                    self.sh_dc = glam::vec3(vec.x, vec.y, vec.z);
                    self.opacity = vec.w;
                }
            }

            _ => {
                let value = match property {
                    Property::Float(value) => value,
                    Property::UChar(value) => (value as f32) / (u8::MAX as f32 - 1.0),
                    Property::UShort(value) => (value as f32) / (u16::MAX as f32 - 1.0),
                    _ => return,
                };

                if value.is_nan() || value.is_infinite() {
                    log::warn!("Invalid numbers in imported splat, skipping parse.");
                    return;
                }

                // Floating point values.
                match ascii {
                    b"x" => self.mean[0] = value,
                    b"y" => self.mean[1] = value,
                    b"z" => self.mean[2] = value,
                    b"scale_0" => self.log_scale[0] = value,
                    b"scale_1" => self.log_scale[1] = value,
                    b"scale_2" => self.log_scale[2] = value,
                    b"opacity" => self.opacity = value,

                    // Rotations are saved in scalar from.
                    b"rot_0" => self.rotation.w = value,
                    b"rot_1" => self.rotation.x = value,
                    b"rot_2" => self.rotation.y = value,
                    b"rot_3" => self.rotation.z = value,

                    b"f_dc_0" => self.sh_dc[0] = value,
                    b"f_dc_1" => self.sh_dc[1] = value,
                    b"f_dc_2" => self.sh_dc[2] = value,
                    b"red" => self.sh_dc[0] = channel_to_sh(value),
                    b"green" => self.sh_dc[1] = channel_to_sh(value),
                    b"blue" => self.sh_dc[2] = channel_to_sh(value),
                    _ if ascii.starts_with(b"f_rest_") => {
                        if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                            if idx >= self.sh_coeffs_rest.len() as u32 {
                                self.sh_coeffs_rest.resize(idx as usize + 1, 0.0);
                            }
                            self.sh_coeffs_rest[idx as usize] = value;
                        }
                    }
                    _ => (),
                }
            }
        }
    }

    fn get_float(&self, key: &str) -> Option<f32> {
        let ascii = key.as_bytes();

        match ascii {
            b"x" => Some(self.mean[0]),
            b"y" => Some(self.mean[1]),
            b"z" => Some(self.mean[2]),
            b"scale_0" => Some(self.log_scale[0]),
            b"scale_1" => Some(self.log_scale[1]),
            b"scale_2" => Some(self.log_scale[2]),
            b"opacity" => Some(self.opacity),
            b"rot_0" => Some(self.rotation.w),
            b"rot_1" => Some(self.rotation.x),
            b"rot_2" => Some(self.rotation.y),
            b"rot_3" => Some(self.rotation.z),
            b"f_dc_0" => Some(self.sh_dc[0]),
            b"f_dc_1" => Some(self.sh_dc[1]),
            b"f_dc_2" => Some(self.sh_dc[2]),
            _ if key.starts_with("f_rest_") => {
                if let Ok(idx) = key["f_rest_".len()..].parse::<usize>() {
                    self.sh_coeffs_rest.get(idx).copied()
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
