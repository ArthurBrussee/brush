use crate::quant::{decode_quat, decode_vec_8_8_8_8, decode_vec_11_10_11};

use glam::{Quat, Vec3, Vec4};
use serde::{self, Deserializer};
use serde::{Deserialize, Serialize};

fn de_vec_11_10_11<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec3, D::Error> {
    let value = u32::deserialize(deserializer)?;
    Ok(decode_vec_11_10_11(value))
}

fn de_packed_quat<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Quat, D::Error> {
    let value = u32::deserialize(deserializer)?;
    Ok(decode_quat(value))
}

fn de_vec_8_8_8_8<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec4, D::Error> {
    let value = u32::deserialize(deserializer)?;
    let vec = decode_vec_8_8_8_8(value);
    Ok(vec)
}

#[derive(Deserialize, Debug)]
pub(crate) struct QuantSplat {
    #[serde(rename = "packed_position", deserialize_with = "de_vec_11_10_11")]
    pub(crate) mean: Vec3,
    #[serde(rename = "packed_scale", deserialize_with = "de_vec_11_10_11")]
    pub(crate) log_scale: Vec3,
    #[serde(rename = "packed_rotation", deserialize_with = "de_packed_quat")]
    pub(crate) rotation: Quat,
    #[serde(rename = "packed_color", deserialize_with = "de_vec_8_8_8_8")]
    pub(crate) rgba: Vec4,
}

fn de_quant<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Dequant;
    impl<'de> serde::de::Visitor<'de> for Dequant {
        type Value = Option<f32>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a quantized value or a float")
        }
        fn visit_f32<E>(self, value: f32) -> Result<Option<f32>, E> {
            Ok(Some(value))
        }
        fn visit_u8<E>(self, value: u8) -> Result<Option<f32>, E> {
            Ok(Some(value as f32 / (u8::MAX - 1) as f32))
        }
        fn visit_u16<E>(self, value: u16) -> Result<Option<f32>, E> {
            Ok(Some(value as f32 / (u16::MAX - 1) as f32))
        }
    }
    deserializer.deserialize_any(Dequant)
}

#[derive(Serialize, Deserialize)]
pub(crate) struct PlyGaussian {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,

    #[serde(default)]
    pub(crate) scale_0: f32,
    #[serde(default)]
    pub(crate) scale_1: f32,
    #[serde(default)]
    pub(crate) scale_2: f32,
    #[serde(default)]
    pub(crate) opacity: f32,
    #[serde(default)]
    pub(crate) rot_0: f32,
    #[serde(default)]
    pub(crate) rot_1: f32,
    #[serde(default)]
    pub(crate) rot_2: f32,
    #[serde(default)]
    pub(crate) rot_3: f32,

    #[serde(default)]
    pub(crate) f_dc_0: f32,
    #[serde(default)]
    pub(crate) f_dc_1: f32,
    #[serde(default)]
    pub(crate) f_dc_2: f32,
    #[serde(default)]
    pub(crate) f_rest_0: f32,
    #[serde(default)]
    pub(crate) f_rest_1: f32,
    #[serde(default)]
    pub(crate) f_rest_2: f32,
    #[serde(default)]
    pub(crate) f_rest_3: f32,
    #[serde(default)]
    pub(crate) f_rest_4: f32,
    #[serde(default)]
    pub(crate) f_rest_5: f32,
    #[serde(default)]
    pub(crate) f_rest_6: f32,
    #[serde(default)]
    pub(crate) f_rest_7: f32,
    #[serde(default)]
    pub(crate) f_rest_8: f32,
    #[serde(default)]
    pub(crate) f_rest_9: f32,
    #[serde(default)]
    pub(crate) f_rest_10: f32,
    #[serde(default)]
    pub(crate) f_rest_11: f32,
    #[serde(default)]
    pub(crate) f_rest_12: f32,
    #[serde(default)]
    pub(crate) f_rest_13: f32,
    #[serde(default)]
    pub(crate) f_rest_14: f32,
    #[serde(default)]
    pub(crate) f_rest_15: f32,
    #[serde(default)]
    pub(crate) f_rest_16: f32,
    #[serde(default)]
    pub(crate) f_rest_17: f32,
    #[serde(default)]
    pub(crate) f_rest_18: f32,
    #[serde(default)]
    pub(crate) f_rest_19: f32,
    #[serde(default)]
    pub(crate) f_rest_20: f32,
    #[serde(default)]
    pub(crate) f_rest_21: f32,
    #[serde(default)]
    pub(crate) f_rest_22: f32,
    #[serde(default)]
    pub(crate) f_rest_23: f32,
    #[serde(default)]
    pub(crate) f_rest_24: f32,
    #[serde(default)]
    pub(crate) f_rest_25: f32,
    #[serde(default)]
    pub(crate) f_rest_26: f32,
    #[serde(default)]
    pub(crate) f_rest_27: f32,
    #[serde(default)]
    pub(crate) f_rest_28: f32,
    #[serde(default)]
    pub(crate) f_rest_29: f32,
    #[serde(default)]
    pub(crate) f_rest_30: f32,
    #[serde(default)]
    pub(crate) f_rest_31: f32,
    #[serde(default)]
    pub(crate) f_rest_32: f32,
    #[serde(default)]
    pub(crate) f_rest_33: f32,
    #[serde(default)]
    pub(crate) f_rest_34: f32,
    #[serde(default)]
    pub(crate) f_rest_35: f32,
    #[serde(default)]
    pub(crate) f_rest_36: f32,
    #[serde(default)]
    pub(crate) f_rest_37: f32,
    #[serde(default)]
    pub(crate) f_rest_38: f32,
    #[serde(default)]
    pub(crate) f_rest_39: f32,
    #[serde(default)]
    pub(crate) f_rest_40: f32,
    #[serde(default)]
    pub(crate) f_rest_41: f32,
    #[serde(default)]
    pub(crate) f_rest_42: f32,
    #[serde(default)]
    pub(crate) f_rest_43: f32,
    #[serde(default)]
    pub(crate) f_rest_44: f32,

    // Color overrides. Potentially quantized.
    #[serde(default, alias = "r", skip_serializing, deserialize_with = "de_quant")]
    pub(crate) red: Option<f32>,
    #[serde(default, alias = "g", skip_serializing, deserialize_with = "de_quant")]
    pub(crate) green: Option<f32>,
    #[serde(default, alias = "b", skip_serializing, deserialize_with = "de_quant")]
    pub(crate) blue: Option<f32>,
}

macro_rules! sh_coeffs_array {
    ($self:expr) => {
        [
            $self.f_rest_0,
            $self.f_rest_1,
            $self.f_rest_2,
            $self.f_rest_3,
            $self.f_rest_4,
            $self.f_rest_5,
            $self.f_rest_6,
            $self.f_rest_7,
            $self.f_rest_8,
            $self.f_rest_9,
            $self.f_rest_10,
            $self.f_rest_11,
            $self.f_rest_12,
            $self.f_rest_13,
            $self.f_rest_14,
            $self.f_rest_15,
            $self.f_rest_16,
            $self.f_rest_17,
            $self.f_rest_18,
            $self.f_rest_19,
            $self.f_rest_20,
            $self.f_rest_21,
            $self.f_rest_22,
            $self.f_rest_23,
            $self.f_rest_24,
            $self.f_rest_25,
            $self.f_rest_26,
            $self.f_rest_27,
            $self.f_rest_28,
            $self.f_rest_29,
            $self.f_rest_30,
            $self.f_rest_31,
            $self.f_rest_32,
            $self.f_rest_33,
            $self.f_rest_34,
            $self.f_rest_35,
            $self.f_rest_36,
            $self.f_rest_37,
            $self.f_rest_38,
            $self.f_rest_39,
            $self.f_rest_40,
            $self.f_rest_41,
            $self.f_rest_42,
            $self.f_rest_43,
            $self.f_rest_44,
        ]
    };
}

impl PlyGaussian {
    pub(crate) fn sh_rest_coeffs(&self) -> [f32; 45] {
        sh_coeffs_array!(self)
    }
}

fn de_quant_sh<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = u8::deserialize(deserializer)? as f32 / (u8::MAX - 1) as f32;
    Ok((value - 0.5) * 8.0)
}

#[derive(Deserialize)]
pub(crate) struct QuantSh {
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_0: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_1: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_2: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_3: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_4: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_5: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_6: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_7: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_8: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_9: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_10: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_11: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_12: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_13: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_14: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_15: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_16: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_17: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_18: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_19: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_20: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_21: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_22: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_23: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_24: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_25: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_26: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_27: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_28: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_29: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_30: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_31: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_32: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_33: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_34: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_35: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_36: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_37: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_38: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_39: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_40: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_41: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_42: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_43: f32,
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) f_rest_44: f32,
}

impl QuantSh {
    pub(crate) fn coeffs(&self) -> [f32; 45] {
        sh_coeffs_array!(self)
    }
}
