use crate::quant::{decode_quat, decode_vec_8_8_8_8, decode_vec_11_10_11};

use glam::{Quat, Vec3, Vec4};
use serde::Deserialize;
use serde::{self, Deserializer};

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
pub struct QuantSplat {
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

#[brush_serde_macros::generate_sh_fields]
#[derive(Deserialize)]
pub struct PlyGaussian {
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

    // This marker field will be replaced with 72 f_rest_N fields by the proc macro
    #[serde(default)]
    pub(crate) _sh_rest_fields: (),

    // Color overrides. Potentially quantized.
    #[serde(default, alias = "r", skip_serializing, deserialize_with = "de_quant")]
    pub(crate) red: Option<f32>,
    #[serde(default, alias = "g", skip_serializing, deserialize_with = "de_quant")]
    pub(crate) green: Option<f32>,
    #[serde(default, alias = "b", skip_serializing, deserialize_with = "de_quant")]
    pub(crate) blue: Option<f32>,
}

// Generate the sh_rest_coeffs() method using proc macro
brush_serde_macros::impl_coeffs!(PlyGaussian);

fn de_quant_sh<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = u8::deserialize(deserializer)? as f32 / (u8::MAX - 1) as f32;
    Ok((value - 0.5) * 8.0)
}

#[brush_serde_macros::generate_sh_fields]
#[derive(Deserialize)]
pub struct QuantSh {
    // This marker field will be replaced with 72 f_rest_N fields by the proc macro
    #[serde(default, deserialize_with = "de_quant_sh")]
    pub(crate) _sh_rest_fields: (),
}

// Generate the coeffs() method using proc macro
brush_serde_macros::impl_coeffs!(QuantSh);
