use crate::quant::{decode_quat, decode_vec_8_8_8_8, decode_vec_11_10_11};

use glam::{Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use serde::{self, Deserializer};

fn de_vec_11_10_11<'de, D>(deserializer: D) -> Result<Vec3, D::Error>
where
    D: Deserializer<'de>,
{
    let value = u32::deserialize(deserializer)?;
    Ok(decode_vec_11_10_11(value))
}

fn de_packed_quat<'de, D>(deserializer: D) -> Result<Quat, D::Error>
where
    D: Deserializer<'de>,
{
    let value = u32::deserialize(deserializer)?;
    Ok(decode_quat(value))
}

fn de_vec_8_8_8_8<'de, D>(deserializer: D) -> Result<Vec4, D::Error>
where
    D: Deserializer<'de>,
{
    let value = u32::deserialize(deserializer)?;
    let vec = decode_vec_8_8_8_8(value);
    Ok(vec)
}

#[derive(Deserialize, Debug)]
pub(crate) struct QuantSplat {
    #[serde(alias = "packed_position", deserialize_with = "de_vec_11_10_11")]
    pub(crate) mean: Vec3,
    #[serde(alias = "packed_scale", deserialize_with = "de_vec_11_10_11")]
    pub(crate) log_scale: Vec3,
    #[serde(alias = "packed_rotation", deserialize_with = "de_packed_quat")]
    pub(crate) rotation: Quat,
    #[serde(alias = "packed_color", deserialize_with = "de_vec_8_8_8_8")]
    pub(crate) rgba: Vec4,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct PlyGaussian {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
    // Scale (log).
    pub(crate) scale_0: f32,
    pub(crate) scale_1: f32,
    pub(crate) scale_2: f32,
    pub(crate) opacity: f32,
    // Rot (w, x, y, z).
    pub(crate) rot_0: f32,
    pub(crate) rot_1: f32,
    pub(crate) rot_2: f32,
    pub(crate) rot_3: f32,
    #[serde(default)]
    pub(crate) f_dc_0: f32,
    #[serde(default)]
    pub(crate) f_dc_1: f32,
    #[serde(default)]
    pub(crate) f_dc_2: f32,

    // Optional r/g/b overrides.
    #[serde(alias = "r")]
    #[serde(skip_serializing)]
    pub(crate) red: Option<f32>,
    #[serde(alias = "g")]
    #[serde(skip_serializing)]
    pub(crate) green: Option<f32>,
    #[serde(alias = "b")]
    #[serde(skip_serializing)]
    pub(crate) blue: Option<f32>,

    // Higher order SH coefficients. Optional.
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
}

impl PlyGaussian {
    pub(crate) fn from_data(
        mean: Vec3,
        scale: Vec3,
        rot: Vec4,
        sh_dc: Vec3,
        opacity: f32,
        sh_rest: &[f32],
    ) -> Self {
        Self {
            x: mean.x,
            y: mean.y,
            z: mean.z,
            scale_0: scale.x,
            scale_1: scale.y,
            scale_2: scale.z,
            opacity,
            rot_0: rot.w,
            rot_1: rot.x,
            rot_2: rot.y,
            rot_3: rot.z,
            f_dc_0: sh_dc.x,
            f_dc_1: sh_dc.y,
            f_dc_2: sh_dc.z,
            red: None,
            green: None,
            blue: None,
            f_rest_0: sh_rest.first().copied().unwrap_or(0.0),
            f_rest_1: sh_rest.get(1).copied().unwrap_or(0.0),
            f_rest_2: sh_rest.get(2).copied().unwrap_or(0.0),
            f_rest_3: sh_rest.get(3).copied().unwrap_or(0.0),
            f_rest_4: sh_rest.get(4).copied().unwrap_or(0.0),
            f_rest_5: sh_rest.get(5).copied().unwrap_or(0.0),
            f_rest_6: sh_rest.get(6).copied().unwrap_or(0.0),
            f_rest_7: sh_rest.get(7).copied().unwrap_or(0.0),
            f_rest_8: sh_rest.get(8).copied().unwrap_or(0.0),
            f_rest_9: sh_rest.get(9).copied().unwrap_or(0.0),
            f_rest_10: sh_rest.get(10).copied().unwrap_or(0.0),
            f_rest_11: sh_rest.get(11).copied().unwrap_or(0.0),
            f_rest_12: sh_rest.get(12).copied().unwrap_or(0.0),
            f_rest_13: sh_rest.get(13).copied().unwrap_or(0.0),
            f_rest_14: sh_rest.get(14).copied().unwrap_or(0.0),
            f_rest_15: sh_rest.get(15).copied().unwrap_or(0.0),
            f_rest_16: sh_rest.get(16).copied().unwrap_or(0.0),
            f_rest_17: sh_rest.get(17).copied().unwrap_or(0.0),
            f_rest_18: sh_rest.get(18).copied().unwrap_or(0.0),
            f_rest_19: sh_rest.get(19).copied().unwrap_or(0.0),
            f_rest_20: sh_rest.get(20).copied().unwrap_or(0.0),
            f_rest_21: sh_rest.get(21).copied().unwrap_or(0.0),
            f_rest_22: sh_rest.get(22).copied().unwrap_or(0.0),
            f_rest_23: sh_rest.get(23).copied().unwrap_or(0.0),
            f_rest_24: sh_rest.get(24).copied().unwrap_or(0.0),
            f_rest_25: sh_rest.get(25).copied().unwrap_or(0.0),
            f_rest_26: sh_rest.get(26).copied().unwrap_or(0.0),
            f_rest_27: sh_rest.get(27).copied().unwrap_or(0.0),
            f_rest_28: sh_rest.get(28).copied().unwrap_or(0.0),
            f_rest_29: sh_rest.get(29).copied().unwrap_or(0.0),
            f_rest_30: sh_rest.get(30).copied().unwrap_or(0.0),
            f_rest_31: sh_rest.get(31).copied().unwrap_or(0.0),
            f_rest_32: sh_rest.get(32).copied().unwrap_or(0.0),
            f_rest_33: sh_rest.get(33).copied().unwrap_or(0.0),
            f_rest_34: sh_rest.get(34).copied().unwrap_or(0.0),
            f_rest_35: sh_rest.get(35).copied().unwrap_or(0.0),
            f_rest_36: sh_rest.get(36).copied().unwrap_or(0.0),
            f_rest_37: sh_rest.get(37).copied().unwrap_or(0.0),
            f_rest_38: sh_rest.get(38).copied().unwrap_or(0.0),
            f_rest_39: sh_rest.get(39).copied().unwrap_or(0.0),
            f_rest_40: sh_rest.get(40).copied().unwrap_or(0.0),
            f_rest_41: sh_rest.get(41).copied().unwrap_or(0.0),
            f_rest_42: sh_rest.get(42).copied().unwrap_or(0.0),
            f_rest_43: sh_rest.get(43).copied().unwrap_or(0.0),
            f_rest_44: sh_rest.get(44).copied().unwrap_or(0.0),
        }
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
    // Higher order SH coefficients. Optional.
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
        [
            self.f_rest_0,
            self.f_rest_1,
            self.f_rest_2,
            self.f_rest_3,
            self.f_rest_4,
            self.f_rest_5,
            self.f_rest_6,
            self.f_rest_7,
            self.f_rest_8,
            self.f_rest_9,
            self.f_rest_10,
            self.f_rest_11,
            self.f_rest_12,
            self.f_rest_13,
            self.f_rest_14,
            self.f_rest_15,
            self.f_rest_16,
            self.f_rest_17,
            self.f_rest_18,
            self.f_rest_19,
            self.f_rest_20,
            self.f_rest_21,
            self.f_rest_22,
            self.f_rest_23,
            self.f_rest_24,
            self.f_rest_25,
            self.f_rest_26,
            self.f_rest_27,
            self.f_rest_28,
            self.f_rest_29,
            self.f_rest_30,
            self.f_rest_31,
            self.f_rest_32,
            self.f_rest_33,
            self.f_rest_34,
            self.f_rest_35,
            self.f_rest_36,
            self.f_rest_37,
            self.f_rest_38,
            self.f_rest_39,
            self.f_rest_40,
            self.f_rest_41,
            self.f_rest_42,
            self.f_rest_43,
            self.f_rest_44,
        ]
    }
}

impl PlyGaussian {
    pub(crate) fn sh_rest_coeffs(&self) -> [f32; 45] {
        [
            self.f_rest_0,
            self.f_rest_1,
            self.f_rest_2,
            self.f_rest_3,
            self.f_rest_4,
            self.f_rest_5,
            self.f_rest_6,
            self.f_rest_7,
            self.f_rest_8,
            self.f_rest_9,
            self.f_rest_10,
            self.f_rest_11,
            self.f_rest_12,
            self.f_rest_13,
            self.f_rest_14,
            self.f_rest_15,
            self.f_rest_16,
            self.f_rest_17,
            self.f_rest_18,
            self.f_rest_19,
            self.f_rest_20,
            self.f_rest_21,
            self.f_rest_22,
            self.f_rest_23,
            self.f_rest_24,
            self.f_rest_25,
            self.f_rest_26,
            self.f_rest_27,
            self.f_rest_28,
            self.f_rest_29,
            self.f_rest_30,
            self.f_rest_31,
            self.f_rest_32,
            self.f_rest_33,
            self.f_rest_34,
            self.f_rest_35,
            self.f_rest_36,
            self.f_rest_37,
            self.f_rest_38,
            self.f_rest_39,
            self.f_rest_40,
            self.f_rest_41,
            self.f_rest_42,
            self.f_rest_43,
            self.f_rest_44,
        ]
    }
}

impl PlyGaussian {
    pub(crate) fn is_finite(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
            && self.rot_0.is_finite()
            && self.rot_1.is_finite()
            && self.rot_2.is_finite()
            && self.rot_3.is_finite()
            && self.opacity.is_finite()
            && self.scale_0.is_finite()
            && self.scale_1.is_finite()
            && self.scale_2.is_finite()
    }
}
