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

#[derive(Serialize, Deserialize)]
pub(crate) struct PlyGaussian {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
    pub(crate) scale_0: f32,
    pub(crate) scale_1: f32,
    pub(crate) scale_2: f32,
    pub(crate) opacity: f32,
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

    #[serde(alias = "r")]
    #[serde(skip_serializing)]
    pub(crate) red: Option<f32>,
    #[serde(alias = "g")]
    #[serde(skip_serializing)]
    pub(crate) green: Option<f32>,
    #[serde(alias = "b")]
    #[serde(skip_serializing)]
    pub(crate) blue: Option<f32>,

    #[serde(default)]
    pub(crate) sh_rest: ShCoeffs,
}

#[derive(Serialize, Deserialize, Default)]
pub(crate) struct ShCoeffs {
    #[serde(default, rename = "f_rest_0")]
    pub(crate) c0: f32,
    #[serde(default, rename = "f_rest_1")]
    pub(crate) c1: f32,
    #[serde(default, rename = "f_rest_2")]
    pub(crate) c2: f32,
    #[serde(default, rename = "f_rest_3")]
    pub(crate) c3: f32,
    #[serde(default, rename = "f_rest_4")]
    pub(crate) c4: f32,
    #[serde(default, rename = "f_rest_5")]
    pub(crate) c5: f32,
    #[serde(default, rename = "f_rest_6")]
    pub(crate) c6: f32,
    #[serde(default, rename = "f_rest_7")]
    pub(crate) c7: f32,
    #[serde(default, rename = "f_rest_8")]
    pub(crate) c8: f32,
    #[serde(default, rename = "f_rest_9")]
    pub(crate) c9: f32,
    #[serde(default, rename = "f_rest_10")]
    pub(crate) c10: f32,
    #[serde(default, rename = "f_rest_11")]
    pub(crate) c11: f32,
    #[serde(default, rename = "f_rest_12")]
    pub(crate) c12: f32,
    #[serde(default, rename = "f_rest_13")]
    pub(crate) c13: f32,
    #[serde(default, rename = "f_rest_14")]
    pub(crate) c14: f32,
    #[serde(default, rename = "f_rest_15")]
    pub(crate) c15: f32,
    #[serde(default, rename = "f_rest_16")]
    pub(crate) c16: f32,
    #[serde(default, rename = "f_rest_17")]
    pub(crate) c17: f32,
    #[serde(default, rename = "f_rest_18")]
    pub(crate) c18: f32,
    #[serde(default, rename = "f_rest_19")]
    pub(crate) c19: f32,
    #[serde(default, rename = "f_rest_20")]
    pub(crate) c20: f32,
    #[serde(default, rename = "f_rest_21")]
    pub(crate) c21: f32,
    #[serde(default, rename = "f_rest_22")]
    pub(crate) c22: f32,
    #[serde(default, rename = "f_rest_23")]
    pub(crate) c23: f32,
    #[serde(default, rename = "f_rest_24")]
    pub(crate) c24: f32,
    #[serde(default, rename = "f_rest_25")]
    pub(crate) c25: f32,
    #[serde(default, rename = "f_rest_26")]
    pub(crate) c26: f32,
    #[serde(default, rename = "f_rest_27")]
    pub(crate) c27: f32,
    #[serde(default, rename = "f_rest_28")]
    pub(crate) c28: f32,
    #[serde(default, rename = "f_rest_29")]
    pub(crate) c29: f32,
    #[serde(default, rename = "f_rest_30")]
    pub(crate) c30: f32,
    #[serde(default, rename = "f_rest_31")]
    pub(crate) c31: f32,
    #[serde(default, rename = "f_rest_32")]
    pub(crate) c32: f32,
    #[serde(default, rename = "f_rest_33")]
    pub(crate) c33: f32,
    #[serde(default, rename = "f_rest_34")]
    pub(crate) c34: f32,
    #[serde(default, rename = "f_rest_35")]
    pub(crate) c35: f32,
    #[serde(default, rename = "f_rest_36")]
    pub(crate) c36: f32,
    #[serde(default, rename = "f_rest_37")]
    pub(crate) c37: f32,
    #[serde(default, rename = "f_rest_38")]
    pub(crate) c38: f32,
    #[serde(default, rename = "f_rest_39")]
    pub(crate) c39: f32,
    #[serde(default, rename = "f_rest_40")]
    pub(crate) c40: f32,
    #[serde(default, rename = "f_rest_41")]
    pub(crate) c41: f32,
    #[serde(default, rename = "f_rest_42")]
    pub(crate) c42: f32,
    #[serde(default, rename = "f_rest_43")]
    pub(crate) c43: f32,
    #[serde(default, rename = "f_rest_44")]
    pub(crate) c44: f32,
}

impl ShCoeffs {
    pub(crate) fn from_slice(sh_rest: &[f32]) -> Self {
        let get = |index| sh_rest.get(index).copied().unwrap_or(0.0);
        Self {
            c0: get(0),
            c1: get(1),
            c2: get(2),
            c3: get(3),
            c4: get(4),
            c5: get(5),
            c6: get(6),
            c7: get(7),
            c8: get(8),
            c9: get(9),
            c10: get(10),
            c11: get(11),
            c12: get(12),
            c13: get(13),
            c14: get(14),
            c15: get(15),
            c16: get(16),
            c17: get(17),
            c18: get(18),
            c19: get(19),
            c20: get(20),
            c21: get(21),
            c22: get(22),
            c23: get(23),
            c24: get(24),
            c25: get(25),
            c26: get(26),
            c27: get(27),
            c28: get(28),
            c29: get(29),
            c30: get(30),
            c31: get(31),
            c32: get(32),
            c33: get(33),
            c34: get(34),
            c35: get(35),
            c36: get(36),
            c37: get(37),
            c38: get(38),
            c39: get(39),
            c40: get(40),
            c41: get(41),
            c42: get(42),
            c43: get(43),
            c44: get(44),
        }
    }

    pub(crate) fn to_array(&self) -> [f32; 45] {
        [
            self.c0, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8,
            self.c9, self.c10, self.c11, self.c12, self.c13, self.c14, self.c15, self.c16,
            self.c17, self.c18, self.c19, self.c20, self.c21, self.c22, self.c23, self.c24,
            self.c25, self.c26, self.c27, self.c28, self.c29, self.c30, self.c31, self.c32,
            self.c33, self.c34, self.c35, self.c36, self.c37, self.c38, self.c39, self.c40,
            self.c41, self.c42, self.c43, self.c44,
        ]
    }
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
            sh_rest: ShCoeffs::from_slice(sh_rest),
        }
    }

    pub(crate) fn sh_rest_coeffs(&self) -> [f32; 45] {
        self.sh_rest.to_array()
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
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_0")]
    pub(crate) c0: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_1")]
    pub(crate) c1: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_2")]
    pub(crate) c2: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_3")]
    pub(crate) c3: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_4")]
    pub(crate) c4: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_5")]
    pub(crate) c5: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_6")]
    pub(crate) c6: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_7")]
    pub(crate) c7: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_8")]
    pub(crate) c8: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_9")]
    pub(crate) c9: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_10")]
    pub(crate) c10: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_11")]
    pub(crate) c11: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_12")]
    pub(crate) c12: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_13")]
    pub(crate) c13: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_14")]
    pub(crate) c14: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_15")]
    pub(crate) c15: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_16")]
    pub(crate) c16: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_17")]
    pub(crate) c17: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_18")]
    pub(crate) c18: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_19")]
    pub(crate) c19: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_20")]
    pub(crate) c20: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_21")]
    pub(crate) c21: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_22")]
    pub(crate) c22: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_23")]
    pub(crate) c23: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_24")]
    pub(crate) c24: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_25")]
    pub(crate) c25: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_26")]
    pub(crate) c26: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_27")]
    pub(crate) c27: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_28")]
    pub(crate) c28: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_29")]
    pub(crate) c29: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_30")]
    pub(crate) c30: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_31")]
    pub(crate) c31: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_32")]
    pub(crate) c32: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_33")]
    pub(crate) c33: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_34")]
    pub(crate) c34: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_35")]
    pub(crate) c35: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_36")]
    pub(crate) c36: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_37")]
    pub(crate) c37: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_38")]
    pub(crate) c38: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_39")]
    pub(crate) c39: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_40")]
    pub(crate) c40: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_41")]
    pub(crate) c41: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_42")]
    pub(crate) c42: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_43")]
    pub(crate) c43: f32,
    #[serde(default, deserialize_with = "de_quant_sh", rename = "f_rest_44")]
    pub(crate) c44: f32,
}

impl QuantSh {
    pub(crate) fn coeffs(&self) -> [f32; 45] {
        [
            self.c0, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8,
            self.c9, self.c10, self.c11, self.c12, self.c13, self.c14, self.c15, self.c16,
            self.c17, self.c18, self.c19, self.c20, self.c21, self.c22, self.c23, self.c24,
            self.c25, self.c26, self.c27, self.c28, self.c29, self.c30, self.c31, self.c32,
            self.c33, self.c34, self.c35, self.c36, self.c37, self.c38, self.c39, self.c40,
            self.c41, self.c42, self.c43, self.c44,
        ]
    }
}
