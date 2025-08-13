use std::vec;

use brush_render::gaussian_splats::Splats;
use burn::prelude::Backend;
use serde::Serialize;
use serde_ply::{SerializeError, SerializeOptions};

use crate::parsed_gaussian::PlyGaussian;

#[derive(Serialize)]
struct Ply {
    vertex: Vec<PlyGaussian>,
}

async fn read_splat_data<B: Backend>(splats: Splats<B>) -> Ply {
    let means = splats
        .means
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let log_scales = splats
        .log_scales
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let rotations = splats
        .rotation
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let opacities = splats
        .raw_opacity
        .val()
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");
    let sh_coeffs = splats
        .sh_coeffs
        .val()
        .permute([0, 2, 1]) // Permute to inria format ([n, channel, coeffs]).
        .into_data_async()
        .await
        .into_vec()
        .expect("Unreachable");

    let sh_coeffs_num = splats.sh_coeffs.dims()[1];

    let vertices = (0..splats.num_splats())
        .map(|i| {
            let i = i as usize;
            // Read SH data from [coeffs, channel] format to
            let sh_start = i * sh_coeffs_num * 3;
            let sh_end = (i + 1) * sh_coeffs_num * 3;
            let splat_sh = &sh_coeffs[sh_start..sh_end];
            let [sh_red, sh_green, sh_blue] = [
                &splat_sh[0..sh_coeffs_num],
                &splat_sh[sh_coeffs_num..sh_coeffs_num * 2],
                &splat_sh[sh_coeffs_num * 2..sh_coeffs_num * 3],
            ];
            let sh_coeffs_rest = [&sh_red[1..], &sh_green[1..], &sh_blue[1..]].concat();
            let get_sh = |index| sh_coeffs_rest.get(index).copied().unwrap_or(0.0);

            PlyGaussian {
                x: means[i * 3],
                y: means[i * 3 + 1],
                z: means[i * 3 + 2],
                scale_0: log_scales[i * 3],
                scale_1: log_scales[i * 3 + 1],
                scale_2: log_scales[i * 3 + 2],
                rot_0: rotations[i * 4],
                rot_1: rotations[i * 4 + 1],
                rot_2: rotations[i * 4 + 2],
                rot_3: rotations[i * 4 + 3],
                opacity: opacities[i],
                f_dc_0: sh_red[0],
                f_dc_1: sh_green[0],
                f_dc_2: sh_blue[0],
                red: None,
                green: None,
                blue: None,
                f_rest_0: get_sh(0),
                f_rest_1: get_sh(1),
                f_rest_2: get_sh(2),
                f_rest_3: get_sh(3),
                f_rest_4: get_sh(4),
                f_rest_5: get_sh(5),
                f_rest_6: get_sh(6),
                f_rest_7: get_sh(7),
                f_rest_8: get_sh(8),
                f_rest_9: get_sh(9),
                f_rest_10: get_sh(10),
                f_rest_11: get_sh(11),
                f_rest_12: get_sh(12),
                f_rest_13: get_sh(13),
                f_rest_14: get_sh(14),
                f_rest_15: get_sh(15),
                f_rest_16: get_sh(16),
                f_rest_17: get_sh(17),
                f_rest_18: get_sh(18),
                f_rest_19: get_sh(19),
                f_rest_20: get_sh(20),
                f_rest_21: get_sh(21),
                f_rest_22: get_sh(22),
                f_rest_23: get_sh(23),
                f_rest_24: get_sh(24),
                f_rest_25: get_sh(25),
                f_rest_26: get_sh(26),
                f_rest_27: get_sh(27),
                f_rest_28: get_sh(28),
                f_rest_29: get_sh(29),
                f_rest_30: get_sh(30),
                f_rest_31: get_sh(31),
                f_rest_32: get_sh(32),
                f_rest_33: get_sh(33),
                f_rest_34: get_sh(34),
                f_rest_35: get_sh(35),
                f_rest_36: get_sh(36),
                f_rest_37: get_sh(37),
                f_rest_38: get_sh(38),
                f_rest_39: get_sh(39),
                f_rest_40: get_sh(40),
                f_rest_41: get_sh(41),
                f_rest_42: get_sh(42),
                f_rest_43: get_sh(43),
                f_rest_44: get_sh(44),
            }
        })
        .collect();
    Ply { vertex: vertices }
}

pub async fn splat_to_ply<B: Backend>(splats: Splats<B>) -> Result<Vec<u8>, SerializeError> {
    let splats = splats.with_normed_rotations();
    let ply = read_splat_data(splats.clone()).await;

    let comments = vec![
        "Exported from Brush".to_owned(),
        "Vertical axis: y".to_owned(),
    ];
    serde_ply::to_bytes(&ply, SerializeOptions::binary_le().with_comments(comments))
}
