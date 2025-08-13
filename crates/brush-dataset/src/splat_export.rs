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
                c0: get_sh(0),
                c1: get_sh(1),
                c2: get_sh(2),
                c3: get_sh(3),
                c4: get_sh(4),
                c5: get_sh(5),
                c6: get_sh(6),
                c7: get_sh(7),
                c8: get_sh(8),
                c9: get_sh(9),
                c10: get_sh(10),
                c11: get_sh(11),
                c12: get_sh(12),
                c13: get_sh(13),
                c14: get_sh(14),
                c15: get_sh(15),
                c16: get_sh(16),
                c17: get_sh(17),
                c18: get_sh(18),
                c19: get_sh(19),
                c20: get_sh(20),
                c21: get_sh(21),
                c22: get_sh(22),
                c23: get_sh(23),
                c24: get_sh(24),
                c25: get_sh(25),
                c26: get_sh(26),
                c27: get_sh(27),
                c28: get_sh(28),
                c29: get_sh(29),
                c30: get_sh(30),
                c31: get_sh(31),
                c32: get_sh(32),
                c33: get_sh(33),
                c34: get_sh(34),
                c35: get_sh(35),
                c36: get_sh(36),
                c37: get_sh(37),
                c38: get_sh(38),
                c39: get_sh(39),
                c40: get_sh(40),
                c41: get_sh(41),
                c42: get_sh(42),
                c43: get_sh(43),
                c44: get_sh(44),
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
