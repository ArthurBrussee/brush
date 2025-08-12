use std::vec;

use brush_render::gaussian_splats::Splats;
use burn::prelude::Backend;
use serde::Serialize;
use serde_ply::{PlyError, PlyFormat::BinaryLittleEndian};

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
        .filter_map(|i| {
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

            let mean = glam::vec3(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]);
            let scale = glam::vec3(
                log_scales[i * 3],
                log_scales[i * 3 + 1],
                log_scales[i * 3 + 2],
            );
            let rot = glam::Vec4::new(
                rotations[i * 4],
                rotations[i * 4 + 1],
                rotations[i * 4 + 2],
                rotations[i * 4 + 3],
            );
            let sh_dc = glam::vec3(sh_red[0], sh_green[0], sh_blue[0]);
            let sh_coeffs_rest = [&sh_red[1..], &sh_green[1..], &sh_blue[1..]].concat();
            let splat =
                PlyGaussian::from_data(mean, scale, rot, sh_dc, opacities[i], &sh_coeffs_rest);
            splat.is_finite().then_some(splat)
        })
        .collect();
    Ply { vertex: vertices }
}

pub async fn splat_to_ply<B: Backend>(splats: Splats<B>) -> Result<Vec<u8>, PlyError> {
    let splats = splats.with_normed_rotations();
    let ply = read_splat_data(splats.clone()).await;

    let comments = vec![
        "Exported from Brush".to_owned(),
        "Vertical axis: y".to_owned(),
    ];
    serde_ply::to_bytes(&ply, BinaryLittleEndian, comments)
}
