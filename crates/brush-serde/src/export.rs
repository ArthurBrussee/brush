use std::vec;

use brush_render::gaussian_splats::Splats;
use brush_render::sh::sh_coeffs_for_degree;
use burn::prelude::Backend;
use burn::tensor::Transaction;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use serde_ply::{SerializeError, SerializeOptions};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Failed to fetch splat data from GPU")]
    FetchFailed,
    #[error("Failed to convert tensor data to f32 - data may be corrupted")]
    DataConversion,
    #[error("PLY serialization failed: {0}")]
    Serialize(#[from] SerializeError),
}

// Dynamic PLY structure that only includes needed SH coefficients
#[derive(Debug)]
struct DynamicPlyGaussian {
    x: f32,
    y: f32,
    z: f32,
    scale_0: f32,
    scale_1: f32,
    scale_2: f32,
    opacity: f32,
    rot_0: f32,
    rot_1: f32,
    rot_2: f32,
    rot_3: f32,
    f_dc_0: f32,
    f_dc_1: f32,
    f_dc_2: f32,
    rest_coeffs: Vec<f32>,
}

impl Serialize for DynamicPlyGaussian {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Calculate total number of fields: 11 core + 3 DC + rest_coeffs
        let field_count = 14 + self.rest_coeffs.len();
        let mut state = serializer.serialize_struct("DynamicPlyGaussian", field_count)?;

        state.serialize_field("x", &self.x)?;
        state.serialize_field("y", &self.y)?;
        state.serialize_field("z", &self.z)?;
        state.serialize_field("scale_0", &self.scale_0)?;
        state.serialize_field("scale_1", &self.scale_1)?;
        state.serialize_field("scale_2", &self.scale_2)?;
        state.serialize_field("opacity", &self.opacity)?;
        state.serialize_field("rot_0", &self.rot_0)?;
        state.serialize_field("rot_1", &self.rot_1)?;
        state.serialize_field("rot_2", &self.rot_2)?;
        state.serialize_field("rot_3", &self.rot_3)?;

        // Serialize DC components
        state.serialize_field("f_dc_0", &self.f_dc_0)?;
        state.serialize_field("f_dc_1", &self.f_dc_1)?;
        state.serialize_field("f_dc_2", &self.f_dc_2)?;

        // Serialize rest coefficients.
        const SH_NAMES: [&str; 72] = brush_serde_macros::sh_field_names!();
        for (name, val) in SH_NAMES.iter().zip(&self.rest_coeffs) {
            state.serialize_field(name, val)?;
        }

        state.end()
    }
}

#[derive(Serialize)]
struct DynamicPly {
    vertex: Vec<DynamicPlyGaussian>,
}

async fn read_splat_data<B: Backend>(splats: Splats<B>) -> Result<DynamicPly, ExportError> {
    let data = Transaction::default()
        .register(splats.transforms.val())
        .register(splats.raw_opacities.val())
        .register(splats.sh_coeffs.val().permute([0, 2, 1])) // Permute to inria format ([n, channel, coeffs]).)
        .execute_async()
        .await
        .map_err(|_fetch| ExportError::FetchFailed)?;

    let vecs: Vec<Vec<f32>> = data
        .into_iter()
        .map(|x| x.into_vec().map_err(|_convert| ExportError::DataConversion))
        .collect::<Result<Vec<_>, _>>()?;

    let [transforms, raw_opacities, sh_coeffs]: [Vec<f32>; 3] = vecs
        .try_into()
        .map_err(|_convert| ExportError::DataConversion)?;

    let sh_coeffs_num = splats.sh_coeffs.dims()[1];
    let sh_degree = splats.sh_degree();

    // Calculate how many rest coefficients we should export based on the actual SH degree
    // SH coefficients structure:
    // - DC component (degree 0): f_dc_0, f_dc_1, f_dc_2 (always present)
    // - Rest coefficients: f_rest_0 through f_rest_N (degree 1+)
    //
    // Total coefficients per channel = (degree + 1)^2
    // Rest coefficients per channel = total - 1 (excluding DC component)
    // Examples:
    // - Degree 0: 1 total, 0 rest coefficients per channel
    // - Degree 1: 4 total, 3 rest coefficients per channel
    // - Degree 2: 9 total, 8 rest coefficients per channel
    // - Degree 3: 16 total, 15 rest coefficients per channel
    let coeffs_per_channel = sh_coeffs_for_degree(sh_degree) as usize;
    let rest_coeffs_per_channel = coeffs_per_channel - 1;

    let vertices = (0..splats.num_splats())
        .map(|i| {
            let i = i as usize;
            // Read SH data from [coeffs, channel] format
            let sh_start = i * sh_coeffs_num * 3;
            let sh_end = (i + 1) * sh_coeffs_num * 3;
            let splat_sh = &sh_coeffs[sh_start..sh_end];
            let [sh_red, sh_green, sh_blue] = [
                &splat_sh[0..sh_coeffs_num],
                &splat_sh[sh_coeffs_num..sh_coeffs_num * 2],
                &splat_sh[sh_coeffs_num * 2..sh_coeffs_num * 3],
            ];
            let sh_red_rest = if sh_red.len() > 1 && rest_coeffs_per_channel > 0 {
                &sh_red[1..=rest_coeffs_per_channel]
            } else {
                &[]
            };
            let sh_green_rest = if sh_green.len() > 1 && rest_coeffs_per_channel > 0 {
                &sh_green[1..=rest_coeffs_per_channel]
            } else {
                &[]
            };
            let sh_blue_rest = if sh_blue.len() > 1 && rest_coeffs_per_channel > 0 {
                &sh_blue[1..=rest_coeffs_per_channel]
            } else {
                &[]
            };

            let rest_coeffs = [sh_red_rest, sh_green_rest, sh_blue_rest].concat();
            // transforms layout: means(3) + rotations(4) + log_scales(3) = stride 10
            let t = i * 10;
            // Normalize the quaternion before export.
            let (r0, r1, r2, r3): (f32, f32, f32, f32) = (
                transforms[t + 3],
                transforms[t + 4],
                transforms[t + 5],
                transforms[t + 6],
            );
            let rn = (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3).sqrt().max(1e-12);
            DynamicPlyGaussian {
                x: transforms[t],
                y: transforms[t + 1],
                z: transforms[t + 2],
                scale_0: transforms[t + 7],
                scale_1: transforms[t + 8],
                scale_2: transforms[t + 9],
                rot_0: r0 / rn,
                rot_1: r1 / rn,
                rot_2: r2 / rn,
                rot_3: r3 / rn,
                opacity: raw_opacities[i],
                f_dc_0: sh_red[0],
                f_dc_1: sh_green[0],
                f_dc_2: sh_blue[0],
                rest_coeffs,
            }
        })
        .collect();
    Ok(DynamicPly { vertex: vertices })
}

pub async fn splat_to_ply<B: Backend>(splats: Splats<B>) -> Result<Vec<u8>, ExportError> {
    let sh_degree = splats.sh_degree();
    let ply = read_splat_data(splats.clone()).await?;

    let render_mode_str = if splats.render_mip { "mip" } else { "default" };

    let comments = vec![
        "Exported from Brush".to_owned(),
        "Vertical axis: y".to_owned(),
        format!("SH degree: {}", sh_degree),
        format!("SplatRenderMode: {}", render_mode_str),
    ];
    Ok(serde_ply::to_bytes(
        &ply,
        SerializeOptions::binary_le().with_comments(comments),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::load_splat_from_ply;
    use crate::test_utils::create_test_splats;
    use brush_render::MainBackend;
    use brush_render::gaussian_splats::SplatRenderMode;
    use std::io::Cursor;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn assert_coeffs_match(orig: &Splats<MainBackend>, imported: &Splats<MainBackend>) {
        let orig_sh: Vec<f32> = orig
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .unwrap()
            .into_vec()
            .expect("Failed to convert SH coefficients to vector");
        let import_sh: Vec<f32> = imported
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .unwrap()
            .into_vec()
            .expect("Failed to convert SH coefficients to vector");

        assert_eq!(orig_sh.len(), import_sh.len());
        for (i, (&orig, &imported)) in orig_sh.iter().zip(import_sh.iter()).enumerate() {
            assert!(
                (orig - imported).abs() < 1e-6_f32,
                "SH coeffs mismatch at index {i}: orig={orig}, imported={imported}",
            );
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sh_degree_exports() {
        let _device = brush_kernel::test_helpers::test_device().await;
        for degree in 0..=2 {
            let splats = create_test_splats(degree);
            assert_eq!(splats.sh_degree(), degree);

            let ply_data = read_splat_data(splats.clone()).await.unwrap();
            let expected_rest_coeffs = if degree == 0 {
                0
            } else {
                (sh_coeffs_for_degree(degree) - 1) * 3
            };

            assert_eq!(
                ply_data.vertex[0].rest_coeffs.len(),
                expected_rest_coeffs as usize
            );
            assert!(splat_to_ply(splats).await.is_ok());
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_ply_field_count_matches_sh_degree() {
        let _device = brush_kernel::test_helpers::test_device().await;
        let test_cases = [(0, 0), (1, 9), (2, 24)];

        for (degree, expected_rest_fields) in test_cases {
            let splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(splats).await.unwrap();
            let ply_string = String::from_utf8_lossy(&ply_bytes);

            let actual_rest_fields = ply_string.matches("property float f_rest_").count();
            assert_eq!(
                actual_rest_fields, expected_rest_fields,
                "Degree {degree} should have {expected_rest_fields} f_rest_ fields",
            );

            assert!(ply_string.contains("f_dc_0"));
            if expected_rest_fields > 0 {
                assert!(ply_string.contains("f_rest_0"));
                assert!(!ply_string.contains(&format!("f_rest_{expected_rest_fields}")));
            } else {
                assert!(!ply_string.contains("f_rest_0"));
            }
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_roundtrip_sh_coefficient_ordering() {
        let device = brush_kernel::test_helpers::test_device().await;

        for degree in [0, 1, 2] {
            let original_splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(original_splats.clone())
                .await
                .expect("Failed to serialize splats");

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None)
                .await
                .expect("Failed to deserialize splats");
            let imported_splats = imported_message
                .data
                .into_splats(&device, SplatRenderMode::Default);

            assert_eq!(imported_splats.sh_degree(), degree);
            assert_coeffs_match(&original_splats, &imported_splats).await;
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_export_roundtrip_multiple_splats() {
        use crate::test_utils::create_test_splats_with_count;

        let device = brush_kernel::test_helpers::test_device().await;
        let num_splats = 100;

        for degree in [0, 1, 2, 3] {
            let original = create_test_splats_with_count(degree, num_splats);
            assert_eq!(original.num_splats(), num_splats as u32);

            let ply_bytes = splat_to_ply(original.clone())
                .await
                .expect("Failed to export splats");

            assert!(!ply_bytes.is_empty(), "Exported PLY should not be empty");

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None)
                .await
                .expect("Failed to reimport exported splats");
            let imported = imported_message
                .data
                .into_splats(&device, SplatRenderMode::Default);

            assert_eq!(
                imported.num_splats(),
                num_splats as u32,
                "Splat count mismatch after roundtrip"
            );
            assert_eq!(imported.sh_degree(), degree);
            assert_coeffs_match(&original, &imported).await;
        }
    }
}
