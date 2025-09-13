use std::collections::BTreeMap;
use std::vec;

use brush_render::gaussian_splats::Splats;
use brush_render::sh::sh_coeffs_for_degree;
use burn::prelude::Backend;
use serde::{Serialize, Serializer};
use serde_ply::{SerializeError, SerializeOptions};

// Dynamic PLY structure that only includes needed SH coefficients
#[derive(Debug)]
struct DynamicPlyGaussian {
    // Core properties
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
        // Create a BTreeMap to maintain field order
        let mut map = BTreeMap::new();

        // Add core properties
        map.insert("x", self.x);
        map.insert("y", self.y);
        map.insert("z", self.z);
        map.insert("scale_0", self.scale_0);
        map.insert("scale_1", self.scale_1);
        map.insert("scale_2", self.scale_2);
        map.insert("opacity", self.opacity);
        map.insert("rot_0", self.rot_0);
        map.insert("rot_1", self.rot_1);
        map.insert("rot_2", self.rot_2);
        map.insert("rot_3", self.rot_3);

        // Add DC components
        map.insert("f_dc_0", self.f_dc_0);
        map.insert("f_dc_1", self.f_dc_1);
        map.insert("f_dc_2", self.f_dc_2);

        let sh_names = [
            "f_rest_0",
            "f_rest_1",
            "f_rest_2",
            "f_rest_3",
            "f_rest_4",
            "f_rest_5",
            "f_rest_6",
            "f_rest_7",
            "f_rest_8",
            "f_rest_9",
            "f_rest_10",
            "f_rest_11",
            "f_rest_12",
            "f_rest_13",
            "f_rest_14",
            "f_rest_15",
            "f_rest_16",
            "f_rest_17",
            "f_rest_18",
            "f_rest_19",
            "f_rest_20",
            "f_rest_21",
            "f_rest_22",
            "f_rest_23",
            "f_rest_24",
            "f_rest_25",
            "f_rest_26",
            "f_rest_27",
            "f_rest_28",
            "f_rest_29",
            "f_rest_30",
            "f_rest_31",
            "f_rest_32",
            "f_rest_33",
            "f_rest_34",
            "f_rest_35",
            "f_rest_36",
            "f_rest_37",
            "f_rest_38",
            "f_rest_39",
            "f_rest_40",
            "f_rest_41",
            "f_rest_42",
            "f_rest_43",
            "f_rest_44",
            "f_rest_45",
        ];
        for (name, val) in sh_names.iter().zip(&self.rest_coeffs) {
            map.insert(name, *val);
        }
        // Serialize as a map
        map.serialize(serializer)
    }
}

#[derive(Serialize)]
struct DynamicPly {
    vertex: Vec<DynamicPlyGaussian>,
}
pub use burn_cubecl::{CubeRuntime, cubecl::Compiler, tensor::CubeTensor};

async fn read_splat_data<B: Backend>(splats: Splats<B>) -> DynamicPly {
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

            DynamicPlyGaussian {
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
                rest_coeffs,
            }
        })
        .collect();
    DynamicPly { vertex: vertices }
}

pub async fn splat_to_ply<B: Backend>(splats: Splats<B>) -> Result<Vec<u8>, SerializeError> {
    let splats = splats.with_normed_rotations();
    let sh_degree = splats.sh_degree();
    let ply = read_splat_data(splats.clone()).await;

    let comments = vec![
        "Exported from Brush".to_owned(),
        "Vertical axis: y".to_owned(),
        format!("SH degree: {}", sh_degree),
    ];
    serde_ply::to_bytes(&ply, SerializeOptions::binary_le().with_comments(comments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::load_splat_from_ply;
    use crate::test_utils::create_test_splats;
    use brush_render::MainBackend;
    use burn::backend::wgpu::WgpuDevice;
    use std::io::Cursor;

    async fn assert_coeffs_match(orig: &Splats<MainBackend>, imported: &Splats<MainBackend>) {
        let orig_sh: Vec<f32> = orig
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .into_vec()
            .expect("Failed to convert SH coefficients to vector");
        let import_sh: Vec<f32> = imported
            .sh_coeffs
            .val()
            .into_data_async()
            .await
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

    #[tokio::test]
    async fn test_sh_degree_exports() {
        for degree in 0..=2 {
            let splats = create_test_splats(degree);
            assert_eq!(splats.sh_degree(), degree);

            let ply_data = read_splat_data(splats.clone()).await;
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

    #[tokio::test]
    async fn test_ply_field_count_matches_sh_degree() {
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

    #[tokio::test]
    async fn test_roundtrip_sh_coefficient_ordering() {
        let device = WgpuDevice::default();

        for degree in [0, 1, 2] {
            let original_splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(original_splats.clone())
                .await
                .expect("Failed to serialize splats");

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None, device.clone())
                .await
                .expect("Failed to deserialize splats");
            let imported_splats = imported_message.splats;

            assert_eq!(imported_splats.sh_degree(), degree);
            assert_coeffs_match(&original_splats, &imported_splats).await;
        }
    }
}
