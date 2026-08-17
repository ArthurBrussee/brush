//! Dev-only calibration harness for `GT_AXIS_SIGN`
//! (`crates/brush-train/src/normals.rs`).
//!
//! Renders the splat pseudo-normal map (through the feature channel) for a
//! trained model and scores it against the dataset's GT monocular normal
//! maps under all 8 axis-sign combinations, by masked mean cosine
//! similarity. The winning combination is the sign that maps the GT
//! predictor's camera convention onto brush's.
//!
//! Usage:
//! ```text
//! cargo run -p brush-bench-test --example normal_calib --release -- \
//!   <dataset_dir> <trained.ply> [max_views]
//! ```

use std::{path::Path, sync::Arc};

use brush_dataset::config::LoadDatasetConfig;
use brush_render::bwd::render_splats_with_features;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_train::normals::splat_camera_normals;
use brush_vfs::BrushVfs;
use clap::Parser;

/// `LoadDatasetConfig` only derives `clap::Args`; flatten it into a tiny
/// parser so defaults can be materialized without listing every field.
#[derive(Parser)]
struct DefaultLoadConfig {
    #[command(flatten)]
    cfg: LoadDatasetConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dataset_dir = args.get(1).expect("usage: normal_calib <dataset> <ply>");
    let ply_path = args.get(2).expect("usage: normal_calib <dataset> <ply>");
    let max_views: usize = args.get(3).map_or(8, |s| s.parse().expect("max_views"));

    let device: burn::tensor::Device =
        brush_cube::test_helpers::test_device().await.into();
    let device = device.autodiff();

    let file = tokio::fs::File::open(ply_path).await?;
    let msg = brush_serde::load_splat_from_ply(tokio::io::BufReader::new(file), None).await?;
    let splats = msg.data.into_splats(&device, SplatRenderMode::Default);
    println!("loaded {} splats from {ply_path}", splats.num_splats());

    let vfs = Arc::new(BrushVfs::from_path(Path::new(dataset_dir)).await?);
    let cfg = DefaultLoadConfig::parse_from(["normal_calib"]).cfg;
    let res = brush_dataset::load_dataset(vfs, &cfg).await?;
    let views = res.dataset.train.views;

    const COMBOS: [[f32; 3]; 8] = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
    ];
    let mut cos_sums = [0.0f64; 8];
    let mut pix_count = 0u64;
    let mut used_views = 0usize;

    for view in views.iter() {
        if used_views >= max_views {
            break;
        }
        let img = view.image.load().await?;
        let (w, h) = (img.width(), img.height());
        let Some(normal_res) = view.image.load_normal(w, h).await else {
            continue;
        };
        let gt = normal_res?.into_rgba8().into_vec();

        let feats = splat_camera_normals(&splats, &view.camera);
        let out = render_splats_with_features(
            splats.clone(),
            &view.camera,
            glam::uvec2(w, h),
            glam::Vec3::ZERO,
            feats,
        )
        .await;
        let pred: Vec<f32> = out
            .features
            .expect("features enabled")
            .into_data_async()
            .await?
            .into_vec()
            .expect("f32 features");

        for pix in 0..(w as usize * h as usize) {
            if gt[pix * 4 + 3] < 128 {
                continue;
            }
            let gn = [
                f32::from(gt[pix * 4]) / 127.5 - 1.0,
                f32::from(gt[pix * 4 + 1]) / 127.5 - 1.0,
                f32::from(gt[pix * 4 + 2]) / 127.5 - 1.0,
            ];
            let p = &pred[pix * 3..pix * 3 + 3];
            let pl = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let gl = (gn[0] * gn[0] + gn[1] * gn[1] + gn[2] * gn[2]).sqrt();
            if pl < 1e-6 || gl < 1e-6 {
                continue;
            }
            for (ci, s) in COMBOS.iter().enumerate() {
                let dot = p[0] * s[0] * gn[0] + p[1] * s[1] * gn[1] + p[2] * s[2] * gn[2];
                cos_sums[ci] += f64::from(dot / (pl * gl));
            }
            pix_count += 1;
        }
        used_views += 1;
        println!("scored view {used_views} ({w}x{h})");
    }

    println!("\nmasked mean cosine over {used_views} views, {pix_count} px:");
    let mut rows: Vec<(usize, f64)> = cos_sums
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s / pix_count as f64))
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, mean) in rows {
        let s = COMBOS[i];
        println!(
            "  [{:+.0} {:+.0} {:+.0}]  ->  {mean:.4}",
            s[0], s[1], s[2]
        );
    }
    Ok(())
}
