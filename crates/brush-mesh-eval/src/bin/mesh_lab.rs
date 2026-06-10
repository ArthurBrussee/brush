//! Extraction-parameter lab (experiment harness, not part of the product
//! pipeline): runs GOF extraction on a fixed splat PLY with overridable
//! `ExtractConfig` knobs and writes the mesh for `brush-mesh-eval` to score.

use std::pin::pin;

use anyhow::Context;
use brush_dataset::config::LoadDataseConfig;
use brush_dataset::load_dataset;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_vfs::DataSource;
use clap::Parser;
use tokio_stream::StreamExt;

#[derive(Parser)]
struct Args {
    #[arg(value_name = "PATH_OR_URL")]
    source: DataSource,
    #[arg(long)]
    ply: std::path::PathBuf,
    #[arg(long)]
    out_mesh: std::path::PathBuf,
    #[arg(long, default_value = "0.6")]
    iso: f32,
    #[arg(long, default_value = "0")]
    smooth_iters: u32,
    #[arg(long, default_value = "100")]
    min_component: usize,
    #[arg(long, default_value = "false")]
    opacity_radius: bool,
    #[arg(long, default_value = "true")]
    octa: bool,
    /// Mesh region: union of camera frustums truncated at this distance.
    #[arg(long, default_value = "4.0")]
    far: f32,
    /// Seed only gaussians with carve-field center alpha at most this
    /// (1.0 disables selection).
    #[arg(long, default_value = "0.5")]
    seed_alpha: f32,
    #[arg(long, default_value = "1920")]
    resolution: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_target(false)
        .try_init();
    let args = Args::parse();

    brush_process::burn_init_setup().await;
    let device = brush_process::wait_for_device().await.clone();
    let device: burn::tensor::Device = device.into();

    let bytes = std::fs::read(&args.ply)
        .with_context(|| format!("read splat ply {}", args.ply.display()))?;
    let mut stream = pin!(brush_serde::stream_splat_from_ply(
        std::io::Cursor::new(bytes),
        None,
        false
    ));
    let mut splat_msg = None;
    while let Some(msg) = stream.next().await {
        splat_msg = Some(msg?);
    }
    let splat_msg = splat_msg.context("PLY produced no splats")?;
    let mode = splat_msg
        .meta
        .render_mode
        .unwrap_or(SplatRenderMode::Default);
    let splats = splat_msg.data.into_splats(&device, mode);
    log::info!("Loaded {} splats", splats.num_splats());

    let vfs = args.source.into_vfs().await?;
    let load_cfg = LoadDataseConfig {
        max_frames: None,
        max_resolution: args.resolution,
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
    };
    let dataset = load_dataset(vfs, &load_cfg).await?;

    let mut views: Vec<(brush_render::camera::Camera, glam::UVec2)> = Vec::new();
    for view in dataset.dataset.train.views.iter() {
        let (w, h) = view.image.dimensions().await.unwrap_or((1, 1));
        views.push((view.camera.with_pinhole(), glam::uvec2(w, h)));
    }

    let tetra_points = brush_mesh::tetra_points::TetraPointsConfig {
        opacity_radius: args.opacity_radius,
        octahedron: args.octa,
        far: args.far,
        ..Default::default()
    };
    let cfg = brush_mesh::ExtractConfig {
        tetra_points,
        iso_value: args.iso,
        smooth_iters: args.smooth_iters,
        min_component_faces: args.min_component,
        seed_center_alpha: args.seed_alpha,
    };
    let mesh = brush_mesh::extract_mesh(splats, &views, &cfg).await;

    if let Some(parent) = args.out_mesh.parent() {
        std::fs::create_dir_all(parent)?;
    }
    brush_mesh::ply::write_ply_file(&mesh, &args.out_mesh)
        .with_context(|| format!("writing mesh {}", args.out_mesh.display()))?;
    log::info!("Wrote {}", args.out_mesh.display());
    Ok(())
}
