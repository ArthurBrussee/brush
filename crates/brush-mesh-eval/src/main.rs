//! Standalone mesh-eval tool: renders an extracted mesh at dataset
//! viewpoints and reports PSNR vs the splat render and the GT images,
//! writing a labeled `GT | splat | mesh | depth` grid.
//!
//! Mesh extraction itself lives in the trainer (`--export-mesh-every`);
//! this tool only evaluates already-extracted meshes.

mod eval;
mod render;

use std::pin::pin;

use anyhow::Context;
use brush_dataset::config::LoadDataseConfig;
use brush_dataset::load_dataset;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_vfs::DataSource;
use clap::Parser;
use tokio_stream::StreamExt;

#[derive(Parser)]
#[command(about = "Evaluate an extracted mesh against its dataset + splat")]
struct Args {
    /// Dataset source (cameras + GT images).
    #[arg(value_name = "PATH_OR_URL")]
    source: DataSource,

    /// Splat PLY to compare against.
    #[arg(long)]
    ply: std::path::PathBuf,

    /// Mesh PLY to evaluate. Renders land in `{mesh dir}/eval_renders`.
    #[arg(long)]
    mesh: std::path::PathBuf,

    /// Number of eval viewpoints (zoomed-out picks, spread across the capture).
    #[arg(long, default_value = "6")]
    eval_views: usize,

    /// Max image resolution for the eval render panels.
    #[arg(long, default_value = "1920")]
    eval_resolution: u32,
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

    let mesh_file = std::fs::File::open(&args.mesh)
        .with_context(|| format!("open mesh {}", args.mesh.display()))?;
    let mesh = brush_mesh::ply::read_ply(&mut std::io::BufReader::new(mesh_file))
        .context("read mesh ply")?;
    log::info!(
        "Loaded mesh: {} verts, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    let vfs = args.source.into_vfs().await?;
    let load_cfg = LoadDataseConfig {
        max_frames: None,
        max_resolution: args.eval_resolution,
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
    };
    let dataset = load_dataset(vfs, &load_cfg).await?;

    let eval_dir = args
        .mesh
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("eval_renders");
    eval::eval_psnr(
        &mesh,
        &dataset.dataset.train.views,
        &splats,
        args.eval_views,
        &eval_dir,
    )
    .await?;
    Ok(())
}
