//! `--extract-mesh` driver.
//!
//! Loads the dataset's cameras + the splat PLY, runs the GOF-style
//! [`brush_mesh::extract_mesh`] pipeline, and writes the resulting mesh as
//! a binary PLY. Bypasses the trainer's `ProcessMessage` actor — the
//! pipeline is sequential and there's no interactive UI to feed.

use std::path::Path;
use std::pin::pin;

use anyhow::Context;
use brush_dataset::config::LoadDataseConfig;
use brush_dataset::load_dataset;
use brush_mesh::tetra_points::TetraPointsConfig;
use brush_mesh::{ExtractConfig, extract_mesh, ply};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_vfs::DataSource;
use glam::UVec2;
use tokio_stream::StreamExt;

use crate::Cli;

pub async fn run(cli: &Cli) -> anyhow::Result<()> {
    brush_process::burn_init_setup().await;
    let device = brush_process::wait_for_device().await.clone();
    let device: burn::tensor::Device = device.into();

    let source: DataSource = cli
        .source
        .clone()
        .context("--extract-mesh requires a source")?;
    log::info!("Loading dataset from {source:?}");
    let vfs = source.into_vfs().await?;

    // 1. Load the splat PLY from the VFS (matches train_stream's lookup —
    // prefer `init.ply`, otherwise the last PLY alphabetically).
    let mut ply_paths: Vec<_> = vfs.files_with_extension("ply").collect();
    ply_paths.sort();
    let main_ply = ply_paths
        .iter()
        .find(|p| p.file_name().is_some_and(|n| n == "init.ply"))
        .or_else(|| ply_paths.last())
        .context("no PLY file found in dataset")?
        .to_path_buf();
    log::info!("Loading splat PLY {main_ply:?}");
    let reader = vfs.reader_at_path(&main_ply).await?;
    let mut splat_stream = pin!(brush_serde::stream_splat_from_ply(
        reader,
        cli.splat_subsample,
        false
    ));
    let mut splats_data = None;
    while let Some(msg) = splat_stream.next().await {
        let msg = msg?;
        splats_data = Some(msg);
    }
    let splat_msg = splats_data.context("PLY produced no splats")?;
    let mode = splat_msg
        .meta
        .render_mode
        .unwrap_or(SplatRenderMode::Default);
    let splats = splat_msg.data.into_splats(&device, mode);
    log::info!("Loaded {} splats", splats.num_splats());

    // 2. Load the dataset's cameras. We don't need the image bytes — only
    // the camera intrinsics/extrinsics and image resolutions.
    let load_cfg = LoadDataseConfig {
        max_frames: None,
        max_resolution: 1920,
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
    };
    let dataset = load_dataset(vfs, &load_cfg).await?;
    // Keep an owned copy of train views for the optional PSNR eval pass
    // (we need access to the GT image bytes via SceneView::image).
    let train_scene_views: Vec<brush_dataset::scene::SceneView> =
        dataset.dataset.train.views.iter().cloned().collect();
    // Force pinhole cameras for the entire extraction pipeline. The
    // mesh wgpu rasterizer is intrinsically pinhole (no per-pixel
    // distortion in the shader), so if the dataset cameras have any
    // distortion model (KB4, RT8, …) the alpha integration & colour
    // sampling would happen against a distorted render but the
    // mesh-render at eval time would be undistorted — producing the
    // "frame jump" between the splat and mesh panels. Re-deriving the
    // camera with `CameraModel::Pinhole` keeps the focal proportional
    // to the same fov (so the framing is right) and just drops the
    // distortion polynomial. Pinhole-source datasets like bonsai are
    // unaffected; distorted datasets get a slight crop/stretch that's
    // identical across both renderers.
    use brush_render::kernels::camera_model::CameraModel;
    let to_pinhole = |mut c: brush_render::camera::Camera| {
        c.camera_model = CameraModel::Pinhole;
        c
    };
    let mut views: Vec<(brush_render::camera::Camera, UVec2)> = Vec::new();
    for v in dataset.dataset.train.views.iter() {
        let (w, h) = v
            .image
            .output_dimensions()
            .await
            .context("failed to read image dims for camera")?;
        views.push((to_pinhole(v.camera), UVec2::new(w, h)));
    }
    if let Some(eval) = dataset.dataset.eval.as_ref() {
        for v in eval.views.iter() {
            let (w, h) = v
                .image
                .output_dimensions()
                .await
                .context("failed to read eval image dims")?;
            views.push((to_pinhole(v.camera), UVec2::new(w, h)));
        }
    }
    log::info!(
        "Using {} views (train + eval) for opacity integration",
        views.len()
    );

    // Log a few training-camera poses in f3d-compatible form. Brush's view
    // frame is +X right, +Y down, +Z forward; f3d expects the world-space
    // view-up direction, which is the camera's −Y axis rotated to world.
    for (i, (cam, sz)) in views.iter().enumerate().take(3) {
        let fwd = cam.rotation * glam::Vec3::Z;
        let up = cam.rotation * -glam::Vec3::Y;
        let focal_point = cam.position + fwd;
        let fov_deg = cam.fov_y.to_degrees();
        log::info!(
            "view[{i}]: --camera-position={:.4},{:.4},{:.4} \
             --camera-focal-point={:.4},{:.4},{:.4} \
             --camera-view-up={:.4},{:.4},{:.4} \
             --camera-view-angle={:.2} (img {}x{})",
            cam.position.x,
            cam.position.y,
            cam.position.z,
            focal_point.x,
            focal_point.y,
            focal_point.z,
            up.x,
            up.y,
            up.z,
            fov_deg,
            sz.x,
            sz.y
        );
    }

    // 3. Run the extraction.
    let cfg = ExtractConfig {
        tetra_points: TetraPointsConfig {
            near: cli.mesh_near,
            far: cli.mesh_far,
            ..Default::default()
        },
        iso_value: cli.iso_value,
    };
    // Clone splats before extraction consumes them — we need a handle
    // for the splat-render-vs-mesh PSNR comparison below. Splats is a
    // burn Module so the clone is shallow (tensor handle copies, not
    // a deep GPU buffer copy).
    let splats_for_eval = splats.clone();
    let mesh = extract_mesh(splats, &views, &cfg).await;

    // 4. Optionally write the mesh PLY.
    if cli.skip_mesh_write {
        log::info!(
            "Skipping mesh PLY write ({} verts, {} faces)",
            mesh.vertices.len(),
            mesh.faces.len()
        );
    } else {
        let t_write = std::time::Instant::now();
        write_mesh(&mesh, &cli.out_mesh)?;
        log::info!(
            "mesh_write: {:.2}s ({} verts, {} faces → {})",
            t_write.elapsed().as_secs_f64(),
            mesh.vertices.len(),
            mesh.faces.len(),
            cli.out_mesh.display()
        );
    }

    // 5. Optional PSNR eval: render the mesh at each training viewpoint
    // with an in-process wgpu pipeline (vertex-colored triangles, no
    // lighting — rendered RGB = barycentric vertex-color interpolation)
    // and compare to the GT image.
    if cli.eval_views > 0 {
        let eval_dir = cli
            .out_mesh
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("eval_renders");
        log::info!(
            "Evaluating {} train views with wgpu unlit renderer",
            cli.eval_views.min(train_scene_views.len()),
        );
        crate::mesh_eval::eval_psnr(
            &mesh,
            &train_scene_views,
            &splats_for_eval,
            cli.eval_views,
            &eval_dir,
        )
        .await?;
    }

    Ok(())
}

fn write_mesh(mesh: &brush_mesh::Mesh, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    // BufWriter is critical: the PLY writer emits ~4-byte writes per
    // float field per vertex (~20M writes for a 5M-vertex mesh).
    // Without buffering each one hits the OS, which on Windows is the
    // bulk of the 65 s mesh-write time observed on garden.
    let file = std::fs::File::create(path)?;
    let mut buf = std::io::BufWriter::with_capacity(1 << 20, file);
    ply::write_ply(&mut buf, mesh)?;
    std::io::Write::flush(&mut buf)
}
