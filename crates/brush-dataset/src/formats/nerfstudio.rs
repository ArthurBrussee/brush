use super::DataStream;
use crate::brush_vfs::BrushVfs;
use crate::splat_import::load_splat_from_ply;
use crate::splat_import::SplatMessage;
use crate::stream_fut_parallel;
use crate::LoadDataseConfig;
use crate::{clamp_img_to_max_size, Dataset};
use anyhow::Context;
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use brush_render::camera::{focal_to_fov, fov_to_focal, Camera};
use brush_render::Backend;
use brush_train::scene::SceneView;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio_stream::StreamExt;

#[derive(serde::Deserialize, Clone)]
#[allow(unused)] // not reading camera distortions yet.
struct JsonScene {
    // Simple synthetic nerf camera model.
    camera_angle_x: Option<f64>,
    // Not really used atm.
    camera_model: Option<String>,

    // Nerfstudio doesn't mention this in their format? But fine to include really.
    ply_file_path: Option<String>,

    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,
    /// Principal point x
    cx: Option<f64>,
    /// Principal point y
    cy: Option<f64>,
    /// Image width
    w: Option<f64>,
    /// Image height
    h: Option<f64>,

    /// First radial distortion parameter used by [`OPENCV`, `OPENCV_FISHEYE`]
    k1: Option<f64>,
    /// Second radial distortion parameter used by [`OPENCV`, `OPENCV_FISHEYE`]
    k2: Option<f64>,
    /// Third radial distortion parameter used by [`OPENCV_FISHEYE`]
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by [`OPENCV_FISHEYE`]
    k4: Option<f64>,
    /// First tangential distortion parameter used by [`OPENCV`]
    p1: Option<f64>,
    /// Second tangential distortion parameter used by [`OPENCV`]
    p2: Option<f64>,

    frames: Vec<FrameData>,
}

#[derive(serde::Deserialize, Clone)]
#[allow(unused)] // not reading camera distortions yet.
struct FrameData {
    // Nerfstudio format
    //
    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,
    /// Principal point x
    cx: Option<f64>,
    /// Principal point y
    cy: Option<f64>,
    /// Image width. Should be an integer but read as float, fine to truncate.
    w: Option<f64>,
    /// Image height. Should be an integer but read as float, fine to truncate.
    h: Option<f64>,

    // TODO: These are unused currently.
    /// First radial distortion parameter used by [`OPENCV`, `OPENCV_FISHEYE`]
    k1: Option<f64>,
    /// Second radial distortion parameter used by [`OPENCV`, `OPENCV_FISHEYE`]
    k2: Option<f64>,
    /// Third radial distortion parameter used by [`OPENCV_FISHEYE`]
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by [`OPENCV_FISHEYE`]
    k4: Option<f64>,
    /// First tangential distortion parameter used by [`OPENCV`]
    p1: Option<f64>,
    /// Second tangential distortion parameter used by [`OPENCV`]
    p2: Option<f64>,

    transform_matrix: Vec<Vec<f32>>,
    file_path: String,
}

fn read_transforms_file(
    scene: JsonScene,
    transforms_path: PathBuf,
    vfs: BrushVfs,
    load_args: &LoadDataseConfig,
) -> Vec<impl Future<Output = anyhow::Result<SceneView>>> {
    let iter = scene
        .frames
        .into_iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .map(move |frame| {
            let mut archive = vfs.clone();
            let load_args = load_args.clone();
            let transforms_path = transforms_path.clone();

            async move {
                // NeRF 'transform_matrix' is a camera-to-world transform
                let transform_matrix: Vec<f32> =
                    frame.transform_matrix.iter().flatten().copied().collect();
                let mut transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();
                // Swap basis to match camera format and reconstrunstion ply (if included).
                transform.y_axis *= -1.0;
                transform.z_axis *= -1.0;
                let (_, rotation, translation) = transform.to_scale_rotation_translation();

                // Read the imageat the specified path, fallback to default .png extension.
                let mut path = transforms_path
                    .clone()
                    .parent()
                    .expect("Transforms path must be a filename")
                    .join(&frame.file_path);
                if path.extension().is_none() {
                    path = path.with_extension("png");
                }

                let mut img_buffer = vec![];
                archive
                    .open_path(&path)
                    .await?
                    .read_to_end(&mut img_buffer)
                    .await?;

                let comp_span = tracing::trace_span!("Decompress image").entered();
                drop(comp_span);

                // Create a cursor from the buffer
                let mut image = tracing::trace_span!("Decode image")
                    .in_scope(|| image::load_from_memory(&img_buffer))?;

                let w = frame.w.or(scene.w).unwrap_or(image.width() as f64) as u32;
                let h = frame.h.or(scene.h).unwrap_or(image.height() as f64) as u32;

                if let Some(max_resolution) = load_args.max_resolution {
                    image = clamp_img_to_max_size(image, max_resolution);
                }

                let focal_x = frame
                    .fl_x
                    .or(scene.fl_x)
                    .or(scene.camera_angle_x.map(|fx| fov_to_focal(fx, w)))
                    .context("Must have a focal length of some kind.")?;

                // Read fov y or derive it from the input.
                let focal_y = frame.fl_y.or(scene.fl_y).unwrap_or(focal_x);

                let fovx = focal_to_fov(focal_x, w);
                let fovy = focal_to_fov(focal_y, h);

                let cx = frame.cx.or(scene.cx).unwrap_or(w as f64 / 2.0);
                let cy = frame.cy.or(scene.cy).unwrap_or(h as f64 / 2.0);

                let cuv = glam::vec2((cx / w as f64) as f32, (cy / h as f64) as f32);

                let view = SceneView {
                    name: frame.file_path.clone(),
                    camera: Camera::new(translation, rotation, fovx, fovy, cuv),
                    image: Arc::new(image),
                };
                anyhow::Result::<SceneView>::Ok(view)
            }
        });

    iter.collect()
}

pub async fn read_dataset<B: Backend>(
    mut vfs: BrushVfs,
    load_args: &LoadDataseConfig,
    device: &B::Device,
) -> Result<(DataStream<SplatMessage<B>>, DataStream<Dataset>)> {
    log::info!("Loading nerfstudio dataset");

    let json_files: Vec<_> = vfs
        .file_names()
        .filter(|&n| n.extension().is_some_and(|p| p == "json"))
        .map(|x| x.to_path_buf())
        .collect();

    let transforms_path = if json_files.len() == 1 {
        json_files.first().cloned().expect("Must have 1 json file")
    } else {
        let train = json_files.iter().find(|x| {
            x.file_name()
                .is_some_and(|p| p.to_string_lossy().contains("_train"))
        });
        let Some(train) = train else {
            anyhow::bail!("No json file found.");
        };
        train.clone()
    };

    let mut buf = String::new();
    vfs.open_path(&transforms_path)
        .await?
        .read_to_string(&mut buf)
        .await?;
    let train_scene: JsonScene = serde_json::from_str(&buf)?;

    let mut train_handles = read_transforms_file(
        train_scene.clone(),
        transforms_path.clone(),
        vfs.clone(),
        load_args,
    );

    if let Some(subsample) = load_args.subsample_frames {
        train_handles = train_handles
            .into_iter()
            .step_by(subsample as usize)
            .collect();
    }

    let load_args_clone = load_args.clone();

    let mut data_clone = vfs.clone();

    let dataset_stream = try_fn_stream(|emitter| async move {
        let mut train_views = vec![];
        let mut eval_views = vec![];

        let eval_trans_path = json_files.iter().find(|x| {
            x.file_name()
                .is_some_and(|p| p.to_string_lossy().contains("_val"))
        });

        // If a seperate eval file is specified, read it.
        let val_stream = if let Some(eval_trans_path) = eval_trans_path {
            let mut json_str = String::new();
            data_clone
                .open_path(eval_trans_path)
                .await?
                .read_to_string(&mut json_str)
                .await?;
            let val_scene = serde_json::from_str(&json_str)?;
            Some(read_transforms_file(
                val_scene,
                eval_trans_path.clone(),
                data_clone,
                &load_args_clone,
            ))
        } else {
            None
        };

        log::info!("Loading transforms_test.json");
        // Not entirely sure yet if we want to report stats on both test
        // and eval, atm this skips "transforms_test.json" even if it's there.

        let train_handles = stream_fut_parallel(train_handles);
        let mut train_handles = std::pin::pin!(train_handles);

        let mut i = 0;
        while let Some(view) = train_handles.next().await {
            if let Some(eval_period) = load_args_clone.eval_split_every {
                // Include extra eval images only when the dataset doesn't have them.
                if i % eval_period == 0 && val_stream.is_some() {
                    eval_views.push(view?);
                } else {
                    train_views.push(view?);
                }
            } else {
                train_views.push(view?);
            }

            emitter
                .emit(Dataset::from_views(train_views.clone(), eval_views.clone()))
                .await;

            i += 1;
        }

        if let Some(val_stream) = val_stream {
            let val_handles = stream_fut_parallel(val_stream);
            let mut val_handles = std::pin::pin!(val_handles);
            while let Some(view) = val_handles.next().await {
                eval_views.push(view?);
                emitter
                    .emit(Dataset::from_views(train_views.clone(), eval_views.clone()))
                    .await;
            }
        }

        Ok(())
    });

    let device = device.clone();
    let load_args = load_args.clone();

    let splat_stream = try_fn_stream(|emitter| async move {
        if let Some(init) = train_scene.ply_file_path {
            let init_path = transforms_path
                .parent()
                .expect("Transforms path must be a filename")
                .join(init);
            let ply_data = vfs.open_path(&init_path).await;

            if let Ok(ply_data) = ply_data {
                let splat_stream =
                    load_splat_from_ply(ply_data, load_args.subsample_points, device.clone());

                let mut splat_stream = std::pin::pin!(splat_stream);

                // If sucesfully extracted, sent this splat as an initial splat.
                while let Some(Ok(splat)) = splat_stream.next().await {
                    emitter.emit(splat).await;
                }
            }
        }
        Ok(())
    });

    Ok((Box::pin(splat_stream), Box::pin(dataset_stream)))
}
