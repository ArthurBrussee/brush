use super::FormatError;
use super::find_mask_path;
use crate::{
    Dataset,
    config::LoadDataseConfig,
    scene::{LoadImage, SceneView},
};
use brush_render::camera::fov_to_focal;
use brush_render::camera::{Camera, focal_to_fov};
use brush_serde::{SplatMessage, load_splat_from_ply};
use brush_vfs::BrushVfs;
use burn::backend::wgpu::WgpuDevice;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio_with_wasm::alias as tokio_wasm;

#[derive(serde::Deserialize, Clone)]
#[allow(unused)] // not reading camera distortions yet.
struct JsonScene {
    // Horizontal FOV.
    camera_angle_x: Option<f64>,
    // Vertical FOV.
    camera_angle_y: Option<f64>,

    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,

    // Not really used atm.
    camera_model: Option<String>,
    // Nerfstudio doesn't mention this in their format? But fine to include really.
    ply_file_path: Option<String>,

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
    // Horizontal FOV.
    camera_angle_x: Option<f64>,
    // Vertical FOV.
    camera_angle_y: Option<f64>,

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

    // Nb: These are unused currently until we can optimize distorted cameras.
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

async fn read_transforms_file(
    scene: JsonScene,
    transforms_path: &Path,
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
) -> Result<Vec<SceneView>, FormatError> {
    let mut results = vec![];
    for frame in scene
        .frames
        .iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
    {
        tokio_wasm::task::yield_now().await;

        // NeRF 'transform_matrix' is a camera-to-world transform
        let transform_matrix: Vec<f32> = frame.transform_matrix.iter().flatten().copied().collect();
        let mut transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();
        // Swap basis to match camera format and reconstrunstion ply (if included).
        transform.y_axis *= -1.0;
        transform.z_axis *= -1.0;
        let (_, rotation, translation) = transform.to_scale_rotation_translation();

        let mut path = transforms_path
            .parent()
            .expect("Transforms path must be a filename")
            .join(&frame.file_path);

        // Assume png's by default if no extension is specified.
        if path.extension().is_none() {
            path = path.with_extension("png");
        }
        let mask_path = find_mask_path(&vfs, &path);

        let image = LoadImage::new(vfs.clone(), &path, mask_path, load_args.max_resolution).await;

        let image = match image {
            Ok(image) => image,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                log::warn!("Image not found: {path:?}");
                continue;
            }
            Err(e) => Err(e)?,
        };

        let w = frame.w.or(scene.w).unwrap_or(image.width() as f64) as u32;
        let h = frame.h.or(scene.h).unwrap_or(image.height() as f64) as u32;

        let fovx = frame
            .camera_angle_x
            .or(frame.fl_x.map(|fx| focal_to_fov(fx, w)))
            .or(scene.camera_angle_x)
            .or(scene.fl_x.map(|fx| focal_to_fov(fx, w)));

        let fovy = frame
            .camera_angle_y
            .or(frame.fl_y.map(|fy| focal_to_fov(fy, h)))
            .or(scene.camera_angle_y)
            .or(scene.fl_y.map(|fy| focal_to_fov(fy, h)));

        let (fovx, fovy) = match (fovx, fovy) {
            (None, None) => Err(FormatError::InvalidCamera(
                "Must have some kind of focal length".to_owned(),
            ))?,
            (None, Some(fovy)) => {
                let fovx = focal_to_fov(fov_to_focal(fovy, h), w);
                (fovx, fovy)
            }
            (Some(fovx), None) => {
                let fovy = focal_to_fov(fov_to_focal(fovx, w), h);
                (fovx, fovy)
            }
            (Some(fovx), Some(fovy)) => (fovx, fovy),
        };

        let cx = frame.cx.or(scene.cx).unwrap_or(w as f64 / 2.0);
        let cy = frame.cy.or(scene.cy).unwrap_or(h as f64 / 2.0);

        let cuv = glam::vec2((cx / w as f64) as f32, (cy / h as f64) as f32);

        let view = SceneView {
            image,
            camera: Camera::new(translation, rotation, fovx, fovy, cuv),
        };
        results.push(view);
    }
    Ok(results)
}

pub async fn read_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &WgpuDevice,
) -> Option<Result<(Option<SplatMessage>, Dataset), FormatError>> {
    log::info!("Loading nerfstudio dataset");

    let json_files: Vec<_> = vfs.files_with_extension("json").collect();

    let transforms_path = if json_files.len() == 1 {
        json_files.first().cloned()?
    } else {
        // If there's multiple options, only pick files which are either exactly
        // transforms.json or end with transforms_train.json (a la transforms_train.json)
        vfs.files_ending_in("transforms.json")
            .next()
            .or_else(|| vfs.files_ending_in("transforms_train.json").next())?
    };

    Some(read_dataset_inner(vfs, load_args, device, json_files, transforms_path).await)
}

async fn read_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &WgpuDevice,
    json_files: Vec<std::path::PathBuf>,
    transforms_path: std::path::PathBuf,
) -> Result<(Option<SplatMessage>, Dataset), FormatError> {
    let mut buf = String::new();
    vfs.reader_at_path(&transforms_path)
        .await?
        .read_to_string(&mut buf)
        .await?;
    let train_scene: JsonScene = serde_json::from_str(&buf)?;
    let train_handles = read_transforms_file(
        train_scene.clone(),
        &transforms_path,
        vfs.clone(),
        load_args,
    )
    .await?;

    // Use transforms_val as eval, or _test if no _val is present. (Brush doesn't really have any notion of a test set).
    let eval_trans_path = json_files
        .iter()
        .find(|x| x.ends_with("transforms_val.json"))
        .or_else(|| {
            json_files
                .iter()
                .find(|x| x.ends_with("transforms_test.json"))
        });
    // If a separate eval file is specified, read it.
    let val_views = if let Some(eval_trans_path) = eval_trans_path {
        let mut json_str = String::new();
        vfs.reader_at_path(eval_trans_path)
            .await?
            .read_to_string(&mut json_str)
            .await?;
        let val_scene = serde_json::from_str(&json_str)?;
        Some(read_transforms_file(val_scene, eval_trans_path, vfs.clone(), load_args).await?)
    } else {
        None
    };

    let mut train_views = vec![];
    let mut eval_views = vec![];
    for (i, view) in train_handles.into_iter().enumerate() {
        if let Some(eval_period) = load_args.eval_split_every {
            // Include extra eval images only when the dataset doesn't have them.
            if i % eval_period == 0 && val_views.is_none() {
                eval_views.push(view);
            } else {
                train_views.push(view);
            }
        } else {
            train_views.push(view);
        }
    }

    if let Some(val_views) = val_views {
        eval_views.extend(val_views);
    }

    let dataset = Dataset::from_views(train_views, eval_views);

    let device = device.clone();
    let load_args = load_args.clone();

    let mut init_splat = None;

    if let Some(init_path) = train_scene.ply_file_path {
        let init_path = transforms_path
            .parent()
            .expect("Transforms path must be a filename")
            .join(init_path);

        let ply_data = vfs.reader_at_path(&init_path).await;

        if let Ok(ply_data) = ply_data {
            init_splat = Some(
                load_splat_from_ply(ply_data, load_args.subsample_points, device.clone()).await?,
            );
        }
    }

    Ok((init_splat, dataset))
}
