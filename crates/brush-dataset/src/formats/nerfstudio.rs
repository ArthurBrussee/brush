use super::{DatasetLoadResult, FormatError, find_mask_path, opengl_c2w_to_pose};
use crate::{
    Dataset,
    config::LoadDatasetConfig,
    scene::{LoadImage, SceneView},
};
use brush_render::camera::fov_to_focal;
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::CameraModel::{
    KannalaBrandt4, Pinhole, RadialTangential8,
};
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use brush_render::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use brush_serde::load_splat_from_ply;
use brush_vfs::BrushVfs;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncReadExt;

#[derive(serde::Deserialize, Clone)]
struct JsonScene {
    // Horizontal FOV.
    camera_angle_x: Option<f64>,
    // Vertical FOV.
    camera_angle_y: Option<f64>,

    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,

    /// Nerfstudio camera model: `"OPENCV"`, `"OPENCV_FISHEYE"`, or unset (pinhole).
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

    /// First radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k1: Option<f64>,
    /// Second radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k2: Option<f64>,
    /// Third radial distortion parameter used by `OPENCV_FISHEYE`
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by `OPENCV_FISHEYE`
    k4: Option<f64>,
    /// First tangential distortion parameter used by `OPENCV`
    p1: Option<f64>,
    /// Second tangential distortion parameter used by `OPENCV`
    p2: Option<f64>,

    frames: Vec<FrameData>,
}

#[derive(serde::Deserialize, Clone)]
struct FrameData {
    /// Nerfstudio camera model override for this frame.
    camera_model: Option<String>,

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

    /// First radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k1: Option<f64>,
    /// Second radial distortion parameter used by `OPENCV`/`OPENCV_FISHEYE`
    k2: Option<f64>,
    /// Third radial distortion parameter used by `OPENCV_FISHEYE`
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by `OPENCV_FISHEYE`
    k4: Option<f64>,
    /// First tangential distortion parameter used by `OPENCV`
    p1: Option<f64>,
    /// Second tangential distortion parameter used by `OPENCV`
    p2: Option<f64>,

    transform_matrix: Vec<Vec<f32>>,
    file_path: String,
}

/// Build a `CameraModel` from a nerfstudio `camera_model` string and the
/// k/p distortion coefficients available at this scope.
fn resolve_camera_model(
    model_name: Option<&str>,
    k1: Option<f64>,
    k2: Option<f64>,
    k3: Option<f64>,
    k4: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
) -> Result<CameraModel, FormatError> {
    let f = |o: Option<f64>| o.unwrap_or(0.0) as f32;
    match model_name {
        None | Some("PERSPECTIVE" | "perspective") => Ok(Pinhole),
        Some("OPENCV" | "opencv") => Ok(RadialTangential8(RadialTangential8Params {
            k1: f(k1),
            k2: f(k2),
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: f(p1),
            p2: f(p2),
        })),
        Some("OPENCV_FISHEYE" | "opencv_fisheye") => Ok(KannalaBrandt4(KannalaBrandt4Params {
            k1: f(k1),
            k2: f(k2),
            k3: f(k3),
            k4: f(k4),
        })),
        Some(other) => Err(FormatError::InvalidCamera(format!(
            "Unsupported nerfstudio camera_model `{other}`"
        ))),
    }
}

async fn read_transforms_file(
    scene: JsonScene,
    transforms_path: &Path,
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
    warnings: &mut Vec<String>,
) -> Result<Vec<SceneView>, FormatError> {
    let mut results = vec![];
    for frame in scene
        .frames
        .iter()
        .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
        .take(load_args.max_frames.unwrap_or(usize::MAX))
    {
        brush_async::yield_now().await;

        // NeRF 'transform_matrix' is a camera-to-world transform
        let transform_matrix: Vec<f32> = frame.transform_matrix.iter().flatten().copied().collect();
        if transform_matrix.len() != 16 {
            return Err(FormatError::InvalidFormat(format!(
                "frame '{}' has a {}-element transform_matrix, expected a 4x4 (16 elements)",
                frame.file_path,
                transform_matrix.len()
            )));
        }
        let transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();
        let (translation, rotation) = opengl_c2w_to_pose(transform);

        let mut path = transforms_path
            .parent()
            .expect("Transforms path must be a filename")
            .join(&frame.file_path);

        // Check if path exists.
        if vfs.reader_at_path(&path).await.is_err() {
            warnings.push(format!(
                "Skipped '{}': image file not found",
                frame.file_path
            ));
            continue;
        }

        // Assume png's by default if no extension is specified.
        if path.extension().is_none() {
            path = path.with_extension("png");
        }
        let mask_path = find_mask_path(&vfs, &path).map(|p| p.to_path_buf());
        let image = LoadImage::new(
            vfs.clone(),
            path,
            mask_path,
            load_args.max_resolution,
            load_args.alpha_mode,
        );

        let w = frame.w.or(scene.w);
        let h = frame.h.or(scene.h);
        // If the json omits the size, read it from the image header (cheap, no
        // full decode).
        let (w, h) = match (w, h) {
            (Some(w), Some(h)) => (w as u32, h as u32),
            _ => image.dimensions().await?,
        };
        let image_size = glam::uvec2(w, h);

        let camera_model = resolve_camera_model(
            frame
                .camera_model
                .as_deref()
                .or(scene.camera_model.as_deref()),
            frame.k1.or(scene.k1),
            frame.k2.or(scene.k2),
            frame.k3.or(scene.k3),
            frame.k4.or(scene.k4),
            frame.p1.or(scene.p1),
            frame.p2.or(scene.p2),
        )?;

        let fovx = frame
            .camera_angle_x
            .or(frame.fl_x.map(|fx| focal_to_fov(fx, w, &camera_model)))
            .or(scene.camera_angle_x)
            .or(scene.fl_x.map(|fx| focal_to_fov(fx, w, &camera_model)));

        let fovy = frame
            .camera_angle_y
            .or(frame.fl_y.map(|fy| focal_to_fov(fy, h, &camera_model)))
            .or(scene.camera_angle_y)
            .or(scene.fl_y.map(|fy| focal_to_fov(fy, h, &camera_model)));

        let (fovx, fovy) = match (fovx, fovy) {
            (None, None) => Err(FormatError::InvalidCamera(
                "Must have some kind of focal length".to_owned(),
            ))?,
            (None, Some(fovy)) => {
                let fovx = focal_to_fov(fov_to_focal(fovy, h, &camera_model), w, &camera_model);
                (fovx, fovy)
            }
            (Some(fovx), None) => {
                let fovy = focal_to_fov(fov_to_focal(fovx, w, &camera_model), h, &camera_model);
                (fovx, fovy)
            }
            (Some(fovx), Some(fovy)) => (fovx, fovy),
        };

        let cx = frame.cx.or(scene.cx);
        let cy = frame.cy.or(scene.cy);

        let cuv = glam::vec2(
            cx.map_or(0.5, |v| v / w as f64) as f32,
            cy.map_or(0.5, |v| v / h as f64) as f32,
        );

        let camera = Camera::new(translation, rotation, fovx, fovy, cuv, camera_model);

        if !camera.is_valid() {
            let msg = format!(
                "Skipped '{}': camera contains nan or inf values",
                frame.file_path
            );
            warnings.push(msg);
            continue;
        }

        let view = SceneView {
            image,
            camera,
            image_size,
        };
        results.push(view);
    }
    Ok(results)
}

pub async fn read_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Option<Result<DatasetLoadResult, FormatError>> {
    log::info!("Loading nerfstudio dataset");

    let json_files: Vec<_> = vfs.files_with_extension("json").collect();

    let transforms_path = if json_files.len() == 1 {
        json_files.first()?
    } else {
        // If there's multiple options, only pick files which are either exactly
        // transforms.json or end with transforms_train.json (a la transforms_train.json)
        vfs.files_ending_in("transforms.json")
            .next()
            .or_else(|| vfs.files_ending_in("transforms_train.json").next())?
    };
    let transforms_path = transforms_path.to_path_buf();
    Some(read_dataset_inner(vfs, load_args, json_files, transforms_path).await)
}

async fn read_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
    json_files: Vec<std::path::PathBuf>,
    transforms_path: std::path::PathBuf,
) -> Result<DatasetLoadResult, FormatError> {
    let mut warnings = Vec::new();

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
        &mut warnings,
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
        Some(
            read_transforms_file(
                val_scene,
                eval_trans_path,
                vfs.clone(),
                load_args,
                &mut warnings,
            )
            .await?,
        )
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

    let load_args = load_args.clone();

    let mut init_splat = None;

    if let Some(init_path) = train_scene.ply_file_path {
        let init_path = transforms_path
            .parent()
            .expect("Transforms path must be a filename")
            .join(init_path);

        let ply_data = vfs.reader_at_path(&init_path).await;

        if let Ok(ply_data) = ply_data {
            init_splat = Some(load_splat_from_ply(ply_data, load_args.subsample_points).await?);
        }
    }

    Ok(DatasetLoadResult {
        init_splat,
        dataset,
        warnings,
    })
}
