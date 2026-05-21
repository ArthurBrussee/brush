use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::{DatasetLoadResult, FormatError};
use crate::{
    Dataset,
    config::LoadDataseConfig,
    formats::find_mask_path,
    scene::{LoadImage, SceneView},
};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::CameraModel::{
    KannalaBrandt4, Pinhole, RadialTangential8,
};
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use brush_render::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use brush_render::{
    camera::{self, Camera},
    sh::rgb_to_sh,
};
use brush_serde::{ParseMetadata, SplatData, SplatMessage};
use brush_vfs::BrushVfs;
use colmap_reader::{ColmapCamera, ColmapCameraModel};
use itertools::Itertools;

fn find_img<'a>(vfs: &'a BrushVfs, name: &str) -> Option<&'a Path> {
    // Colmap only specifies an image name, not a full path. We brute force
    // search for the image in the archive.
    //
    // Make sure this path doesn't start with a '/' as the files_ending_in expects
    // things in that format (like a "filename with slashes").
    vfs.files_ending_in(name)
        .filter(|p| !p.iter().any(|f| f == "masks")) // Skip anything that is a mask.
        .min()
}

/// COLMAP can emit several independent sparse reconstructions (`sparse/0`,
/// `sparse/1`, ...) when the image graph is disconnected. They share no
/// coordinate frame and cannot be merged here, so we pick the one that
/// registered the most images (COLMAP's own "largest first" convention,
/// determined empirically rather than trusting directory names).
async fn select_colmap_model(vfs: &BrushVfs) -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = vfs
        .files_ending_in("cameras.bin")
        .chain(vfs.files_ending_in("cameras.txt"))
        .map(Path::to_path_buf)
        .collect();

    if candidates.len() <= 1 {
        return candidates.into_iter().next();
    }

    let mut best: Option<(usize, PathBuf)> = None;
    for cam in &candidates {
        let dir = cam
            .parent()
            .expect("colmap cameras file must have a parent");
        let is_binary = cam.extension().and_then(|e| e.to_str()) == Some("bin");
        let img_path = dir.join(if is_binary {
            "images.bin"
        } else {
            "images.txt"
        });

        let Some(count) = count_registered_images(vfs, &img_path, is_binary).await else {
            log::warn!(
                "Skipping colmap model '{}': can't read images",
                dir.display()
            );
            continue;
        };
        log::info!("Colmap model '{}' registered {count} images", dir.display());

        // Tie-break on path so the choice is deterministic (VFS iteration isn't).
        let better = best
            .as_ref()
            .is_none_or(|(bc, bp)| count > *bc || (count == *bc && cam < bp));
        if better {
            best = Some((count, cam.clone()));
        }
    }

    // If every candidate failed to read, fall through to a deterministic pick
    // so the caller still surfaces a proper parse error downstream.
    let chosen = best
        .map(|(_, p)| p)
        .or_else(|| candidates.iter().min().cloned())?;
    log::info!(
        "Selected colmap model '{}'",
        chosen
            .parent()
            .expect("colmap cameras file must have a parent")
            .display()
    );
    Some(chosen)
}

async fn count_registered_images(
    vfs: &BrushVfs,
    img_path: &Path,
    is_binary: bool,
) -> Option<usize> {
    let mut file = vfs.reader_at_path(img_path).await.ok()?;
    let imgs = colmap_reader::read_images(&mut file, is_binary, false)
        .await
        .ok()?;
    Some(imgs.len())
}

pub(crate) async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
) -> Option<Result<DatasetLoadResult, FormatError>> {
    log::info!("Loading colmap dataset");

    let cam_path = select_colmap_model(&vfs).await?;
    let dir = cam_path
        .parent()
        .expect("colmap cameras file must have a parent");
    let is_binary = cam_path.extension().and_then(|e| e.to_str()) == Some("bin");
    let img_path = dir.join(if is_binary {
        "images.bin"
    } else {
        "images.txt"
    });

    Some(load_dataset_inner(vfs, load_args, cam_path, img_path).await)
}

async fn load_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    cam_path: PathBuf,
    img_path: PathBuf,
) -> Result<DatasetLoadResult, FormatError> {
    let is_binary = cam_path.ends_with("cameras.bin");

    log::info!("Parsing colmap camera info");

    let load_args = load_args.clone();
    let vfs = vfs.clone();

    let vfs_init = vfs.clone();

    // Resolve points3d from the same reconstruction as the chosen cameras,
    // not an arbitrary one elsewhere in the VFS.
    let points_dir = cam_path
        .parent()
        .expect("colmap cameras file must have a parent")
        .to_path_buf();

    // One actor for both halves of the colmap load — the camera/image
    // parse and the points3d parse run concurrently on the same thread
    // (no cross-stream GPU concerns; this is pure CPU/I/O).
    let actor = brush_async::Actor::new("colmap-loader");
    let dataset = actor.run(move || async move {
        let mut cam_file = vfs.reader_at_path(&cam_path).await?;
        let cam_model_data = colmap_reader::read_cameras(&mut cam_file, is_binary).await?;
        let cam_model_data = cam_model_data
            .into_iter()
            .map(|cam| (cam.id, cam))
            .collect::<HashMap<_, _>>();
        let mut img_file = vfs.reader_at_path(&img_path).await?;
        let img_infos = colmap_reader::read_images(&mut img_file, is_binary, false).await?;
        let mut img_info_list = img_infos.into_iter().collect::<Vec<_>>();
        img_info_list.sort_by(|img_a, img_b| img_a.name.cmp(&img_b.name));

        log::info!("Loading {} images for colmap dataset", img_info_list.len());

        let mut views = Vec::new();
        let mut warnings = Vec::new();

        for img_info in img_info_list
            .iter()
            .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
            .take(load_args.max_frames.unwrap_or(usize::MAX))
        {
            let colmap_camera = cam_model_data
                .get(&img_info.camera_id)
                .ok_or_else(|| {
                    FormatError::InvalidFormat(format!(
                        "Image '{}' references camera ID {} which doesn't exist in camera data",
                        img_info.name, img_info.camera_id
                    ))
                })?
                .clone();

            // Create a future to handle loading the image.
            let camera_model = build_camera_model(&colmap_camera);
            let focal = colmap_camera.focal();
            let fovx = camera::focal_to_fov(focal.0, colmap_camera.width as u32, &camera_model);
            let fovy = camera::focal_to_fov(focal.1, colmap_camera.height as u32, &camera_model);
            let center = colmap_camera.principal_point();
            let center_uv =
                center / glam::vec2(colmap_camera.width as f32, colmap_camera.height as f32);

            let Some(path) = find_img(&vfs, &img_info.name) else {
                warnings.push(format!("Skipped '{}': image file not found", img_info.name));
                continue;
            };

            let mask_path = find_mask_path(&vfs, path);

            // Convert w2c to c2w.
            let world_to_cam =
                glam::Affine3A::from_rotation_translation(img_info.quat, img_info.tvec);
            let cam_to_world = world_to_cam.inverse();
            let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

            let camera = Camera::new(translation, quat, fovx, fovy, center_uv, camera_model);

            if !camera.is_valid() {
                warnings.push(format!(
                    "Skipped '{}': camera contains nan or inf values",
                    img_info.name
                ));
                continue;
            }

            let image = LoadImage::new(
                vfs.clone(),
                path.to_path_buf(),
                mask_path.map(|p| p.to_path_buf()),
                load_args.max_resolution,
                load_args.alpha_mode,
            );

            views.push(SceneView { camera, image });
        }

        let (train_views, eval_views) = views.into_iter().enumerate().partition_map(|(i, v)| {
            if let Some(split) = load_args.eval_split_every
                && i % split == 0
            {
                itertools::Either::Right(v)
            } else {
                itertools::Either::Left(v)
            }
        });

        Result::<_, FormatError>::Ok((Dataset::from_views(train_views, eval_views), warnings))
    });

    let load_args = load_args.clone();

    let init = actor.run(move || async move {
        let points_path = vfs_init
            .files_ending_in("points3d.txt")
            .chain(vfs_init.files_ending_in("points3d.bin"))
            .find(|p| p.parent() == Some(points_dir.as_path()))?;
        let is_binary = matches!(
            points_path.extension().and_then(|p| p.to_str()),
            Some("bin")
        );
        // At this point the VFS has said this file exists so just unwrap.
        let mut points_file = vfs_init
            .reader_at_path(points_path)
            .await
            .expect("unreachable");

        let step = load_args.subsample_points.unwrap_or(1) as usize;
        let points_data = colmap_reader::read_points3d(&mut points_file, is_binary, false)
            .await
            .ok()?;

        if points_data.is_empty() {
            return None;
        }

        let positions: Vec<f32> = points_data
            .iter()
            .step_by(step)
            .flat_map(|p| p.xyz.to_array())
            .collect();
        let colors: Vec<f32> = points_data
            .iter()
            .step_by(step)
            .flat_map(|p| {
                let sh = rgb_to_sh(glam::vec3(
                    p.rgb[0] as f32 / 255.0,
                    p.rgb[1] as f32 / 255.0,
                    p.rgb[2] as f32 / 255.0,
                ));
                [sh.x, sh.y, sh.z]
            })
            .collect();

        let n_splats = positions.len() / 3;
        log::info!("Starting from colmap points: {n_splats}");
        let data = SplatData {
            means: positions,
            rotations: None,
            log_scales: None,
            sh_coeffs: Some(colors),
            raw_opacities: None,
        };

        Some(SplatMessage {
            meta: ParseMetadata {
                up_axis: None,
                render_mode: None,
                total_splats: n_splats as u32,
                progress: 1.0,
            },
            data,
        })
    });

    // Wait for both halves.
    let (dataset, init) = tokio::join!(dataset, init);
    let ((dataset, warnings), init_splat) = (dataset?, init);

    Ok(DatasetLoadResult {
        init_splat,
        dataset,
        warnings,
    })
}

fn build_camera_model(colmap_camera: &ColmapCamera) -> CameraModel {
    let p = &colmap_camera.params;
    match colmap_camera.model {
        ColmapCameraModel::SimpleRadialFisheye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[3] as f32,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        }),
        ColmapCameraModel::RadialFisheye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[3] as f32,
            k2: p[4] as f32,
            k3: 0.0,
            k4: 0.0,
        }),
        ColmapCameraModel::OpenCvFishEye => KannalaBrandt4(KannalaBrandt4Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            k3: p[6] as f32,
            k4: p[7] as f32,
        }),
        ColmapCameraModel::FullOpenCV => RadialTangential8(RadialTangential8Params {
            k1: p[4] as f32,
            k2: p[5] as f32,
            k3: p[8] as f32,
            k4: p[9] as f32,
            k5: p[10] as f32,
            k6: p[11] as f32,
            p1: p[6] as f32,
            p2: p[7] as f32,
        }),
        ColmapCameraModel::Pinhole => Pinhole,
        _ => {
            log::warn!(
                "Unsupported camera model: {:?}! Falling back to pinhole camera",
                colmap_camera.model
            );
            Pinhole
        }
    }
}
