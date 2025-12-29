use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::FormatError;
use crate::{
    Dataset,
    config::LoadDataseConfig,
    formats::find_mask_path,
    scene::{LoadImage, SceneView},
};
use brush_render::{
    camera::{self, Camera},
    sh::rgb_to_sh,
};
use brush_serde::{ParseMetadata, SplatData, SplatMessage};
use brush_vfs::BrushVfs;
use itertools::Itertools;
use tokio_with_wasm::alias as tokio_wasm;

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

pub(crate) async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
) -> Option<Result<(Option<SplatMessage>, Dataset), FormatError>> {
    log::info!("Loading colmap dataset");

    let (cam_path, img_path) = if let Some(path) = vfs.files_ending_in("cameras.bin").next() {
        let path = path.parent().expect("unreachable");
        (path.join("cameras.bin"), path.join("images.bin"))
    } else if let Some(path) = vfs.files_ending_in("cameras.txt").next() {
        let path = path.parent().expect("unreachable");
        (path.join("cameras.txt"), path.join("images.txt"))
    } else {
        return None;
    };
    Some(load_dataset_inner(vfs, load_args, cam_path, img_path).await)
}

async fn load_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    cam_path: PathBuf,
    img_path: PathBuf,
) -> Result<(Option<SplatMessage>, Dataset), FormatError> {
    let is_binary = cam_path.ends_with("cameras.bin");

    log::info!("Parsing colmap camera info");

    let load_args = load_args.clone();
    let vfs = vfs.clone();

    let vfs_init = vfs.clone();

    // Spawn three tasks
    let dataset = tokio_wasm::spawn(async move {
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

        let views: Vec<_> = img_info_list
            .iter()
            .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
            .take(load_args.max_frames.unwrap_or(usize::MAX))
            .filter_map(|img_info| {
                let cam_data = cam_model_data
                    .get(&img_info.camera_id)
                    .ok_or_else(|| {
                        FormatError::InvalidFormat(format!(
                            "Image '{}' references camera ID {} which doesn't exist in camera data",
                            img_info.name, img_info.camera_id
                        ))
                    })
                    .unwrap()
                    .clone();

                // Create a future to handle loading the image.
                let focal = cam_data.focal();
                let fovx = camera::focal_to_fov(focal.0, cam_data.width as u32);
                let fovy = camera::focal_to_fov(focal.1, cam_data.height as u32);
                let center = cam_data.principal_point();
                let center_uv = center / glam::vec2(cam_data.width as f32, cam_data.height as f32);

                let Some(path) = find_img(&vfs, &img_info.name) else {
                    log::warn!("Image not found: {}", img_info.name);
                    return None;
                };

                let mask_path = find_mask_path(&vfs, path);

                // Convert w2c to c2w.
                let world_to_cam =
                    glam::Affine3A::from_rotation_translation(img_info.quat, img_info.tvec);
                let cam_to_world = world_to_cam.inverse();
                let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

                let camera = Camera::new(translation, quat, fovx, fovy, center_uv);
                let image = LoadImage::new(
                    vfs.clone(),
                    path.to_path_buf(),
                    mask_path.map(|p| p.to_path_buf()),
                    load_args.max_resolution,
                    load_args.alpha_mode,
                );

                Some(SceneView { camera, image })
            })
            .collect();

        let (train_views, eval_views) = views.into_iter().enumerate().partition_map(|(i, v)| {
            if let Some(split) = load_args.eval_split_every
                && i % split == 0
            {
                itertools::Either::Right(v)
            } else {
                itertools::Either::Left(v)
            }
        });

        Result::<_, FormatError>::Ok(Dataset::from_views(train_views, eval_views))
    });

    let load_args = load_args.clone();

    let init = tokio_wasm::spawn(async move {
        let points_path = { vfs_init.files_ending_in("points3d.txt").next() }
            .or_else(|| vfs_init.files_ending_in("points3d.bin").next())?;
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
                frame_count: 1,
                current_frame: 0,
                progress: 1.0,
            },
            data,
        })
    });

    // Wait for all tasks and get results
    let (dataset, init) = tokio::join!(dataset, init);
    let (dataset, init) = (dataset.expect("Join failed")?, init.expect("Join failed"));

    Ok((init, dataset))
}
