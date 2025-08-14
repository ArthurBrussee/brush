use std::{path::PathBuf, sync::Arc};

use super::FormatError;
use crate::{
    Dataset,
    config::LoadDataseConfig,
    formats::find_mask_path,
    scene::{LoadImage, SceneView},
    splat_import::{ParseMetadata, SplatMessage},
};
use brush_render::{
    camera::{self, Camera},
    gaussian_splats::Splats,
    sh::rgb_to_sh,
};
use brush_vfs::BrushVfs;
use burn::backend::wgpu::WgpuDevice;
use std::collections::HashMap;

fn find_mask_and_img(vfs: &BrushVfs, name: &str) -> Option<(PathBuf, Option<PathBuf>)> {
    // Colmap only specifies an image name, not a full path. We brute force
    // search for the image in the archive.
    //
    // Make sure this path doesn't start with a '/' as the files_ending_in expects
    // things in that format (like a "filename with slashes").
    let name = name.strip_prefix('/').unwrap_or(name);

    let paths: Vec<_> = vfs.files_ending_in(name).collect();

    let mut path_masks = HashMap::new();
    let mut masks = vec![];

    // First pass: collect images & masks.
    for path in paths {
        let mask = find_mask_path(vfs, &path);
        path_masks.insert(path.clone(), mask.clone());
        if let Some(mask_path) = mask {
            masks.push(mask_path);
        }
    }

    // Remove masks from candidates - shouldn't count as an input image.
    for mask in masks {
        path_masks.remove(&mask);
    }

    // Sort and return the first candidate (alphabetically).
    path_masks.into_iter().min_by_key(|kv| kv.0.clone())
}

pub(crate) async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &WgpuDevice,
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

    Some(load_dataset_inner(vfs, load_args, device, cam_path, img_path).await)
}

async fn load_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &WgpuDevice,
    cam_path: PathBuf,
    img_path: PathBuf,
) -> Result<(Option<SplatMessage>, Dataset), FormatError> {
    let is_binary = cam_path.ends_with("cameras.bin");

    let cam_model_data = {
        let mut cam_file = vfs.reader_at_path(&cam_path).await?;
        colmap_reader::read_cameras(&mut cam_file, is_binary).await?
    };

    let img_infos = {
        let img_file = vfs.reader_at_path(&img_path).await?;
        let mut buf_reader = tokio::io::BufReader::new(img_file);
        colmap_reader::read_images(&mut buf_reader, is_binary).await?
    };

    let mut img_info_list = img_infos.into_iter().collect::<Vec<_>>();
    img_info_list.sort_by_key(|key_img| key_img.1.name.clone());

    log::info!("Loading colmap dataset with {} images", img_info_list.len());

    let mut train_views = vec![];
    let mut eval_views = vec![];

    for (i, (_img_id, img_info)) in img_info_list
        .into_iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .step_by(load_args.subsample_frames.unwrap_or(1) as usize)
        .enumerate()
    {
        let cam_data = cam_model_data
            .get(&img_info.camera_id)
            .ok_or_else(|| {
                FormatError::InvalidFormat(format!(
                    "Image '{}' with ID {} references camera ID {} which doesn't exist in camera data",
                    img_info.name, _img_id, img_info.camera_id
                ))
            })?
            .clone();
        let vfs = vfs.clone();

        // Create a future to handle loading the image.
        let focal = cam_data.focal();

        let fovx = camera::focal_to_fov(focal.0, cam_data.width as u32);
        let fovy = camera::focal_to_fov(focal.1, cam_data.height as u32);

        let center = cam_data.principal_point();
        let center_uv = center / glam::vec2(cam_data.width as f32, cam_data.height as f32);

        // If image isn't found, just ignore it. We can still train on the remaining images.
        let Some((path, mask_path)) = find_mask_and_img(&vfs, &img_info.name) else {
            log::warn!("Image not found: {}", img_info.name);
            continue;
        };

        // Convert w2c to c2w.
        let world_to_cam = glam::Affine3A::from_rotation_translation(img_info.quat, img_info.tvec);
        let cam_to_world = world_to_cam.inverse();
        let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

        let camera = Camera::new(translation, quat, fovx, fovy, center_uv);

        log::info!("Loaded COLMAP image at path {path:?}");

        let load_img =
            LoadImage::new(vfs.clone(), &path, mask_path, load_args.max_resolution).await?;

        let view = SceneView {
            camera,
            image: load_img,
        };

        if let Some(eval_period) = load_args.eval_split_every {
            if i % eval_period == 0 {
                eval_views.push(view);
            } else {
                train_views.push(view);
            }
        } else {
            train_views.push(view);
        }
    }

    let init = try_load_init(vfs, device, load_args).await;
    let dataset = Dataset::from_views(train_views, eval_views);
    Ok((init, dataset))
}

async fn try_load_init(
    vfs: Arc<BrushVfs>,
    device: &WgpuDevice,
    load_args: &LoadDataseConfig,
) -> Option<SplatMessage> {
    let points_path = { vfs.files_ending_in("points3d.txt").next() }
        .or_else(|| vfs.files_ending_in("points3d.bin").next())?;

    let is_binary = matches!(
        points_path.extension().and_then(|p| p.to_str()),
        Some("bin")
    );

    // Extract COLMAP sfm points.
    let points_data = {
        // At this point the VFS has said this file exists so just unwrap.
        let mut points_file = vfs.reader_at_path(&points_path).await.expect("unreachable");
        colmap_reader::read_points3d(&mut points_file, is_binary).await
    };

    let Ok(points_data) = points_data else {
        return None;
    };

    if points_data.is_empty() {
        return None;
    }

    // Ignore empty points data.
    log::info!("Starting from colmap points {}", points_data.len());

    // The ply importer handles subsampling normally. Here we just
    // do it manually.
    let step = load_args.subsample_points.unwrap_or(1) as usize;

    let positions: Vec<f32> = points_data
        .values()
        .step_by(step)
        .flat_map(|p| p.xyz.to_array())
        .collect();
    let colors: Vec<f32> = points_data
        .values()
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
    let init_splat = Splats::from_raw(positions, None, None, Some(colors), None, device);
    Some(SplatMessage {
        meta: ParseMetadata {
            up_axis: None,
            total_splats: init_splat.num_splats(),
            frame_count: 1,
            current_frame: 0,
            progress: 1.0,
        },
        splats: init_splat,
    })
}
