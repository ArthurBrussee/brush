use std::{future::Future, sync::Arc};

use super::{DataStream, DatasetZip, LoadDatasetArgs, LoadInitArgs};
use crate::{stream_fut_parallel, Dataset};
use anyhow::{Context, Result};
use brush_render::{
    camera::{self, Camera},
    gaussian_splats::Splats,
    Backend,
};
use brush_train::scene::SceneView;
use glam::Vec3;
use tokio_stream::StreamExt;

fn read_views(
    mut archive: DatasetZip,
    load_args: &LoadDatasetArgs,
) -> Result<Vec<impl Future<Output = Result<SceneView>>>> {
    log::info!("Loading colmap dataset");

    let (is_binary, base_path) = if let Some(path) = archive.find_base_path("sparse/0/cameras.bin")
    {
        (true, path)
    } else if let Some(path) = archive.find_base_path("sparse/0/cameras.txt") {
        (false, path)
    } else {
        anyhow::bail!("No COLMAP data found (either text or binary.")
    };

    let (cam_path, img_path) = if is_binary {
        (
            base_path.join("sparse/0/cameras.bin"),
            base_path.join("sparse/0/images.bin"),
        )
    } else {
        (
            base_path.join("sparse/0/cameras.txt"),
            base_path.join("sparse/0/images.txt"),
        )
    };

    let cam_model_data = {
        let mut cam_file = archive.file_at_path(&cam_path)?;
        colmap_reader::read_cameras(&mut cam_file, is_binary)?
    };

    let img_infos = {
        let img_file = archive.file_at_path(&img_path)?;
        let mut buf_reader = std::io::BufReader::new(img_file);
        colmap_reader::read_images(&mut buf_reader, is_binary)?
    };

    let mut img_info_list = img_infos.into_iter().collect::<Vec<_>>();

    log::info!("Colmap dataset contains {} images", img_info_list.len());

    // Sort by image ID. Not entirely sure whether it's better to
    // load things in COLMAP order or sorted by file name. Either way, at least
    // it is consistent
    img_info_list.sort_by_key(|key_img| key_img.0);

    let handles = img_info_list
        .into_iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .map(move |(_, img_info)| {
            let cam_data = cam_model_data[&img_info.camera_id].clone();
            let load_args = load_args.clone();
            let base_path = base_path.clone();
            let mut archive = archive.clone();

            // Create a future to handle loading the image.
            async move {
                let focal = cam_data.focal();

                let fovx = camera::focal_to_fov(focal.0, cam_data.width as u32);
                let fovy = camera::focal_to_fov(focal.1, cam_data.height as u32);

                let center = cam_data.principal_point();
                let center_uv = center / glam::vec2(cam_data.width as f32, cam_data.height as f32);

                let img_path = base_path.join(format!("images/{}", img_info.name));

                let img_bytes = archive.read_bytes_at_path(&img_path)?;
                let mut img = image::load_from_memory(&img_bytes)?;

                if let Some(max) = load_args.max_resolution {
                    img = crate::clamp_img_to_max_size(img, max);
                }

                // Convert w2c to c2w.
                let world_to_cam =
                    glam::Affine3A::from_rotation_translation(img_info.quat, img_info.tvec);
                let cam_to_world = world_to_cam.inverse();
                let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

                let camera = Camera::new(translation, quat, fovx, fovy, center_uv);

                let view = SceneView {
                    name: img_path.to_str().context("Invalid file name")?.to_owned(),
                    camera,
                    image: Arc::new(img),
                };
                Ok(view)
            }
        })
        .collect();

    Ok(handles)
}

pub(crate) fn load_dataset(
    archive: DatasetZip,
    load_args: &LoadDatasetArgs,
) -> Result<DataStream<Dataset>> {
    let handles = read_views(archive, load_args)?;

    let mut train_views = vec![];
    let mut eval_views = vec![];

    let load_args = load_args.clone();
    let mut i = 0;
    let stream = stream_fut_parallel(handles).map(move |view| {
        // I cannot wait for let chains.
        if let Some(eval_period) = load_args.eval_split_every {
            if i % eval_period == 0 {
                eval_views.push(view?);
            } else {
                train_views.push(view?);
            }
        } else {
            train_views.push(view?);
        }

        i += 1;
        Ok(Dataset::from_views(train_views.clone(), eval_views.clone()))
    });

    Ok(Box::pin(stream))
}

pub(crate) fn load_initial_splat<B: Backend>(
    mut archive: DatasetZip,
    device: &B::Device,
    load_args: &LoadInitArgs,
) -> Result<Splats<B>> {
    let (is_binary, base_path) = if let Some(path) = archive.find_base_path("sparse/0/cameras.bin")
    {
        (true, path)
    } else if let Some(path) = archive.find_base_path("sparse/0/cameras.txt") {
        (false, path)
    } else {
        anyhow::bail!("No COLMAP data found (either text or binary.")
    };

    let points_path = if is_binary {
        base_path.join("sparse/0/points3D.bin")
    } else {
        base_path.join("sparse/0/points3D.txt")
    };

    // Extract COLMAP sfm points.
    let points_data = {
        let mut points_file = archive.file_at_path(&points_path)?;
        colmap_reader::read_points3d(&mut points_file, is_binary)?
    };

    let positions = points_data.values().map(|p| p.xyz).collect();
    let colors = points_data
        .values()
        .map(|p| Vec3::new(p.rgb[0] as f32, p.rgb[1] as f32, p.rgb[2] as f32) / 255.0)
        .collect();

    Ok(Splats::from_point_cloud(
        positions,
        colors,
        load_args.sh_degree,
        device,
    ))
}
