use std::{
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::DataStream;
use crate::{
    brush_vfs::{normalized_path, BrushVfs},
    splat_import::SplatMessage,
    stream_fut_parallel, Dataset, LoadDataseConfig,
};
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use brush_render::{
    camera::{self, Camera},
    gaussian_splats::Splats,
    render::rgb_to_sh,
    Backend,
};
use brush_train::scene::SceneView;
use glam::Vec3;
use tokio::io::AsyncReadExt;
use tokio_stream::StreamExt;

fn find_base_path(archive: &BrushVfs, search_path: &str) -> Option<PathBuf> {
    for file in archive.file_names() {
        let path = normalized_path(Path::new(file));
        if path.ends_with(search_path) {
            return path
                .ancestors()
                .nth(Path::new(search_path).components().count())
                .map(|x| x.to_owned());
        }
    }
    None
}

async fn read_views(
    archive: BrushVfs,
    load_args: &LoadDataseConfig,
) -> Result<Vec<impl Future<Output = Result<SceneView>>>> {
    log::info!("Loading colmap dataset");
    let mut archive = archive;

    let (is_binary, base_path) =
        if let Some(path) = find_base_path(&archive, "sparse/0/cameras.bin") {
            (true, path)
        } else if let Some(path) = find_base_path(&archive, "sparse/0/cameras.txt") {
            (false, path)
        } else {
            anyhow::bail!("No COLMAP data found (either text or binary.)")
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
        let mut cam_file = archive.open_path(&cam_path).await?;
        colmap_reader::read_cameras(&mut cam_file, is_binary).await?
    };

    let img_infos = {
        let img_file = archive.open_path(&img_path).await?;
        let mut buf_reader = tokio::io::BufReader::new(img_file);
        colmap_reader::read_images(&mut buf_reader, is_binary).await?
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

                let mut img_bytes = vec![];
                archive
                    .open_path(&img_path)
                    .await?
                    .read_to_end(&mut img_bytes)
                    .await?;
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
                    name: img_path.to_string_lossy().to_string(),
                    camera,
                    image: Arc::new(img),
                };
                Ok(view)
            }
        })
        .collect();

    Ok(handles)
}

pub(crate) async fn load_dataset<B: Backend>(
    mut archive: BrushVfs,
    load_args: &LoadDataseConfig,
    device: &B::Device,
) -> Result<(DataStream<SplatMessage<B>>, DataStream<Dataset>)> {
    let mut handles = read_views(archive.clone(), load_args).await?;

    if let Some(subsample) = load_args.subsample_frames {
        handles = handles.into_iter().step_by(subsample as usize).collect();
    }

    let mut train_views = vec![];
    let mut eval_views = vec![];

    let load_args = load_args.clone();
    let device = device.clone();

    let mut i = 0;
    let stream = stream_fut_parallel(handles).map(move |view| {
        if let Ok(view) = view {
            // I cannot wait for let chains.
            if let Some(eval_period) = load_args.eval_split_every {
                if i % eval_period == 0 {
                    log::info!("Adding split eval view");
                    eval_views.push(view);
                } else {
                    train_views.push(view);
                }
            } else {
                train_views.push(view);
            }
        }

        i += 1;
        Ok(Dataset::from_views(train_views.clone(), eval_views.clone()))
    });

    let init_stream = try_fn_stream(|emitter| async move {
        let (is_binary, base_path) =
            if let Some(path) = find_base_path(&archive, "sparse/0/cameras.bin") {
                (true, path)
            } else if let Some(path) = find_base_path(&archive, "sparse/0/cameras.txt") {
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
            let mut points_file = archive.open_path(&points_path).await?;
            colmap_reader::read_points3d(&mut points_file, is_binary).await
        };

        // Ignore empty points data.
        if let Ok(points_data) = points_data {
            if !points_data.is_empty() {
                log::info!("Starting from colmap points {}", points_data.len());

                let mut positions: Vec<Vec3> = points_data.values().map(|p| p.xyz).collect();
                let mut colors: Vec<f32> = points_data
                    .values()
                    .flat_map(|p| {
                        [
                            rgb_to_sh(p.rgb[0] as f32 / 255.0),
                            rgb_to_sh(p.rgb[1] as f32 / 255.0),
                            rgb_to_sh(p.rgb[2] as f32 / 255.0),
                        ]
                    })
                    .collect();

                // Other dataloaders handle subsampling in the ply import. Here just
                // do it manually, maybe nice to unify at some point.
                if let Some(subsample) = load_args.subsample_points {
                    positions = positions.into_iter().step_by(subsample as usize).collect();
                    colors = colors.into_iter().step_by(subsample as usize * 3).collect();
                }

                let init_splat =
                    Splats::from_raw(&positions, None, None, Some(&colors), None, &device);
                emitter
                    .emit(SplatMessage {
                        meta: crate::splat_import::SplatMetadata {
                            up_axis: Vec3::Y,
                            total_splats: init_splat.num_splats(),
                            frame_count: 1,
                            current_frame: 0,
                        },
                        splats: init_splat,
                    })
                    .await;
            }
        }

        Ok(())
    });

    Ok((Box::pin(init_stream), Box::pin(stream)))
}
