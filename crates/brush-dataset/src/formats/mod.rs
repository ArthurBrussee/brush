use crate::{Dataset, config::LoadDataseConfig};
use brush_serde::{DeserializeError, SplatMessage, load_splat_from_ply};
use brush_vfs::BrushVfs;
use burn::backend::wgpu::WgpuDevice;
use image::ImageError;
use std::{path::Path, sync::Arc};

pub mod colmap;
pub mod nerfstudio;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("IO error while loading dataset.")]
    Io(#[from] std::io::Error),

    #[error("Error decoding JSON file.")]
    Json(#[from] serde_json::Error),

    #[error("Error decoding camera parameters: {0}")]
    InvalidCamera(String),

    #[error("Error when decoding format: {0}")]
    InvalidFormat(String),

    #[error("Error loading splat data: {0}")]
    PlyError(#[from] DeserializeError),

    #[error("Error loading image in data: {0}")]
    ImageError(#[from] ImageError),
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("Failed to load format.")]
    FormatError(#[from] FormatError),

    #[error("Failed to load initial point cloud.")]
    InitialPointCloudError(#[from] DeserializeError),

    #[error("Format not recognized: Only colmap and nerfstudio json are supported.")]
    FormatNotSupported,
}

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &WgpuDevice,
) -> Result<(Option<SplatMessage>, Dataset), DatasetError> {
    let mut dataset = colmap::load_dataset(vfs.clone(), load_args, device).await;

    if dataset.is_none() {
        dataset = nerfstudio::read_dataset(vfs.clone(), load_args, device).await;
    }

    let Some(dataset) = dataset else {
        return Err(DatasetError::FormatNotSupported);
    };

    let (data_splat_init, dataset) = dataset?;

    // If there's an initial ply file, override the init stream with that.
    let ply_paths: Vec<_> = vfs.files_with_extension("ply").collect();

    let main_ply_path = if ply_paths.len() == 1 {
        Some(ply_paths.first().expect("unreachable"))
    } else {
        ply_paths.iter().find(|p| {
            p.file_name()
                .and_then(|p| p.to_str())
                .is_some_and(|p| p == "init.ply")
        })
    };

    let init_splat = if let Some(main_path) = main_ply_path {
        log::info!("Using ply {main_path:?} as initial point cloud.");

        let reader = vfs
            .reader_at_path(main_path)
            .await
            .map_err(DeserializeError)?;
        Some(load_splat_from_ply(reader, load_args.subsample_points, device.clone()).await?)
    } else {
        data_splat_init
    };

    Ok((init_splat, dataset))
}

fn find_mask_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Option<&'a Path> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");
    let mut search_mask = search_stem.to_owned();
    search_mask.push(".mask");
    let search_mask = &search_mask;

    vfs.iter_files().find(|candidate| {
        // For the target, we don't care about its actual extension. Lets see if either the name or stem matches.
        let Some(stem) = path.file_stem() else {
            return false;
        };

        // We have the name of the file a la img.png, and the stem a la img.
        // We now want to accept any of img.png.*, img.*, img.mask.*.
        if stem.eq_ignore_ascii_case(search_name)
            || stem.eq_ignore_ascii_case(search_stem)
            || stem.eq_ignore_ascii_case(search_mask)
        {
            // Find "masks" directory in candidate path
            let masks_idx = candidate
                .components()
                .position(|c| c.as_os_str().eq_ignore_ascii_case("masks"));

            // Check if the image directory path ends with the directory subpath after "masks/"
            // e.g., masks/foo/bar/bla.png should match images/foo/bar/bla.jpeg
            masks_idx.is_some_and(|idx| {
                let candidate_components: Vec<_> = candidate.components().collect();

                // Get directory components only (excluding filename)
                let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
                let mask_dir_subpath =
                    &candidate_components[idx + 1..candidate_components.len() - 1];
                path_dir_components.ends_with(mask_dir_subpath)
            })
        } else {
            false
        }
    })
}
