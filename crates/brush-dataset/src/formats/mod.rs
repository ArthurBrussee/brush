use crate::{
    Dataset,
    config::LoadDataseConfig,
    splat_import::{SplatMessage, load_splat_from_ply},
};
use brush_vfs::BrushVfs;
use burn::backend::wgpu::WgpuDevice;
use path_clean::PathClean;
use serde_ply::DeserializeError;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

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
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("Failed to load format.")]
    FormatError(#[from] FormatError),

    #[error("Failed to load initial point cloud.")]
    InitialPointCloudError(#[from] serde_ply::DeserializeError),

    #[error("Format not recognized: Only colmap and nerfstudio json are supported.")]
    FormatNotSupported,
}

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &WgpuDevice,
) -> Result<(Option<SplatMessage>, Dataset), DatasetError> {
    let nerfstudio_fmt = nerfstudio::read_dataset(vfs.clone(), load_args, device).await;

    let format = if let Some(fmt) = nerfstudio_fmt {
        fmt?
    } else {
        let Some(stream) = colmap::load_dataset(vfs.clone(), load_args, device).await else {
            return Err(DatasetError::FormatNotSupported);
        };
        stream?
    };

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
            .map_err(serde_ply::DeserializeError)?;
        Some(load_splat_from_ply(reader, load_args.subsample_points, device.clone()).await?)
    } else {
        format.0
    };

    Ok((init_splat, format.1))
}

fn find_mask_path(vfs: &BrushVfs, path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?.clean();
    let file_stem = path.file_stem()?.to_str()?;
    let masks_dir = parent.parent()?.join("masks").clean();
    for candidate in vfs.files_with_stem(file_stem) {
        let Some(file_parent) = candidate.parent() else {
            continue;
        };
        if file_parent == masks_dir {
            return Some(candidate);
        }
    }
    None
}
