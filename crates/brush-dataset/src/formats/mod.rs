use crate::{
    Dataset,
    config::LoadDataseConfig,
    splat_import::{SplatMessage, load_splat_from_ply},
};
use anyhow::Context;
use brush_vfs::{BrushVfs, DynStream};
use burn::backend::wgpu::WgpuDevice;
use path_clean::PathClean;
use std::{
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

pub mod colmap;
pub mod nerfstudio;

pub type DataStream<T> = Pin<Box<dyn DynStream<anyhow::Result<T>>>>;

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &WgpuDevice,
) -> anyhow::Result<(DataStream<SplatMessage>, Dataset)> {
    let data_read = nerfstudio::read_dataset(vfs.clone(), load_args, device).await;

    let data_read = if let Some(data_read) = data_read {
        data_read.context("Failed to load as json format.")?
    } else {
        let stream = colmap::load_dataset(vfs.clone(), load_args, device)
            .await
            .context("Dataset was neither in nerfstudio or COLMAP format.")?;
        stream.context("Failed to load as COLMAP format.")?
    };

    // If there's an initial ply file, override the init stream with that.
    let path: Vec<_> = vfs.files_with_extension("ply").collect();

    let init_stream = if path.len() == 1 {
        let main_path = path.first().expect("unreachable");
        log::info!("Using ply {main_path:?} as initial point cloud.");

        let reader = vfs.reader_at_path(main_path).await?;
        Box::pin(load_splat_from_ply(
            reader,
            load_args.subsample_points,
            device.clone(),
        ))
    } else {
        data_read.0
    };

    Ok((init_stream, data_read.1))
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
