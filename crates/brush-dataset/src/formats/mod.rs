use crate::{
    Dataset,
    config::LoadDataseConfig,
    splat_import::{SplatMessage, load_splat_from_ply},
};
use anyhow::Context;
use brush_vfs::{BrushVfs, WasmNotSend};
use burn::prelude::Backend;
use path_clean::PathClean;
use std::{
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};
use tokio_stream::Stream;

pub mod colmap;
pub mod nerfstudio;

pub trait DynStream<Item>: Stream<Item = Item> + WasmNotSend {}
impl<Item, T: Stream<Item = Item> + WasmNotSend> DynStream<Item> for T {}
pub type DataStream<T> = Pin<Box<dyn DynStream<anyhow::Result<T>> + 'static>>;

pub async fn load_dataset<B: Backend>(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
    device: &B::Device,
) -> anyhow::Result<(DataStream<SplatMessage<B>>, Dataset)> {
    let data_read = nerfstudio::read_dataset(vfs.clone(), load_args, device).await;

    let data_read = if let Some(data_read) = data_read {
        data_read.context("Failed to load as json format.")?
    } else {
        let stream = colmap::load_dataset::<B>(vfs.clone(), load_args, device)
            .await
            .context("Dataset was neither in nerfstudio or COLMAP format.")?;
        stream.context("Failed to load as COLMAP format.")?
    };

    // If there's an initial ply file, override the init stream with that.
    let path: Vec<_> = vfs
        .file_names()
        .filter(|x| x.extension().is_some_and(|ext| ext == "ply"))
        .collect();

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
    let masked_name = format!("{file_stem}_mask");
    let masks_dir = parent.parent()?.join("masks").clean();

    vfs.file_names().find(|file| {
        let Some(file_parent) = file.parent() else {
            return false;
        };

        let Some(stem) = file.file_stem().and_then(|p| p.to_str()) else {
            return false;
        };

        file_parent == parent && stem == masked_name
            || file_parent == masks_dir && stem == file_stem
    })
}
