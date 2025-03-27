use crate::{
    Dataset, LoadDataseConfig, WasmNotSend,
    brush_vfs::BrushVfs,
    splat_import::{SplatMessage, load_splat_from_ply},
};
use burn::prelude::Backend;
use image::DynamicImage;
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
    let mut err_context = anyhow::anyhow!("Attempting to load dataset.");

    let stream = nerfstudio::read_dataset(vfs.clone(), load_args, device).await;

    let stream = match stream {
        Ok(s) => Ok(s),
        Err(e) => {
            err_context = err_context
                .context(e)
                .context("Failed to load as json format.");

            colmap::load_dataset::<B>(vfs.clone(), load_args, device).await
        }
    };

    let init_stream_dataset = match stream {
        Ok(stream) => stream,
        Err(e) => {
            err_context = err_context
                .context(e)
                .context("Failed to load as COLMAP format.");

            Err(err_context.context("Failed to load dataset as any format."))?
        }
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
        init_stream_dataset.0
    };

    Ok((init_stream, init_stream_dataset.1))
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

pub fn clamp_img_to_max_size(image: DynamicImage, max_size: u32) -> DynamicImage {
    if image.width() <= max_size && image.height() <= max_size {
        return image;
    }
    image.resize(max_size, max_size, image::imageops::FilterType::Lanczos3)
}
