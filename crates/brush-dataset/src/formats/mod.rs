use crate::{
    brush_vfs::BrushVfs,
    splat_import::{load_splat_from_ply, SplatMessage},
    Dataset, LoadDataseConfig, WasmNotSend,
};
use brush_render::Backend;
use image::DynamicImage;
use path_clean::PathClean;
use std::{
    path::{Path, PathBuf},
    pin::Pin,
};
use tokio::io::AsyncReadExt;
use tokio_stream::Stream;

pub mod colmap;
pub mod nerfstudio;

pub trait DynStream<Item>: Stream<Item = Item> + WasmNotSend {}
impl<Item, T: Stream<Item = Item> + WasmNotSend> DynStream<Item> for T {}
pub type DataStream<T> = Pin<Box<dyn DynStream<anyhow::Result<T>> + 'static>>;

pub async fn load_dataset<B: Backend>(
    mut vfs: BrushVfs,
    load_args: &LoadDataseConfig,
    device: &B::Device,
) -> anyhow::Result<(DataStream<SplatMessage<B>>, DataStream<Dataset>)> {
    let stream = nerfstudio::read_dataset(vfs.clone(), load_args, device).await;

    let stream = match stream {
        Ok(s) => Ok(s),
        Err(_) => colmap::load_dataset::<B>(vfs.clone(), load_args, device).await,
    };

    let stream = match stream {
        Ok(stream) => stream,
        Err(e) => anyhow::bail!(
            "Couldn't parse dataset as any format. Only some formats are supported. {e}"
        ),
    };

    // If there's an initial ply file, override the init stream with that.
    let path: Vec<_> = vfs
        .file_names()
        .filter(|x| x.extension().is_some_and(|ext| ext == "ply"))
        .collect();

    let init_stream = if path.len() == 1 {
        let main_path = path.first().expect("unreachable");
        log::info!("Using ply {main_path:?} as initial point cloud.");

        let reader = vfs.open_path(main_path).await?;
        Box::pin(load_splat_from_ply(
            reader,
            load_args.subsample_points,
            device.clone(),
        ))
    } else {
        stream.0
    };

    Ok((init_stream, stream.1))
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

pub(crate) fn clamp_img_to_max_size(image: DynamicImage, max_size: u32) -> DynamicImage {
    if image.width() <= max_size && image.height() <= max_size {
        return image;
    }

    let aspect_ratio = image.width() as f32 / image.height() as f32;
    let (new_width, new_height) = if image.width() > image.height() {
        (max_size, (max_size as f32 / aspect_ratio) as u32)
    } else {
        ((max_size as f32 * aspect_ratio) as u32, max_size)
    };
    image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

pub(crate) async fn load_image(
    vfs: &mut BrushVfs,
    img_path: &Path,
    mask_path: Option<&Path>,
) -> anyhow::Result<DynamicImage> {
    log::info!("Loading image at {img_path:?}, with a mask {mask_path:?}");

    let mut img_bytes = vec![];

    vfs.open_path(img_path)
        .await?
        .read_to_end(&mut img_bytes)
        .await?;
    let mut img = image::load_from_memory(&img_bytes)?;

    // Copy over mask
    if let Some(mask_path) = mask_path {
        // if img.color().has_alpha() {
        //     anyhow::bail!("Image has both an alpha channel and mask specified, bailing.");
        // }

        let mut mask_bytes = vec![];

        vfs.open_path(mask_path)
            .await?
            .read_to_end(&mut mask_bytes)
            .await?;

        let mask_img = image::load_from_memory(&mask_bytes)?;

        let mut img_masked = img.to_rgba8();

        if mask_img.color().has_alpha() {
            let mask_img = mask_img.to_rgba8();
            for (buf, mask) in img_masked.pixels_mut().zip(mask_img.pixels()) {
                buf[3] = mask[0];
            }
        } else {
            let mask_img = mask_img.grayscale().to_rgb8();
            for (buf, mask) in img_masked.pixels_mut().zip(mask_img.pixels()) {
                buf[3] = mask[0];
            }
        }

        img = img_masked.into();
    }

    Ok(img)
}
