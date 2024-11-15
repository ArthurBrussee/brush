use crate::{
    splat_import::load_splat_from_ply, zip::DatasetZip, Dataset, LoadDatasetArgs, LoadInitArgs,
};
use anyhow::Result;
use async_fn_stream::fn_stream;
use brush_render::{gaussian_splats::Splats, Backend};
use std::{io::Cursor, path::Path, pin::Pin};
use tokio_stream::Stream;

pub mod colmap;
pub mod nerf_synthetic;

// A dynamic stream of datasets
type DataStream<T> = Pin<Box<dyn Stream<Item = Result<T>> + Send + 'static>>;

pub fn load_dataset(
    archive: DatasetZip,
    load_args: &LoadDatasetArgs,
) -> Result<DataStream<Dataset>> {
    nerf_synthetic::read_dataset(archive.clone(), load_args)
        .or_else(|_| colmap::load_dataset(archive.clone(), load_args))
        .map_err(|_| {
            anyhow::anyhow!(
                "Couldn't parse dataset as any format. Only some formats are supported."
            )
        })
}

fn read_init_ply<B: Backend>(
    mut archive: DatasetZip,
    device: &B::Device,
) -> Result<DataStream<Splats<B>>> {
    let data = archive.read_bytes_at_path(Path::new("init.ply"))?;
    let splat_stream = load_splat_from_ply(Cursor::new(data), device.clone());
    Ok(Box::pin(splat_stream))
}

pub fn load_initial_splat<B: Backend>(
    archive: DatasetZip,
    device: &B::Device,
    load_args: &LoadInitArgs,
) -> Option<DataStream<Splats<B>>> {
    // If there's an init.ply definitey use that. Nb:
    // this ignores the specified number of SH channels atm.
    if let Ok(stream) = read_init_ply(archive.clone(), device) {
        return Some(stream);
    }

    let start_splats = colmap::load_initial_splat(archive.clone(), device, load_args);
    if let Ok(splats) = start_splats {
        // Return a stream with just the init splat.
        let stream = fn_stream(|emitter| async move {
            emitter.emit(Ok(splats)).await;
        });
        return Some(Box::pin(stream));
    }

    None
}
