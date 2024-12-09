use crate::{
    brush_vfs::BrushVfs,
    splat_import::{load_splat_from_ply, SplatMessage},
    Dataset, LoadDatasetArgs,
};
use anyhow::Result;
use brush_render::Backend;
use std::{path::Path, pin::Pin};
use tokio_stream::Stream;

pub mod colmap;
pub mod nerfstudio;

// A dynamic stream of datasets
type DataStream<T> = Pin<Box<dyn Stream<Item = Result<T>> + Send + 'static>>;

pub async fn load_dataset<B: Backend>(
    mut vfs: BrushVfs,
    load_args: &LoadDatasetArgs,
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

    // If there's an init.ply definitey override the init stream with that.
    let init_stream = if let Ok(reader) = vfs.open_path(Path::new("init.ply")).await {
        log::info!("Using init.ply as initial point cloud.");
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
