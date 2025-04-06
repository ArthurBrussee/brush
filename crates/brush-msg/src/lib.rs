pub mod config;

use std::str::FromStr;

use brush_dataset::Dataset;
use brush_render::{MainBackend, gaussian_splats::Splats};
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
};
use glam::Vec3;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use web_time::Duration;

use anyhow::anyhow;

use brush_dataset::WasmNotSend;
use brush_dataset::brush_vfs::{BrushVfs, PathReader};
use tokio::io::{AsyncRead, AsyncReadExt, BufReader};
use tokio_stream::StreamExt;
use tokio_util::io::StreamReader;

#[derive(Clone)]
pub struct RefineStats {
    pub num_added: u32,
    pub num_pruned: u32,
}

#[derive(Clone)]
pub struct TrainStepStats<B: Backend> {
    pub pred_image: Tensor<B, 3>,

    pub num_intersections: Tensor<B, 1, Int>,
    pub num_visible: Tensor<B, 1, Int>,
    pub loss: Tensor<B, 1>,

    pub lr_mean: f64,
    pub lr_rotation: f64,
    pub lr_scale: f64,
    pub lr_coeffs: f64,
    pub lr_opac: f64,
}

pub enum ProcessMessage {
    NewSource,
    StartLoading {
        training: bool,
    },
    /// Loaded a splat from a ply file.
    ///
    /// Nb: This includes all the intermediately loaded splats.
    /// Nb: Animated splats will have the 'frame' number set.
    ViewSplats {
        up_axis: Option<Vec3>,
        splats: Box<Splats<MainBackend>>,
        frame: u32,
        total_frames: u32,
    },
    /// Loaded a bunch of viewpoints to train on.
    Dataset {
        dataset: Dataset,
    },
    /// Splat, or dataset and initial splat, are done loading.
    #[allow(unused)]
    DoneLoading {
        training: bool,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    TrainStep {
        splats: Box<Splats<MainBackend>>,
        stats: Box<TrainStepStats<MainBackend>>,
        iter: u32,
        total_elapsed: Duration,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    RefineStep {
        stats: Box<RefineStats>,
        cur_splat_count: u32,
        iter: u32,
    },
    /// Eval was run successfully with these results.
    #[allow(unused)]
    EvalResult {
        iter: u32,
        avg_psnr: f32,
        avg_ssim: f32,
    },
}

#[derive(Clone, Debug)]
pub enum DataSource {
    PickFile,
    PickDirectory,
    Url(String),
    Path(String),
}

// Implement FromStr to allow Clap to parse string arguments into DataSource
impl FromStr for DataSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pick-file" => Ok(Self::PickFile),
            "pick-directory" | "dir" => Ok(Self::PickDirectory),
            s if s.starts_with("http://") || s.starts_with("https://") => {
                Ok(Self::Url(s.to_owned()))
            }
            s if std::fs::exists(s).is_ok() => Ok(Self::Path(s.to_owned())),
            s => Err(format!("Invalid data source. Can't find {s}")),
        }
    }
}

async fn read_at_most<R: AsyncRead + Unpin>(
    reader: &mut R,
    limit: usize,
) -> std::io::Result<Vec<u8>> {
    let mut buffer = vec![0; limit];
    let bytes_read = reader.read(&mut buffer).await?;
    buffer.truncate(bytes_read);
    Ok(buffer)
}

impl DataSource {
    async fn vfs_from_reader(
        reader: impl AsyncRead + WasmNotSend + Unpin + 'static,
    ) -> anyhow::Result<BrushVfs> {
        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let mut data = BufReader::new(reader);
        let peek = read_at_most(&mut data, 64).await?;
        let reader = std::io::Cursor::new(peek.clone()).chain(data);

        if peek.as_slice().starts_with(b"ply") {
            let mut path_reader = PathReader::default();
            path_reader.add(Path::new("input.ply"), reader);
            Ok(BrushVfs::from_paths(path_reader))
        } else if peek.starts_with(b"PK") {
            BrushVfs::from_zip_reader(reader)
                .await
                .map_err(|e| anyhow::anyhow!(e))
        } else if peek.starts_with(b"<!DOCTYPE html>") {
            anyhow::bail!("Failed to download data.")
        } else if let Some(path_bytes) = peek.strip_prefix(b"BRUSH_PATH") {
            let string = String::from_utf8(path_bytes.to_vec())?;
            let path = Path::new(&string);
            BrushVfs::from_directory(path).await
        } else {
            anyhow::bail!("only zip and ply files are supported.")
        }
    }

    pub async fn into_vfs(self) -> anyhow::Result<BrushVfs> {
        match self {
            Self::PickFile => {
                let picked = rrfd::pick_file().await.map_err(|e| anyhow!(e))?;
                let data = picked.read().await;
                let reader = Cursor::new(data);
                Self::vfs_from_reader(reader).await
            }
            Self::PickDirectory => {
                let picked = rrfd::pick_directory().await.map_err(|e| anyhow!(e))?;
                BrushVfs::from_directory(&picked).await
            }
            Self::Url(url) => {
                let mut url = url.clone();

                url = url.replace("https://", "");

                if url.starts_with("https://") || url.starts_with("http://") {
                    // fine, can use as is.
                } else if url.starts_with('/') {
                    #[cfg(target_family = "wasm")]
                    {
                        // Assume that this instead points to a GET request for the server.
                        url = web_sys::window()
                            .expect("No window object available")
                            .location()
                            .origin()
                            .expect("Coultn't figure out origin")
                            + &url;
                    }

                    // On non-wasm... not much we can do here, what server would we ask?
                } else {
                    // Just try to add https:// and hope for the best. Eg. if someone specifies google.com/splat.ply.
                    url = format!("https://{url}");
                }

                let response = reqwest::get(url)
                    .await
                    .map_err(|e| anyhow!(e))?
                    .bytes_stream();

                let response =
                    response.map(|b| b.map_err(|_e| std::io::ErrorKind::ConnectionAborted));
                let reader = StreamReader::new(response);
                Self::vfs_from_reader(reader).await
            }
            Self::Path(path) => BrushVfs::from_directory(&PathBuf::from(path)).await,
        }
    }
}
