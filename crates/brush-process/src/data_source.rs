use std::io::Cursor;
use std::path::PathBuf;
use std::{path::Path, str::FromStr};

use anyhow::anyhow;

use brush_dataset::WasmNotSend;
use brush_dataset::brush_vfs::{BrushVfs, PathReader};
use tokio::io::{AsyncRead, AsyncReadExt, BufReader};
use tokio_stream::StreamExt;
use tokio_util::io::StreamReader;

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
            anyhow::bail!(
                "Failed to download data (are you trying to download from Google Drive? You might have to use the proxy."
            )
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
                if !url.starts_with("http://") && !url.starts_with("https://") {
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
