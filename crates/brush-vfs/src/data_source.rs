use crate::BrushVfs;
use anyhow::anyhow;
use std::{path::Path, str::FromStr};
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
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            s if s.starts_with("http://") || s.starts_with("https://") => {
                Ok(Self::Url(s.to_owned()))
            }
            s if std::fs::exists(s).is_ok() => Ok(Self::Path(s.to_owned())),
            s => anyhow::bail!("Invalid data source. Can't find {s}"),
        }
    }
}

impl DataSource {
    pub async fn into_vfs(self) -> anyhow::Result<BrushVfs> {
        match self {
            Self::PickFile => {
                let reader = rrfd::pick_file().await.map_err(|e| anyhow!(e))?;
                BrushVfs::from_reader(reader).await
            }
            Self::PickDirectory => {
                let picked = rrfd::pick_directory().await.map_err(|e| anyhow!(e))?;
                BrushVfs::from_path(&picked).await
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
                BrushVfs::from_reader(reader).await
            }
            Self::Path(path) => BrushVfs::from_path(Path::new(&path)).await,
        }
    }
}
