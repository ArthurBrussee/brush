use anyhow::Context;
use async_fn_stream::try_fn_stream;

use tokio::io::AsyncRead;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tokio_util::{bytes::Bytes, io::StreamReader};
use tokio_with_wasm::alias as tokio_wasm;

#[derive(Clone, Debug)]
pub enum DataSource {
    PickFile,
    PickDirectory,
    Url(String),
}

impl DataSource {
    pub fn into_reader(self) -> impl AsyncRead + Send {
        let (send, rec) = tokio::sync::mpsc::channel(16);

        // Spawn the data reading.
        tokio_wasm::spawn(async move {
            let stream = try_fn_stream(|emitter| async move {
                match self {
                    Self::PickFile => {
                        let picked = rrfd::pick_file()
                            .await
                            .map_err(|_e| std::io::ErrorKind::NotFound)?;
                        let data = picked.read().await;
                        emitter.emit(Bytes::from_owner(data)).await;
                    }
                    Self::PickDirectory => {
                        let picked = rrfd::pick_directory()
                            .await
                            .map_err(|_e| std::io::ErrorKind::NotFound)?;
                        let data = picked;
                        let mut bytes = b"BRUSH_PATH".to_vec();
                        let path_bytes = data
                            .to_str()
                            .context("invalid path")
                            .map_err(|_e| std::io::ErrorKind::InvalidData)?
                            .as_bytes();
                        bytes.extend(path_bytes);
                        emitter.emit(Bytes::from_owner(bytes)).await;
                    }
                    Self::Url(url) => {
                        let mut url = url.clone();
                        if !url.starts_with("http://") && !url.starts_with("https://") {
                            url = format!("https://{url}");
                        }
                        let mut response = reqwest::get(url)
                            .await
                            .map_err(|_e| std::io::ErrorKind::InvalidInput)?
                            .bytes_stream();

                        while let Some(bytes) = response.next().await {
                            let bytes =
                                bytes.map_err(|_e| std::io::ErrorKind::ConnectionAborted)?;
                            emitter.emit(bytes).await;
                        }
                    }
                };
                anyhow::Result::<(), std::io::Error>::Ok(())
            });

            let mut stream = std::pin::pin!(stream);

            while let Some(data) = stream.next().await {
                if send.send(data).await.is_err() {
                    break;
                }
            }
        });

        StreamReader::new(ReceiverStream::new(rec))
    }
}
