#[cfg(target_os = "android")]
pub mod android;

#[allow(unused)]
use anyhow::Context;
use anyhow::Result;
use std::path::PathBuf;

pub enum FileHandle {
    #[cfg(not(target_os = "android"))]
    Rfd(rfd::FileHandle),
    #[cfg(target_os = "android")]
    Android(tokio::fs::File),
}

impl FileHandle {
    pub async fn write(&self, data: &[u8]) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_os = "android"))]
            Self::Rfd(file_handle) => file_handle.write(data).await,
            #[cfg(target_os = "android")]
            Self::Android(_) => {
                let _ = data;
                unimplemented!("No saving on Android yet.")
            }
        }
    }

    pub async fn read(mut self) -> Vec<u8> {
        match &mut self {
            #[cfg(not(target_os = "android"))]
            Self::Rfd(file_handle) => file_handle.read().await,
            #[cfg(target_os = "android")]
            Self::Android(file) => {
                use tokio::io::AsyncReadExt;

                let mut buf = vec![];
                file.read_to_end(&mut buf).await.unwrap();
                buf
            }
        }
    }
}

/// Pick a file and return the name & bytes of the file.
pub async fn pick_file() -> Result<FileHandle> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .pick_file()
            .await
            .context("No file selected")?;
        Ok(FileHandle::Rfd(file))
    }

    #[cfg(target_os = "android")]
    {
        android::pick_file().await.map(FileHandle::Android)
    }
}

pub async fn pick_directory() -> Result<PathBuf> {
    #[cfg(all(not(target_os = "android"), not(target_family = "wasm")))]
    {
        let dir = rfd::AsyncFileDialog::new()
            .pick_folder()
            .await
            .context("No folder selected")?;

        Ok(dir.path().to_path_buf())
    }

    #[cfg(any(target_os = "android", target_family = "wasm"))]
    {
        unimplemented!("No folder picking on Android yet.")
    }
}

/// Saves data to a file and returns the filename the data was saved too.
///
/// Nb: Does not work on Android currently.
pub async fn save_file(default_name: &str) -> Result<FileHandle> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .set_file_name(default_name)
            .save_file()
            .await
            .context("No file selected")?;
        Ok(FileHandle::Rfd(file))
    }

    #[cfg(target_os = "android")]
    {
        let _ = default_name;
        unimplemented!("No saving on Android yet.")
    }
}
