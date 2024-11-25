pub mod android;
pub mod ios;

#[allow(unused)]
use anyhow::Context;
use anyhow::Result;
use tokio::sync::mpsc::Sender;
use std::path::PathBuf;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub use rfd;

#[cfg(any(target_os = "android", target_os = "ios"))]
pub struct FileDialog;

#[cfg(any(target_os = "android", target_os = "ios"))]
impl FileDialog {
    pub fn new() -> Self {
        FileDialog
    }

    pub async fn pick_file(self) -> Option<std::path::PathBuf> {
        None // Mobile platforms don't support direct file picking
    }
}

pub async fn pick_file_async(_sender: Sender<Result<Vec<u8>>>) -> Result<()> {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        if let Some(file) = rfd::FileDialog::new().pick_file() {
            let contents = std::fs::read(file)?;
            _sender.send(Ok(contents)).await?;
        }
    }
    
    #[cfg(target_os = "ios")]
    {
        if let Ok(picked_file) = ios::pick_file().await {
            _sender.send(Ok(picked_file.data)).await?;
        }
    }
    #[cfg(target_os = "ios")]
    Ok(())
}

pub enum FileHandle {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    Rfd(rfd::FileHandle),
    #[cfg(target_os = "android")]
    Android(android::PickedFile),
    #[cfg(target_os = "ios")]
    Ios(ios::PickedFile),
}

impl FileHandle {
    pub fn file_name(&self) -> String {
        match self {
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            FileHandle::Rfd(file_handle) => file_handle.file_name(),
            #[cfg(target_os = "android")]
            FileHandle::Android(picked_file) => picked_file.file_name.clone(),
            #[cfg(target_os = "ios")]
            FileHandle::Ios(picked_file) => picked_file.file_name.clone(),
        }
    }

    pub async fn write(&self, data: &[u8]) -> std::io::Result<()> {
        match self {
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            FileHandle::Rfd(file_handle) => file_handle.write(data).await,
            #[cfg(target_os = "android")]
            FileHandle::Android(_) => {
                let _ = data;
                unimplemented!("No saving on Android yet.")
            }
            #[cfg(target_os = "ios")]
            FileHandle::Ios(_) => {
                let _ = data;
                unimplemented!("No saving on iOS yet.")
            }
        }
    }

    pub async fn read(&self) -> Vec<u8> {
        match self {
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            FileHandle::Rfd(file_handle) => file_handle.read().await,
            #[cfg(target_os = "android")]
            FileHandle::Android(picked_file) => picked_file.data.clone(),
            #[cfg(target_os = "ios")]
            FileHandle::Ios(picked_file) => picked_file.data.clone(),
        }
    }
}

/// Saves data to a file and returns the filename the data was saved too.
///
/// Nb: Does not work on Android currently.
pub async fn save_file(default_name: &str) -> Result<FileHandle> {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        let file = rfd::AsyncFileDialog::new()
            .set_file_name(default_name)
            .save_file()
            .await
            .context("No file selected")?;
        Ok(FileHandle::Rfd(file))
    }
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let _ = default_name;
        unimplemented!("No saving on mobile platforms yet.")
    }
}

#[cfg(not(any(target_os = "ios", target_os = "android")))]
pub async fn pick_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .pick_file()
        .await
}

#[cfg(any(target_os = "ios", target_os = "android"))]
pub async fn pick_file() -> Option<PathBuf> {
    if let Ok(picked_file) = ios::pick_file().await {
        Some(picked_file.path)
    } else {
        None
    }
}
