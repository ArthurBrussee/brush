pub mod android;

#[allow(unused)]
use anyhow::Context;
use anyhow::Result;
use tokio::sync::mpsc::Sender;

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

pub async fn pick_file_async(sender: Sender<Result<Vec<u8>>>) -> Result<()> {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        if let Some(file) = rfd::FileDialog::new().pick_file() {
            let contents = std::fs::read(file)?;
            sender.send(Ok(contents)).await?;
        }
    }
    
    Ok(())
}

pub enum FileHandle {
    #[cfg(not(target_os = "android"))]
    Rfd(rfd::FileHandle),
    Android(android::PickedFile),
}

impl FileHandle {
    pub fn file_name(&self) -> String {
        match self {
            #[cfg(not(target_os = "android"))]
            FileHandle::Rfd(file_handle) => file_handle.file_name(),
            FileHandle::Android(picked_file) => picked_file.file_name.clone(),
        }
    }

    pub async fn write(&self, data: &[u8]) -> std::io::Result<()> {
        match self {
            #[cfg(not(target_os = "android"))]
            FileHandle::Rfd(file_handle) => file_handle.write(data).await,
            FileHandle::Android(_) => {
                let _ = data;
                unimplemented!("No saving on Android yet.")
            }
        }
    }

    pub async fn read(&self) -> Vec<u8> {
        match self {
            #[cfg(not(target_os = "android"))]
            FileHandle::Rfd(file_handle) => file_handle.read().await,
            FileHandle::Android(picked_file) => picked_file.data.clone(),
        }
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

#[cfg(not(any(target_os = "ios", target_os = "android")))]
pub fn pick_file() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .pick_file()
}

#[cfg(any(target_os = "ios", target_os = "android"))]
pub fn pick_file() -> Option<PathBuf> {
    None // Mobile platforms don't support standard file dialogs
}
