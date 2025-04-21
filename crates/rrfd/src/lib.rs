#[cfg(target_os = "android")]
pub mod android;

#[allow(unused)]
use anyhow::Context;
use anyhow::Result;
use std::path::PathBuf;
use tokio::io::AsyncRead;

/// Pick a file and return the name & bytes of the file.
pub async fn pick_file() -> Result<impl AsyncRead + Unpin> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .pick_file()
            .await
            .context("No file selected")?;

        #[cfg(target_family = "wasm")]
        {
            Ok(std::io::Cursor::new(file.read().await))
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let file = tokio::fs::File::open(file.path())
                .await
                .expect("Internal file picking error");
            Ok(tokio::io::BufReader::new(file))
        }
    }

    #[cfg(target_os = "android")]
    {
        let file = android::pick_file().await?;
        tokio::io::BufReader::new(file)
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
        panic!("No folder picking on Android or wasm yet.")
    }
}

/// Saves data to a file and returns the filename the data was saved too.
///
/// Nb: Does not work on Android currently.
pub async fn save_file(default_name: &str, data: Vec<u8>) -> Result<()> {
    #[cfg(not(target_os = "android"))]
    {
        let file = rfd::AsyncFileDialog::new()
            .set_file_name(default_name)
            .save_file()
            .await
            .context("No file selected")?;

        #[cfg(not(target_family = "wasm"))]
        tokio::fs::write(file.path(), data).await?;

        #[cfg(target_family = "wasm")]
        file.write(&data);

        Ok(())
    }

    #[cfg(target_os = "android")]
    {
        let _ = default_name;
        panic!("No saving on Android yet.")
    }
}
