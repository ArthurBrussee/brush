use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use tokio::io::AsyncReadExt;

/// Lazily-loaded per-view depth map, but for single-channel float32 depth stored as TIFF
/// (depth in metres, 0 marks an invalid depth).
#[derive(Clone, Debug)]
pub struct LoadDepth {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
}

impl PartialEq for LoadDepth {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl LoadDepth {
    pub fn new(vfs: Arc<BrushVfs>, path: PathBuf) -> Self {
        Self { vfs, path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn load(
        &self,
        expected_h: usize,
        expected_w: usize,
    ) -> Result<TensorData, LoadDepthError> {
        let depth = self.load_vec(expected_h, expected_w).await?;
        Ok(TensorData::new(depth, [expected_h, expected_w]))
    }

    pub async fn load_vec(
        &self,
        expected_h: usize,
        expected_w: usize,
    ) -> Result<Vec<f32>, LoadDepthError> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;

        let (depth, w, h) = decode_f32_tiff(&bytes)?;

        if w != expected_w || h != expected_h {
            Err(LoadDepthError::ReadTiffError(format!(
                "invalid depth size {w} x {h}, expected {expected_w} x {expected_h}"
            )))
        } else {
            Ok(depth)
        }
    }
}

/// Decode a single-channel float32 TIFF into in row-major order.
fn decode_f32_tiff(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), LoadDepthError> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes))?;

    let tiff::decoder::DecodingResult::F32(depth) = decoder.read_image()? else {
        return Err(LoadDepthError::ReadTiffError(
            "unsupported TIFF sample format (expected float32 depth)".to_owned(),
        ));
    };

    let (w, h) = decoder.dimensions()?;
    let (w, h) = (w as usize, h as usize);

    if w * h != depth.len() {
        Err(LoadDepthError::ReadTiffError(
            "expected only a single channel".to_owned(),
        ))
    } else {
        Ok((depth, w, h))
    }
}

#[derive(Error, Debug)]
pub enum LoadDepthError {
    #[error("I/O error while loading depth map: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error while loading TIFF file: {0}")]
    LoadTiffError(#[from] tiff::TiffError),

    #[error("Error while reading TIFF file: {0}")]
    ReadTiffError(String),
}
