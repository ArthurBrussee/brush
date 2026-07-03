use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use tokio::io::AsyncReadExt;

/// Lazily-loaded per-view feature map, stored as a numpy `.npy` file
/// containing a C-order `[H, W, C]` little-endian float32 array.
#[derive(Clone, Debug)]
pub struct LoadFeatures {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
}

impl PartialEq for LoadFeatures {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl LoadFeatures {
    pub fn new(vfs: Arc<BrushVfs>, path: PathBuf) -> Self {
        Self { vfs, path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load the feature map as `[H, W, C]` f32 [`TensorData`] plus the channel count.
    pub async fn load(&self) -> Result<(TensorData, usize), LoadFeaturesError> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;

        let (features, h, w, c) = decode_npy_f32_3d(&bytes)?;
        Ok((TensorData::new(features, [h, w, c]), c))
    }
}

/// Extract the value following `'key':` in an npy header dict.
fn npy_dict_value<'a>(header: &'a str, key: &str) -> Result<&'a str, LoadFeaturesError> {
    let pattern = format!("'{key}'");
    let start = header
        .find(&pattern)
        .ok_or_else(|| LoadFeaturesError::ReadNpyError(format!("missing '{key}' in header")))?;
    let rest = header[start + pattern.len()..].trim_start();
    rest.strip_prefix(':')
        .map(str::trim_start)
        .ok_or_else(|| LoadFeaturesError::ReadNpyError(format!("malformed '{key}' in header")))
}

/// Decode the subset of the numpy `.npy` format we need: version 1.0/2.0,
/// dtype `<f4`, C-order, 3-D shape. Returns the data in row-major order
/// along with the `[h, w, c]` shape.
fn decode_npy_f32_3d(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize, usize), LoadFeaturesError> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(LoadFeaturesError::ReadNpyError(
            "not an npy file (bad magic)".to_owned(),
        ));
    }

    let version = bytes[6];
    let (header_len, header_start) = match version {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 => {
            if bytes.len() < 12 {
                return Err(LoadFeaturesError::ReadNpyError(
                    "truncated npy header".to_owned(),
                ));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (len as usize, 12)
        }
        _ => {
            return Err(LoadFeaturesError::ReadNpyError(format!(
                "unsupported npy version {version}"
            )));
        }
    };

    let data_start = header_start + header_len;
    if bytes.len() < data_start {
        return Err(LoadFeaturesError::ReadNpyError(
            "truncated npy header".to_owned(),
        ));
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|e| LoadFeaturesError::ReadNpyError(format!("invalid npy header: {e}")))?;

    let descr = npy_dict_value(header, "descr")?;
    if !descr.starts_with("'<f4'") {
        return Err(LoadFeaturesError::ReadNpyError(format!(
            "unsupported npy dtype (expected '<f4'), header: {header}"
        )));
    }
    if !npy_dict_value(header, "fortran_order")?.starts_with("False") {
        return Err(LoadFeaturesError::ReadNpyError(
            "fortran-order npy files are not supported".to_owned(),
        ));
    }

    let shape_str = npy_dict_value(header, "shape")?;
    let shape_end = shape_str
        .find(')')
        .ok_or_else(|| LoadFeaturesError::ReadNpyError("malformed 'shape' in header".to_owned()))?;
    let shape: Vec<usize> = shape_str[1..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| LoadFeaturesError::ReadNpyError(format!("invalid shape: {e}")))
        })
        .collect::<Result<_, _>>()?;
    let [h, w, c] = shape[..] else {
        return Err(LoadFeaturesError::ReadNpyError(format!(
            "expected a 3-D array, got shape {shape:?}"
        )));
    };

    let data = &bytes[data_start..];
    if data.len() != h * w * c * 4 {
        return Err(LoadFeaturesError::ReadNpyError(format!(
            "invalid npy data size {} for shape [{h}, {w}, {c}]",
            data.len()
        )));
    }

    let features = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((features, h, w, c))
}

#[derive(Error, Debug)]
pub enum LoadFeaturesError {
    #[error("I/O error while loading feature map: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error while reading npy file: {0}")]
    ReadNpyError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal npy v1.0 file with the given header dict and f32 data.
    fn make_npy(header: &str, data: &[f32]) -> Vec<u8> {
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let mut header = header.to_owned();
        header.push('\n');
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(data.iter().flat_map(|v| v.to_le_bytes()));
        bytes
    }

    #[test]
    fn test_decode_npy_round_trip() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        let bytes = make_npy(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 4), }",
            &data,
        );

        let (features, h, w, c) = decode_npy_f32_3d(&bytes).expect("valid npy");
        assert_eq!((h, w, c), (2, 3, 4));
        assert_eq!(features, data);
    }

    #[test]
    fn test_decode_npy_rejects_bad_input() {
        // Bad magic.
        assert!(decode_npy_f32_3d(b"not an npy file").is_err());
        // Fortran order.
        let bytes = make_npy(
            "{'descr': '<f4', 'fortran_order': True, 'shape': (1, 1, 1), }",
            &[0.0],
        );
        assert!(decode_npy_f32_3d(&bytes).is_err());
        // Unsupported dtype.
        let bytes = make_npy(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 1, 1), }",
            &[0.0],
        );
        assert!(decode_npy_f32_3d(&bytes).is_err());
        // Wrong dimensionality.
        let bytes = make_npy(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }",
            &[0.0; 4],
        );
        assert!(decode_npy_f32_3d(&bytes).is_err());
    }
}
