use brush_vfs::BrushVfs;
use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::io::AsyncReadExt;

/// Header magic for the packed `LiDAR` depth files emitted by
/// `tools/ios_lidar_to_colmap.py`: `magic(8) | width u32 | height u32 |
/// depth f32[w*h] (z-depth, metres) | confidence u8[w*h] (``ARKit`` 0/1/2)`.
const DEPTH_MAGIC: &[u8; 8] = b"BRDPTH\x01\x00";

/// Loader for a per-view metric `LiDAR` depth + confidence sidecar.
#[derive(Clone, Debug)]
pub struct LoadDepth {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
}

/// Decoded depth: row-major `z`-depth in metres + `ARKit` confidence.
#[derive(Clone, Debug)]
pub struct DepthData {
    pub width: usize,
    pub height: usize,
    pub depth: Vec<f32>,
    pub conf: Vec<u8>,
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

    pub async fn load(&self) -> io::Result<DepthData> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;

        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_owned());
        if bytes.len() < 16 || &bytes[..8] != DEPTH_MAGIC {
            return Err(bad("depth file: bad magic / too short"));
        }
        let width = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let n = width * height;
        let depth_end = 16 + n * 4;
        if bytes.len() < depth_end + n {
            return Err(bad("depth file: truncated payload"));
        }
        // Reinterpret the little-endian f32 payload in a single copy.
        // `pod_collect_to_vec` tolerates the unaligned `&[u8]` (a plain
        // `cast_slice` would need 4-byte alignment); native byte order is
        // little-endian on all supported targets.
        let depth = bytemuck::pod_collect_to_vec::<u8, f32>(&bytes[16..depth_end]);
        let conf = bytes[depth_end..depth_end + n].to_vec();
        Ok(DepthData {
            width,
            height,
            depth,
            conf,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
