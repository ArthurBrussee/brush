use std::pin::pin;
use std::time::Duration;

use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::{Splats, inverse_sigmoid};
use brush_render::sh::rgb_to_sh;
use brush_vfs::SendNotWasm;
use glam::{Vec3, Vec4Swizzles};
use serde::Deserialize;
use serde::de::{DeserializeSeed, Error};
use serde_ply::{DeserializeError, PlyChunkedReader, RowVisitor};
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio_stream::{Stream, StreamExt};
use tokio_with_wasm::alias as tokio_wasm;

use crate::ply_gaussian::{PlyGaussian, QuantSh, QuantSplat};

type StreamEmitter = TryStreamEmitter<SplatMessage, DeserializeError>;

pub struct ParseMetadata {
    pub up_axis: Option<Vec3>,
    pub total_splats: u32,
    pub frame_count: u32,
    pub current_frame: u32,
    pub progress: f32,
}

/// Raw splat data parsed from a PLY file.
/// Fields are optional - only positions are guaranteed.
#[derive(Clone)]
pub struct SplatData {
    /// Position data (x, y, z) - always present
    pub means: Vec<f32>,
    pub rotations: Option<Vec<f32>>,
    pub log_scales: Option<Vec<f32>>,
    pub sh_coeffs: Option<Vec<f32>>,
    pub raw_opacities: Option<Vec<f32>>,
}

impl SplatData {
    pub fn num_splats(&self) -> usize {
        self.means.len() / 3
    }

    /// Convert into Splats using simple defaults for missing fields.
    pub fn into_splats<B: burn::prelude::Backend>(self, device: &B::Device) -> Splats<B> {
        let n_splats = self.num_splats();
        let rotations = self
            .rotations
            .unwrap_or_else(|| [1.0, 0.0, 0.0, 0.0].repeat(n_splats));
        let log_scales = self.log_scales.unwrap_or_else(|| vec![-4.0; n_splats * 3]);
        let sh_coeffs = self.sh_coeffs.unwrap_or_else(|| vec![0.5; n_splats * 3]);
        let opacities = self
            .raw_opacities
            .unwrap_or_else(|| vec![inverse_sigmoid(0.5); n_splats]);

        Splats::from_raw(
            self.means, rotations, log_scales, sh_coeffs, opacities, device,
        )
    }
}

pub struct SplatMessage {
    pub meta: ParseMetadata,
    pub data: SplatData,
}

enum PlyFormat {
    Ply,
    SuperSplatCompressed,
}

struct TimedUpdate {
    last_update: web_time::Instant,
    update_every: Option<web_time::Duration>,
}

impl TimedUpdate {
    fn new(update_every: Option<web_time::Duration>) -> Self {
        Self {
            last_update: web_time::Instant::now(),
            update_every,
        }
    }

    fn should_update(&mut self, perc_done: f32) -> bool {
        // Don't bother updating if we're almost done
        if perc_done >= 0.95 {
            return false;
        }
        if let Some(duration) = self.update_every
            && self.last_update.elapsed() >= duration
        {
            self.last_update = web_time::Instant::now();
            return true;
        }

        false
    }
}

fn interleave_coeffs(sh_dc: Vec3, sh_rest: &[f32], result: &mut Vec<f32>) {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;

    result.extend([sh_dc.x, sh_dc.y, sh_dc.z]);
    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
}

async fn read_chunk<T: AsyncRead + Unpin>(
    mut reader: T,
    buf: &mut Vec<u8>,
) -> tokio::io::Result<()> {
    buf.reserve(8 * 1024 * 1024);
    let mut total_read = buf.len();
    while total_read < buf.capacity() {
        let bytes_read = reader.read_buf(buf).await?;
        if bytes_read == 0 {
            break;
        }
        total_read += bytes_read;
        tokio_wasm::task::yield_now().await;
    }
    if total_read == 0 {
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Unexpected EOF",
        ))
    } else {
        Ok(())
    }
}

pub async fn load_splat_from_ply<T: AsyncRead + SendNotWasm + Unpin>(
    reader: T,
    subsample_points: Option<u32>,
) -> Result<SplatMessage, DeserializeError> {
    let stream = stream_splat_from_ply(reader, subsample_points, false);
    let Some(splat) = pin!(stream).next().await else {
        return Err(DeserializeError::custom(
            "Couldn't load single splat from ply",
        ));
    };
    splat
}

pub fn stream_splat_from_ply<T: AsyncRead + SendNotWasm + Unpin>(
    mut reader: T,
    subsample_points: Option<u32>,
    streaming: bool,
) -> impl Stream<Item = Result<SplatMessage, DeserializeError>> {
    try_fn_stream(|emitter| async move {
        let mut file = PlyChunkedReader::new();
        read_chunk(&mut reader, file.buffer_mut()).await?;

        let header = file.header().expect("Must have header");
        // Parse some metadata.
        let up_axis = header
            .comments
            .iter()
            .filter_map(|c| match c.to_lowercase().strip_prefix("vertical axis: ") {
                Some("x") => Some(Vec3::X),
                Some("y") => Some(Vec3::NEG_Y),
                Some("z") => Some(Vec3::NEG_Z),
                _ => None,
            })
            .next_back();

        // Check whether there is a vertex header that has at least XYZ.
        let has_vertex = header.elem_defs.iter().any(|el| el.name == "vertex");

        let ply_type = if has_vertex
            && header
                .elem_defs
                .first()
                .is_some_and(|el| el.name == "chunk")
        {
            PlyFormat::SuperSplatCompressed
        } else if has_vertex {
            PlyFormat::Ply
        } else {
            return Err(DeserializeError::custom("Unknown format"));
        };

        let subsample = subsample_points.unwrap_or(1) as usize;
        let mut updater = TimedUpdate::new(streaming.then(|| Duration::from_millis(1500)));

        match ply_type {
            PlyFormat::Ply => {
                parse_ply(
                    reader,
                    subsample,
                    &mut file,
                    up_axis,
                    &emitter,
                    &mut updater,
                )
                .await?;
            }
            PlyFormat::SuperSplatCompressed => {
                parse_compressed_ply(reader, subsample, file, up_axis, emitter, updater).await?;
            }
        }
        Ok(())
    })
}

fn progress(index: usize, len: usize) -> f32 {
    ((index + 1) as f32) / len as f32
}

fn vec_exact(cap: usize) -> Vec<f32> {
    let mut r = vec![];
    r.reserve_exact(cap);
    r
}

async fn parse_ply<T: AsyncRead + Unpin>(
    mut reader: T,
    subsample: usize,
    file: &mut PlyChunkedReader,
    up_axis: Option<Vec3>,
    emitter: &StreamEmitter,
    update: &mut TimedUpdate,
) -> Result<(), DeserializeError> {
    let header = file.header().expect("Must have header");
    let vertex = header
        .get_element("vertex")
        .ok_or(DeserializeError::custom("Unknown format"))?;
    let total_splats = vertex.count;
    let max_splats = total_splats / subsample;

    let sh_count = vertex
        .properties
        .iter()
        .filter(|x| {
            x.name.starts_with("f_rest_")
                || x.name.starts_with("f_dc_")
                || matches!(x.name.as_str(), "r" | "g" | "b" | "red" | "green" | "blue")
        })
        .count();

    let mut data = SplatData {
        means: vec_exact(max_splats * 3),
        rotations: vertex
            .has_property("rot_0")
            .then(|| vec_exact(max_splats * 4)),
        log_scales: vertex
            .has_property("scale_0")
            .then(|| vec_exact(max_splats * 3)),
        sh_coeffs: (sh_count > 0).then(|| vec_exact(max_splats * sh_count)),
        raw_opacities: vertex
            .has_property("opacity")
            .then(|| vec_exact(max_splats)),
    };

    let mut row_index: usize = 0;

    loop {
        read_chunk(&mut reader, file.buffer_mut()).await?;

        RowVisitor::new(|mut gauss: PlyGaussian| {
            row_index += 1;
            if !row_index.is_multiple_of(subsample) {
                return;
            }
            data.means.extend([gauss.x, gauss.y, gauss.z]);

            // Prefer rgb if specified.
            if let Some(r) = gauss.red
                && let Some(g) = gauss.green
                && let Some(b) = gauss.blue
            {
                let sh_dc = rgb_to_sh(Vec3::new(r, g, b));
                gauss.f_dc_0 = sh_dc.x;
                gauss.f_dc_1 = sh_dc.y;
                gauss.f_dc_2 = sh_dc.z;
            }

            if let Some(coeffs) = &mut data.sh_coeffs {
                interleave_coeffs(
                    Vec3::new(gauss.f_dc_0, gauss.f_dc_1, gauss.f_dc_2),
                    &gauss.sh_rest_coeffs()[..sh_count - 3],
                    coeffs,
                );
            }

            if let Some(scales) = &mut data.log_scales {
                scales.extend([gauss.scale_0, gauss.scale_1, gauss.scale_2]);
            }
            if let Some(rotation) = &mut data.rotations {
                rotation.extend([gauss.rot_0, gauss.rot_1, gauss.rot_2, gauss.rot_3]);
            }
            if let Some(opacity) = &mut data.raw_opacities {
                opacity.push(gauss.opacity);
            }
        })
        .deserialize(&mut *file)?;

        if update.should_update(row_index as f32 / total_splats as f32) || row_index == total_splats
        {
            let meta = ParseMetadata {
                total_splats: max_splats as u32,
                up_axis,
                progress: progress(row_index, total_splats),
                frame_count: 0,
                current_frame: 0,
            };

            if row_index == total_splats {
                emitter.emit(SplatMessage { meta, data }).await;
                return Ok(());
            } else {
                emitter
                    .emit(SplatMessage {
                        meta,
                        data: data.clone(),
                    })
                    .await;
            }
        }
    }
}

async fn parse_compressed_ply<T: AsyncRead + Unpin>(
    mut reader: T,
    subsample: usize,
    mut file: PlyChunkedReader,
    up_axis: Option<Vec3>,
    emitter: StreamEmitter,
    mut update: TimedUpdate,
) -> Result<(), DeserializeError> {
    #[derive(Default, Deserialize)]
    struct QuantMeta {
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        min_z: f32,
        max_z: f32,
        min_scale_x: f32,
        max_scale_x: f32,
        min_scale_y: f32,
        max_scale_y: f32,
        min_scale_z: f32,
        max_scale_z: f32,
        min_r: f32,
        max_r: f32,
        min_g: f32,
        max_g: f32,
        min_b: f32,
        max_b: f32,
    }

    impl QuantMeta {
        fn mean(&self, raw: Vec3) -> Vec3 {
            let min = glam::vec3(self.min_x, self.min_y, self.min_z);
            let max = glam::vec3(self.max_x, self.max_y, self.max_z);
            raw * (max - min) + min
        }

        fn scale(&self, raw: Vec3) -> Vec3 {
            let min = glam::vec3(self.min_scale_x, self.min_scale_y, self.min_scale_z);
            let max = glam::vec3(self.max_scale_x, self.max_scale_y, self.max_scale_z);
            raw * (max - min) + min
        }

        fn color(&self, raw: Vec3) -> Vec3 {
            let min = glam::vec3(self.min_r, self.min_g, self.min_b);
            let max = glam::vec3(self.max_r, self.max_g, self.max_b);
            raw * (max - min) + min
        }
    }

    let mut quant_metas = vec![];

    while let Some(element) = file.current_element()
        && element.name == "chunk"
    {
        read_chunk(&mut reader, file.buffer_mut()).await?;
        RowVisitor::new(|meta: QuantMeta| {
            quant_metas.push(meta);
        })
        .deserialize(&mut file)?;
    }

    let vertex = file
        .current_element()
        .ok_or(DeserializeError::custom("Unknown format"))?;

    if vertex.name != "vertex" {
        return Err(DeserializeError::custom("Unknown format"));
    }
    let total_splats = vertex.count;
    let max_splats = total_splats / subsample;

    let mut means = Vec::with_capacity(max_splats * 3);
    // Atm, unlike normal plys, these values aren't optional.
    let mut log_scales = Vec::with_capacity(max_splats * 3);
    let mut rotations = Vec::with_capacity(max_splats * 4);
    let mut sh_coeffs = Vec::with_capacity(max_splats * 3);
    let mut opacity = Vec::with_capacity(max_splats);

    let mut row_count = 0;

    let sh_vals = file
        .header()
        .expect("Must have header")
        .elem_defs
        .get(2)
        .cloned();

    while let Some(element) = file.current_element()
        && element.name == "vertex"
    {
        read_chunk(&mut reader, file.buffer_mut()).await?;

        RowVisitor::new(|splat: QuantSplat| {
            let quant_data = &quant_metas[row_count / 256];
            row_count += 1;
            if row_count % subsample != 0 {
                return;
            }
            means.extend(quant_data.mean(splat.mean).to_array());
            log_scales.extend(quant_data.scale(splat.log_scale).to_array());
            // Nb: Scalar order.
            rotations.extend([
                splat.rotation.w,
                splat.rotation.x,
                splat.rotation.y,
                splat.rotation.z,
            ]);
            // Compressed ply specifies things in post-activated values. Convert to pre-activated values.
            opacity.push(inverse_sigmoid(splat.rgba.w));
            // These come in as RGB colors. Convert to base SH coefficients.
            let sh_dc = rgb_to_sh(quant_data.color(splat.rgba.xyz()));
            sh_coeffs.extend([sh_dc.x, sh_dc.y, sh_dc.z]);
        })
        .deserialize(&mut file)?;

        // Occasionally send some updated splats.
        if update.should_update(row_count as f32 / total_splats as f32) || row_count == total_splats
        {
            // Leave 20% of progress for loading the SH's, just an estimate.
            let max_time = if sh_vals.is_some() { 0.8 } else { 1.0 };
            let progress = progress(row_count, total_splats) * max_time;
            let meta = ParseMetadata {
                total_splats: max_splats as u32,
                up_axis,
                frame_count: 0,
                current_frame: 0,
                progress,
            };

            let data = SplatData {
                means: means.clone(),
                rotations: Some(rotations.clone()),
                log_scales: Some(log_scales.clone()),
                sh_coeffs: Some(sh_coeffs.clone()),
                raw_opacities: Some(opacity.clone()),
            };
            emitter.emit(SplatMessage { meta, data }).await;
        }
    }

    if let Some(sh_vals) = sh_vals {
        let sh_count = sh_vals.properties.len();
        let mut total_coeffs = Vec::with_capacity(sh_vals.count * (3 + sh_count));
        let mut splat_index = 0;

        let mut row_count = 0;

        while let Some(element) = file.current_element()
            && element.name == "sh"
        {
            read_chunk(&mut reader, file.buffer_mut()).await?;

            RowVisitor::new(|quant_sh: QuantSh| {
                row_count += 1;
                if row_count % subsample != 0 {
                    return;
                }
                let dc = glam::vec3(
                    sh_coeffs[splat_index * 3],
                    sh_coeffs[splat_index * 3 + 1],
                    sh_coeffs[splat_index * 3 + 2],
                );
                interleave_coeffs(
                    dc,
                    &quant_sh.sh_rest_coeffs()[..sh_count],
                    &mut total_coeffs,
                );
                splat_index += 1;
            })
            .deserialize(&mut file)?;
        }

        let meta = ParseMetadata {
            total_splats: (means.len() / 3) as u32,
            up_axis,
            frame_count: 0,
            current_frame: 0,
            progress: 1.0,
        };
        let data = SplatData {
            means,
            rotations: Some(rotations),
            log_scales: Some(log_scales),
            sh_coeffs: Some(total_coeffs),
            raw_opacities: Some(opacity),
        };
        emitter.emit(SplatMessage { meta, data }).await;
    }

    Ok(())
}

#[cfg(all(test, feature = "export"))]
mod tests {
    use super::*;
    use crate::export::splat_to_ply;
    use crate::test_utils::{create_test_splats, create_test_splats_with_count};
    use brush_render::sh::sh_coeffs_for_degree;
    use std::io::Cursor;

    #[tokio::test]
    async fn test_import_basic_functionality() {
        let original_splats = create_test_splats(1);
        let ply_bytes = splat_to_ply(original_splats.clone()).await.unwrap();

        let cursor = Cursor::new(ply_bytes);
        let imported_message = load_splat_from_ply(cursor, None).await.unwrap();

        assert_eq!(imported_message.data.num_splats(), 1);
        assert_eq!(imported_message.meta.total_splats, 1);
        // All fields should be present for a full PLY
        assert!(imported_message.data.rotations.is_some());
        assert!(imported_message.data.log_scales.is_some());
        assert!(imported_message.data.sh_coeffs.is_some());
        assert!(imported_message.data.raw_opacities.is_some());
    }

    #[tokio::test]
    async fn test_import_different_sh_degrees() {
        for degree in [0, 1, 2] {
            let original_splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(original_splats).await.unwrap();

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None).await.unwrap();

            let n_splats = imported_message.data.num_splats();
            let sh_coeffs = imported_message.data.sh_coeffs.unwrap();
            let n_coeffs = sh_coeffs.len() / n_splats / 3;
            assert_eq!(n_coeffs, sh_coeffs_for_degree(degree) as usize);
        }
    }

    #[tokio::test]
    async fn test_import_with_subsample() {
        // Create 4 test splats
        let original_splats = create_test_splats_with_count(0, 4);
        assert_eq!(original_splats.num_splats(), 4);

        let ply_bytes = splat_to_ply(original_splats).await.unwrap();

        // Test no subsampling
        let cursor = Cursor::new(ply_bytes.clone());
        let imported_message = load_splat_from_ply(cursor, None).await.unwrap();
        assert_eq!(imported_message.data.num_splats(), 4);

        // Test subsample every 2nd splat
        let cursor = Cursor::new(ply_bytes);
        let imported_message = load_splat_from_ply(cursor, Some(2)).await.unwrap();
        assert_eq!(imported_message.data.num_splats(), 2);
    }
}
