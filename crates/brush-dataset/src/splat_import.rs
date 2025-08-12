use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::Splats;
use brush_render::{MainBackend, gaussian_splats::inverse_sigmoid, sh::rgb_to_sh};
use brush_vfs::{DynStream, SendNotWasm};
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};
use glam::{Quat, Vec3, Vec4, Vec4Swizzles};
use serde::Deserialize;
use serde::de::DeserializeSeed;
use serde_ply::RowVisitor;
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufRead, AsyncRead, BufReader};

use crate::parsed_gaussian::{PlyGaussian, QuantSh, QuantSplat};

pub struct ParseMetadata {
    pub up_axis: Option<Vec3>,
    pub total_splats: u32,
    pub frame_count: u32,
    pub current_frame: u32,
    pub progress: f32,
}

pub struct SplatMessage {
    pub meta: ParseMetadata,
    pub splats: Splats<MainBackend>,
}

enum PlyFormat {
    Ply,
    Brush4DCompressed,
    SuperSplatCompressed,
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

#[derive(Debug, Error)]
pub enum SplatImportError {
    #[error("IO error while importing ply file.")]
    Io(#[from] std::io::Error),

    #[error("Invalid ply format")]
    InvalidFormat,

    #[error("Failed to parse ply file.")]
    ParseError(#[from] serde_ply::PlyError),
}

async fn read_up_to<T: AsyncRead + Unpin>(
    reader: &mut T,
    buf: &mut Vec<u8>,
    read_amount: usize,
) -> tokio::io::Result<usize> {
    buf.reserve(read_amount);
    let mut total_read = buf.len();
    while total_read < buf.capacity() {
        let bytes_read = reader.read_buf(buf).await?;
        if bytes_read == 0 {
            break;
        }
        total_read += bytes_read;
    }
    Ok(total_read)
}

pub fn load_splat_from_ply<T: AsyncRead + SendNotWasm + Unpin + 'static>(
    reader: T,
    subsample_points: Option<u32>,
    device: WgpuDevice,
) -> impl DynStream<Result<SplatMessage, SplatImportError>> {
    try_fn_stream(|emitter| async move {
        // set up a reader
        let mut reader = BufReader::with_capacity(1024 * 32, reader);

        // TODO: Just make chunk ply take in data and try to get a header? Simpler maybe.
        let mut file = serde_ply::ChunkPlyFile::new();
        let _ = read_up_to(&mut reader, file.buffer_mut(), 1024 * 1024).await?;

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
        } else if has_vertex
            && header
                .elem_defs
                .iter()
                .any(|el| el.name.starts_with("delta_vertex_"))
        {
            PlyFormat::Brush4DCompressed
        } else if has_vertex {
            PlyFormat::Ply
        } else {
            return Err(SplatImportError::InvalidFormat);
        };

        match ply_type {
            PlyFormat::Ply => {
                parse_ply(
                    reader,
                    subsample_points,
                    device,
                    &mut file,
                    up_axis,
                    &emitter,
                )
                .await?;
            }
            PlyFormat::Brush4DCompressed => {
                parse_delta_ply(reader, subsample_points, device, file, up_axis, &emitter).await?;
            }
            PlyFormat::SuperSplatCompressed => {
                parse_compressed_ply(reader, subsample_points, device, file, up_axis, &emitter)
                    .await?;
            }
        }

        Ok(())
    })
}

fn progress(index: usize, len: usize) -> f32 {
    ((index + 1) as f32) / len as f32
}

async fn parse_ply<T: AsyncBufRead + Unpin>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: WgpuDevice,
    file: &mut serde_ply::ChunkPlyFile,
    up_axis: Option<Vec3>,
    emitter: &TryStreamEmitter<SplatMessage, SplatImportError>,
) -> Result<Splats<MainBackend>, SplatImportError> {
    let header = file.header().expect("Must have header");
    let vertex = header
        .get_element("vertex")
        .ok_or(SplatImportError::InvalidFormat)?;

    let total_splats = vertex.count;
    let mut means = Vec::with_capacity(total_splats);
    let mut log_scales = vertex
        .has_property("scale_0")
        .then(|| Vec::with_capacity(total_splats));
    let mut rotations = vertex
        .has_property("rot_0")
        .then(|| Vec::with_capacity(total_splats));
    let mut opacity = vertex
        .has_property("opacity")
        .then(|| Vec::with_capacity(total_splats));
    let sh_count = vertex
        .properties
        .iter()
        .filter(|x| {
            x.name.starts_with("f_rest_")
                || x.name.starts_with("f_dc_")
                || matches!(x.name.as_str(), "r" | "g" | "b" | "red" | "green" | "blue")
        })
        .count();
    let mut coeffs = (sh_count > 0).then(|| Vec::with_capacity(total_splats * sh_count));

    let update_every = total_splats.div_ceil(5);
    let mut splat_count: usize = 0;
    let mut last_update = 0;

    loop {
        read_up_to(&mut reader, file.buffer_mut(), 16 * 1024 * 1024).await?;

        let Some(element) = file.current_element() else {
            return Err(SplatImportError::InvalidFormat);
        };

        if element.name != "vertex" {
            return Err(SplatImportError::InvalidFormat);
        }

        RowVisitor::new(|mut gauss: PlyGaussian| {
            if !gauss.is_finite() {
                return;
            }

            splat_count += 1;

            // Don't add subsampled gaussians.
            if let Some(subsample) = subsample_points
                && splat_count % (subsample as usize) == 0
            {
                return;
            }

            means.push(Vec3::new(gauss.x, gauss.y, gauss.z));

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

            if let Some(coeffs) = &mut coeffs {
                interleave_coeffs(
                    Vec3::new(gauss.f_dc_0, gauss.f_dc_1, gauss.f_dc_2),
                    &gauss.sh_rest_coeffs()[..sh_count - 3],
                    coeffs,
                );
            }

            if let Some(scales) = &mut log_scales {
                scales.push(Vec3::new(gauss.scale_0, gauss.scale_1, gauss.scale_2));
            }
            if let Some(rotation) = &mut rotations {
                // Ply files are in scalar order.
                rotation.push(Quat::from_xyzw(
                    gauss.rot_1,
                    gauss.rot_2,
                    gauss.rot_3,
                    gauss.rot_0,
                ));
            }
            if let Some(opacity) = &mut opacity {
                opacity.push(gauss.opacity);
            }
        })
        .deserialize(&mut *file)?;

        if splat_count - last_update > update_every || splat_count == total_splats {
            let splats = Splats::from_raw(
                &means,
                rotations.as_deref(),
                log_scales.as_deref(),
                coeffs.as_deref(),
                opacity.as_deref(),
                &device,
            );

            emitter
                .emit(SplatMessage {
                    meta: ParseMetadata {
                        total_splats: total_splats as u32,
                        up_axis,
                        frame_count: 0,
                        current_frame: 0,
                        progress: progress(splat_count, total_splats),
                    },
                    splats: splats.clone(),
                })
                .await;

            last_update = splat_count;

            if splat_count == total_splats {
                return Ok(splats);
            }
        }
    }
}

async fn parse_delta_ply<T: AsyncBufRead + Unpin + 'static>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: WgpuDevice,
    mut file: serde_ply::ChunkPlyFile,
    up_axis: Option<Vec3>,
    emitter: &TryStreamEmitter<SplatMessage, SplatImportError>,
) -> Result<(), SplatImportError> {
    let splats = parse_ply(
        &mut reader,
        subsample_points,
        device.clone(),
        &mut file,
        up_axis,
        emitter,
    )
    .await?;

    // Check for frame count.
    let frame_count = file
        .header()
        .expect("Must have header")
        .elem_defs
        .iter()
        .filter(|e| e.name.starts_with("delta_vertex_"))
        .count() as u32;

    let mut frame = 0;

    fn dequant_or_f32<'de, D>(deserializer: D) -> Result<f32, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct DequantOrF32Visitor;
        impl<'de> serde::de::Visitor<'de> for DequantOrF32Visitor {
            type Value = f32;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a float or a quantized value")
            }

            fn visit_f32<E>(self, value: f32) -> Result<f32, E> {
                Ok(value)
            }

            fn visit_u8<E>(self, value: u8) -> Result<f32, E> {
                Ok(value as f32 / (u8::MAX - 1) as f32)
            }

            fn visit_u16<E>(self, value: u16) -> Result<f32, E> {
                Ok(value as f32 / (u16::MAX - 1) as f32)
            }
        }
        deserializer.deserialize_any(DequantOrF32Visitor)
    }

    #[derive(Deserialize, Default)]
    struct Frame {
        #[serde(deserialize_with = "dequant_or_f32")]
        x: f32,
        #[serde(deserialize_with = "dequant_or_f32")]
        y: f32,
        #[serde(deserialize_with = "dequant_or_f32")]
        z: f32,

        scale_0: f32,
        scale_1: f32,
        scale_2: f32,

        rot_0: f32,
        rot_1: f32,
        rot_2: f32,
        rot_3: f32,
    }

    let mut min_mean = Vec3::ZERO;
    let mut max_mean = Vec3::ZERO;

    let mut min_scale = Vec3::ZERO;
    let mut max_scale = Vec3::ZERO;

    let mut min_rot = Vec4::ZERO;
    let mut max_rot = Vec4::ZERO;

    loop {
        if read_up_to(&mut reader, file.buffer_mut(), 16 * 1024 * 1024).await? == 0 {
            break;
        }

        let Some(element) = file.current_element() else {
            break;
        };

        if element.name.starts_with("meta_delta_min_") {
            RowVisitor::new(|meta: Frame| {
                min_mean = glam::vec3(meta.x, meta.y, meta.z);
                min_scale = glam::vec3(meta.scale_0, meta.scale_1, meta.scale_2);
                min_rot = glam::vec4(meta.rot_0, meta.rot_1, meta.rot_2, meta.rot_3);
            })
            .deserialize(&mut file)?;
        } else if element.name.starts_with("meta_delta_max_") {
            RowVisitor::new(|meta: Frame| {
                max_mean = glam::vec3(meta.x, meta.y, meta.z);
                max_scale = glam::vec3(meta.scale_0, meta.scale_1, meta.scale_2);
                max_rot = glam::vec4(meta.rot_0, meta.rot_1, meta.rot_2, meta.rot_3);
            })
            .deserialize(&mut file)?;
        } else if element.name.starts_with("delta_vertex_") {
            let count = element.count;
            let mut means = Vec::with_capacity(count * 3);
            let mut scales = Vec::with_capacity(count * 3);
            let mut rotations = Vec::with_capacity(count * 4);

            // The splat we decode is normed to 0-1 (if quantized), so rescale to
            // actual values afterwards.
            // Let's only animate transforms for now.
            RowVisitor::new(|meta: Frame| {
                let mean = glam::vec3(meta.x, meta.y, meta.z) * (max_mean - min_mean) + min_mean;
                let scale = glam::vec3(meta.scale_0, meta.scale_1, meta.scale_2)
                    * (max_scale - min_scale)
                    + min_scale;
                let rot = glam::vec4(meta.rot_0, meta.rot_1, meta.rot_2, meta.rot_3)
                    * (max_rot - min_rot)
                    + min_rot;
                means.extend(mean.to_array());
                scales.extend(scale.to_array());
                rotations.extend(rot.to_array());
            })
            .deserialize(&mut file)?;

            let n_splats = splats.num_splats() as usize;

            let means = Tensor::from_data(TensorData::new(means, [n_splats, 3]), &device)
                + splats.means.val();
            // The encoding is just literal delta encoding in floats - nothing fancy
            // like actually considering the quaternion transform.
            let rotations = Tensor::from_data(TensorData::new(rotations, [n_splats, 4]), &device)
                + splats.rotation.val();
            let log_scales = Tensor::from_data(TensorData::new(scales, [n_splats, 3]), &device)
                + splats.log_scales.val();

            // Emit newly animated splat.
            emitter
                .emit(SplatMessage {
                    meta: ParseMetadata {
                        total_splats: count as u32,
                        up_axis,
                        frame_count,
                        current_frame: frame,
                        progress: 1.0,
                    },
                    splats: Splats::from_tensor_data(
                        means,
                        rotations,
                        log_scales,
                        splats.sh_coeffs.val(),
                        splats.raw_opacity.val(),
                    ),
                })
                .await;

            frame += 1;
        }
    }

    Ok(())
}

async fn parse_compressed_ply<T: AsyncBufRead + Unpin + 'static>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: WgpuDevice,
    mut file: serde_ply::ChunkPlyFile,
    up_axis: Option<Vec3>,
    emitter: &TryStreamEmitter<SplatMessage, SplatImportError>,
) -> Result<(), SplatImportError> {
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

    loop {
        if read_up_to(&mut reader, file.buffer_mut(), 16 * 1024 * 1024).await? == 0 {
            return Err(SplatImportError::InvalidFormat);
        }
        let Some(element) = file.current_element() else {
            return Err(SplatImportError::InvalidFormat);
        };
        // Deserialize until we're done with the chunk element.
        if element.name != "chunk" {
            break;
        }

        RowVisitor::new(|meta: QuantMeta| {
            quant_metas.push(meta);
        })
        .deserialize(&mut file)?;
    }

    let vertex = file
        .current_element()
        .ok_or(SplatImportError::InvalidFormat)?;
    if vertex.name != "vertex" {
        return Err(SplatImportError::InvalidFormat);
    }
    let total_splats = vertex.count;

    let mut means = Vec::with_capacity(total_splats);
    // Atm, unlike normal plys, these values aren't optional.
    let mut log_scales = Vec::with_capacity(total_splats);
    let mut rotations = Vec::with_capacity(total_splats);
    let mut sh_coeffs = Vec::with_capacity(total_splats);
    let mut opacity = Vec::with_capacity(total_splats);

    let update_every = total_splats.div_ceil(5);
    let mut last_update = 0;

    let mut splat_count = 0;

    let sh_vals = file
        .header()
        .expect("Must have header")
        .elem_defs
        .get(2)
        .cloned();

    loop {
        if read_up_to(&mut reader, file.buffer_mut(), 16 * 1024 * 1024).await? == 0 {
            return Err(SplatImportError::InvalidFormat);
        }
        let Some(element) = file.current_element() else {
            return Err(SplatImportError::InvalidFormat);
        };
        // Deserialize until we're done with the chunk element.
        if element.name != "vertex" {
            break;
        }

        RowVisitor::new(|splat: QuantSplat| {
            // Doing this after first reading and parsing the points is quite wasteful, but
            // we do need to advance the reader.
            if let Some(subsample) = subsample_points
                && splat_count % subsample as usize != 0
            {
                return;
            }

            let quant_data = &quant_metas[splat_count / 256];

            means.push(quant_data.mean(splat.mean));
            log_scales.push(quant_data.scale(splat.log_scale));
            rotations.push(splat.rotation);

            // Compressed ply specifies things in post-activated values. Convert to pre-activated values.
            opacity.push(inverse_sigmoid(splat.rgba.w));

            // These come in as RGB colors. Convert to base SH coeffecients.
            let sh_dc = rgb_to_sh(quant_data.color(splat.rgba.xyz()));
            sh_coeffs.extend([sh_dc.x, sh_dc.y, sh_dc.z]);

            splat_count += 1;
        })
        .deserialize(&mut file)?;

        // Occasionally send some updated splats.
        if (splat_count - last_update) >= update_every || splat_count == total_splats {
            // Leave 20% of progress for loading the SH's, just an estimate.
            let max_time = if sh_vals.is_some() { 0.8 } else { 1.0 };
            let progress = progress(splat_count, total_splats) * max_time;
            emitter
                .emit(SplatMessage {
                    meta: ParseMetadata {
                        total_splats: total_splats as u32,
                        up_axis,
                        frame_count: 0,
                        current_frame: 0,
                        progress,
                    },
                    splats: Splats::from_raw(
                        &means,
                        Some(&rotations),
                        Some(&log_scales),
                        Some(&sh_coeffs),
                        Some(&opacity),
                        &device,
                    ),
                })
                .await;
            last_update = splat_count;
        }
    }

    if let Some(sh_vals) = sh_vals {
        if sh_vals.name != "sh" {
            return Err(SplatImportError::InvalidFormat);
        }

        let sh_count = sh_vals.properties.len();
        let mut total_coeffs = Vec::with_capacity(sh_vals.count * (3 + sh_count));
        let mut splat_index = 0;

        loop {
            if read_up_to(&mut reader, file.buffer_mut(), 16 * 1024 * 1024).await? == 0
                || file.current_element().is_none()
            {
                break;
            }

            RowVisitor::new(|quant_sh: QuantSh| {
                let dc = glam::vec3(
                    sh_coeffs[splat_index * 3],
                    sh_coeffs[splat_index * 3 + 1],
                    sh_coeffs[splat_index * 3 + 2],
                );
                interleave_coeffs(dc, &quant_sh.coeffs()[..sh_count], &mut total_coeffs);
                splat_index += 1;
            })
            .deserialize(&mut file)?;
        }

        emitter
            .emit(SplatMessage {
                meta: ParseMetadata {
                    total_splats: splat_count as u32,
                    up_axis,
                    frame_count: 0,
                    current_frame: 0,
                    progress: 1.0,
                },
                splats: Splats::from_raw(
                    &means,
                    Some(&rotations),
                    Some(&log_scales),
                    Some(&total_coeffs),
                    Some(&opacity),
                    &device,
                ),
            })
            .await;
    }

    Ok(())
}
