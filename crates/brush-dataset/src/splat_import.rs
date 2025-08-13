use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::Splats;
use brush_render::{MainBackend, gaussian_splats::inverse_sigmoid, sh::rgb_to_sh};
use brush_vfs::{DynStream, SendNotWasm};
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};
use glam::{Vec3, Vec4, Vec4Swizzles};
use serde::Deserialize;
use serde::de::{DeserializeSeed, Error};
use serde_ply::{DeserializeError, RowVisitor};
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;

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

async fn read_chunk<T: AsyncRead + Unpin>(
    reader: &mut T,
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

pub fn load_splat_from_ply<T: AsyncRead + SendNotWasm + Unpin + 'static>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: WgpuDevice,
) -> impl DynStream<Result<SplatMessage, DeserializeError>> {
    try_fn_stream(|emitter| async move {
        // TODO: Just make chunk ply take in data and try to get a header? Simpler maybe.
        let mut file = serde_ply::PlyChunkedReader::new();
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
            return Err(DeserializeError::custom("Unknown format"));
        };

        let subsample = subsample_points.unwrap_or(1) as usize;

        match ply_type {
            PlyFormat::Ply => {
                parse_ply(reader, subsample, device, &mut file, up_axis, &emitter).await?;
            }
            PlyFormat::Brush4DCompressed => {
                parse_delta_ply(reader, subsample, device, file, up_axis, &emitter).await?;
            }
            PlyFormat::SuperSplatCompressed => {
                parse_compressed_ply(reader, subsample, device, file, up_axis, &emitter).await?;
            }
        }

        Ok(())
    })
}

fn progress(index: usize, len: usize) -> f32 {
    ((index + 1) as f32) / len as f32
}

async fn parse_ply<T: AsyncRead + Unpin>(
    mut reader: T,
    subsample: usize,
    device: WgpuDevice,
    file: &mut serde_ply::PlyChunkedReader,
    up_axis: Option<Vec3>,
    emitter: &TryStreamEmitter<SplatMessage, DeserializeError>,
) -> Result<Splats<MainBackend>, DeserializeError> {
    let header = file.header().expect("Must have header");
    let vertex = header
        .get_element("vertex")
        .ok_or(DeserializeError::custom("Unknown format"))?;

    let max_splats = vertex.count / subsample;

    let mut means = Vec::with_capacity(max_splats * 3);
    let mut log_scales = vertex
        .has_property("scale_0")
        .then(|| Vec::with_capacity(max_splats * 3));
    let mut rotations = vertex
        .has_property("rot_0")
        .then(|| Vec::with_capacity(max_splats * 4));
    let mut opacity = vertex
        .has_property("opacity")
        .then(|| Vec::with_capacity(max_splats));
    let sh_count = vertex
        .properties
        .iter()
        .filter(|x| {
            x.name.starts_with("f_rest_")
                || x.name.starts_with("f_dc_")
                || matches!(x.name.as_str(), "r" | "g" | "b" | "red" | "green" | "blue")
        })
        .count();
    let mut coeffs = (sh_count > 0).then(|| Vec::with_capacity(max_splats * sh_count));

    let update_every = max_splats.div_ceil(5);
    let mut row_index: usize = 0;
    let mut last_update = 0;

    loop {
        read_chunk(&mut reader, file.buffer_mut()).await?;

        RowVisitor::new(|mut gauss: PlyGaussian| {
            row_index += 1;
            if row_index % subsample != 0 {
                return;
            }

            means.extend([gauss.x, gauss.y, gauss.z]);

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
                scales.extend([gauss.scale_0, gauss.scale_1, gauss.scale_2]);
            }
            if let Some(rotation) = &mut rotations {
                rotation.extend([gauss.rot_0, gauss.rot_1, gauss.rot_2, gauss.rot_3]);
            }
            if let Some(opacity) = &mut opacity {
                opacity.push(gauss.opacity);
            }
        })
        .deserialize(&mut *file)?;

        if row_index - last_update > update_every || row_index == max_splats {
            let splats = Splats::from_raw(
                means.clone(),
                rotations.clone(),
                log_scales.clone(),
                coeffs.clone(),
                opacity.clone(),
                &device,
            );

            emitter
                .emit(SplatMessage {
                    meta: ParseMetadata {
                        total_splats: max_splats as u32,
                        up_axis,
                        frame_count: 0,
                        current_frame: 0,
                        progress: progress(row_index, max_splats),
                    },
                    splats: splats.clone(),
                })
                .await;

            last_update = row_index;

            if row_index == max_splats {
                return Ok(splats);
            }
        }
    }
}

async fn parse_delta_ply<T: AsyncRead + Unpin + 'static>(
    mut reader: T,
    subsample: usize,
    device: WgpuDevice,
    mut file: serde_ply::PlyChunkedReader,
    up_axis: Option<Vec3>,
    emitter: &TryStreamEmitter<SplatMessage, DeserializeError>,
) -> Result<(), DeserializeError> {
    let splats = parse_ply(
        &mut reader,
        subsample,
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

    fn de_quant<'de, D>(deserializer: D) -> Result<f32, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Dequant;
        impl<'de> serde::de::Visitor<'de> for Dequant {
            type Value = f32;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a quantized value or a float")
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
        deserializer.deserialize_any(Dequant)
    }

    #[derive(Deserialize, Default)]
    struct Frame {
        #[serde(deserialize_with = "de_quant")]
        x: f32,
        #[serde(deserialize_with = "de_quant")]
        y: f32,
        #[serde(deserialize_with = "de_quant")]
        z: f32,
        #[serde(deserialize_with = "de_quant")]
        scale_0: f32,
        #[serde(deserialize_with = "de_quant")]
        scale_1: f32,
        #[serde(deserialize_with = "de_quant")]
        scale_2: f32,
        #[serde(deserialize_with = "de_quant")]
        rot_0: f32,
        #[serde(deserialize_with = "de_quant")]
        rot_1: f32,
        #[serde(deserialize_with = "de_quant")]
        rot_2: f32,
        #[serde(deserialize_with = "de_quant")]
        rot_3: f32,
    }

    // Leave unscaled if there are no meta_delta_ frames present.
    let mut min_mean = Vec3::ZERO;
    let mut max_mean = Vec3::ONE;

    let mut min_scale = Vec3::ZERO;
    let mut max_scale = Vec3::ONE;

    let mut min_rot = Vec4::ZERO;
    let mut max_rot = Vec4::ONE;
    let mut row_count = 0;

    while let Some(element) = file.current_element().cloned() {
        read_chunk(&mut reader, file.buffer_mut()).await?;

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
                row_count += 1;
                if row_count % subsample != 0 {
                    return;
                }

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

async fn parse_compressed_ply<T: AsyncRead + Unpin + 'static>(
    mut reader: T,
    subsample: usize,
    device: WgpuDevice,
    mut file: serde_ply::PlyChunkedReader,
    up_axis: Option<Vec3>,
    emitter: &TryStreamEmitter<SplatMessage, DeserializeError>,
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
    let max_splats = vertex.count / subsample;

    let mut means = Vec::with_capacity(max_splats * 3);
    // Atm, unlike normal plys, these values aren't optional.
    let mut log_scales = Vec::with_capacity(max_splats * 3);
    let mut rotations = Vec::with_capacity(max_splats * 4);
    let mut sh_coeffs = Vec::with_capacity(max_splats * 3);
    let mut opacity = Vec::with_capacity(max_splats);

    let update_every = max_splats.div_ceil(5);
    let mut last_update = 0;

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
            // These come in as RGB colors. Convert to base SH coeffecients.
            let sh_dc = rgb_to_sh(quant_data.color(splat.rgba.xyz()));
            sh_coeffs.extend([sh_dc.x, sh_dc.y, sh_dc.z]);
        })
        .deserialize(&mut file)?;

        // Occasionally send some updated splats.
        if (row_count - last_update) >= update_every || row_count == max_splats {
            // Leave 20% of progress for loading the SH's, just an estimate.
            let max_time = if sh_vals.is_some() { 0.8 } else { 1.0 };
            let progress = progress(row_count, max_splats) * max_time;
            emitter
                .emit(SplatMessage {
                    meta: ParseMetadata {
                        total_splats: max_splats as u32,
                        up_axis,
                        frame_count: 0,
                        current_frame: 0,
                        progress,
                    },
                    splats: Splats::from_raw(
                        means.clone(),
                        Some(rotations.clone()),
                        Some(log_scales.clone()),
                        Some(sh_coeffs.clone()),
                        Some(opacity.clone()),
                        &device,
                    ),
                })
                .await;
            last_update = row_count;
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
                interleave_coeffs(dc, &quant_sh.coeffs()[..sh_count], &mut total_coeffs);
                splat_index += 1;
            })
            .deserialize(&mut file)?;
        }

        emitter
            .emit(SplatMessage {
                meta: ParseMetadata {
                    total_splats: means.len() as u32,
                    up_axis,
                    frame_count: 0,
                    current_frame: 0,
                    progress: 1.0,
                },
                splats: Splats::from_raw(
                    means,
                    Some(rotations),
                    Some(log_scales),
                    Some(total_coeffs),
                    Some(opacity),
                    &device,
                ),
            })
            .await;
    }

    Ok(())
}
