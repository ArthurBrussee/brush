use std::collections::HashSet;

use async_fn_stream::try_fn_stream;
use brush_render::render::rgb_to_sh;
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use glam::{Quat, Vec3, Vec4};
use ply_rs::{
    parser::Parser,
    ply::{ElementDef, Header, Property, PropertyAccess},
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, BufReader};
use tokio_stream::Stream;
use tokio_with_wasm::alias as tokio_wasm;
use tracing::trace_span;

use anyhow::Result;
use brush_render::gaussian_splats::Splats;

pub(crate) struct GaussianData {
    pub(crate) means: Vec3,
    pub(crate) log_scale: Vec3,
    pub(crate) opacity: f32,
    pub(crate) rotation: Quat,
    pub(crate) sh_dc: [f32; 3],
    // NB: This is in the inria format, aka [channels, coeffs]
    // not [coeffs, channels].
    pub(crate) sh_coeffs_rest: Vec<f32>,
}

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        Self {
            means: Vec3::ZERO,
            log_scale: Vec3::ZERO,
            opacity: 0.0,
            rotation: Quat::IDENTITY,
            sh_dc: [0.0, 0.0, 0.0],
            sh_coeffs_rest: Vec::new(),
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        let ascii = key.as_bytes();

        let mut value = if let Property::Float(value) = property {
            value
        } else if let Property::UChar(value) = property {
            (value as f32) / (u8::MAX as f32)
        } else if let Property::UShort(value) = property {
            (value as f32) / (u16::MAX as f32)
        } else {
            return;
        };

        if value.is_nan() || value.is_infinite() {
            log::warn!("Invalid numbers in imported splat, defaulting to 0");
            value = 0.0;
        }

        match ascii {
            b"x" => self.means[0] = value,
            b"y" => self.means[1] = value,
            b"z" => self.means[2] = value,
            b"scale_0" => self.log_scale[0] = value,
            b"scale_1" => self.log_scale[1] = value,
            b"scale_2" => self.log_scale[2] = value,
            b"opacity" => self.opacity = value,
            b"rot_0" => self.rotation.w = value,
            b"rot_1" => self.rotation.x = value,
            b"rot_2" => self.rotation.y = value,
            b"rot_3" => self.rotation.z = value,
            b"f_dc_0" => self.sh_dc[0] = value,
            b"f_dc_1" => self.sh_dc[1] = value,
            b"f_dc_2" => self.sh_dc[2] = value,
            b"red" => self.sh_dc[0] = rgb_to_sh(value),
            b"green" => self.sh_dc[1] = rgb_to_sh(value),
            b"blue" => self.sh_dc[2] = rgb_to_sh(value),
            _ if key.starts_with("f_rest_") => {
                if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                    if idx >= self.sh_coeffs_rest.len() as u32 {
                        self.sh_coeffs_rest.resize(idx as usize + 1, 0.0);
                    }
                    self.sh_coeffs_rest[idx as usize] = value;
                }
            }
            _ => (),
        }
    }

    fn get_float(&self, key: &str) -> Option<f32> {
        let ascii = key.as_bytes();

        match ascii {
            b"x" => Some(self.means[0]),
            b"y" => Some(self.means[1]),
            b"z" => Some(self.means[2]),
            b"scale_0" => Some(self.log_scale[0]),
            b"scale_1" => Some(self.log_scale[1]),
            b"scale_2" => Some(self.log_scale[2]),
            b"opacity" => Some(self.opacity),
            b"rot_0" => Some(self.rotation.w),
            b"rot_1" => Some(self.rotation.x),
            b"rot_2" => Some(self.rotation.y),
            b"rot_3" => Some(self.rotation.z),
            b"f_dc_0" => Some(self.sh_dc[0]),
            b"f_dc_1" => Some(self.sh_dc[1]),
            b"f_dc_2" => Some(self.sh_dc[2]),
            _ if key.starts_with("f_rest_") => {
                if let Ok(idx) = key["f_rest_".len()..].parse::<usize>() {
                    self.sh_coeffs_rest.get(idx).copied()
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

fn interleave_coeffs(sh_dc: [f32; 3], sh_rest: &[f32]) -> Vec<f32> {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;
    let mut result = Vec::with_capacity(sh_rest.len() + 3);
    result.extend(sh_dc);

    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
    result
}

async fn decode_splat<T: AsyncBufRead + Unpin + 'static>(
    reader: &mut T,
    parser: &Parser<GaussianData>,
    header: &Header,
    element: &ElementDef,
) -> tokio::io::Result<GaussianData> {
    match header.encoding {
        ply_rs::ply::Encoding::Ascii => {
            let mut ascii_line = String::new();
            reader.read_line(&mut ascii_line).await?;
            let elem = parser.read_ascii_element(&ascii_line, element)?;
            ascii_line.clear();
            Ok(elem)
        }
        ply_rs::ply::Encoding::BinaryBigEndian => {
            parser.read_big_endian_element(reader, element).await
        }
        ply_rs::ply::Encoding::BinaryLittleEndian => {
            parser.read_little_endian_element(reader, element).await
        }
    }
}

pub struct SplatMetadata {
    pub up_axis: Option<Vec3>,
    pub total_splats: u32,
    pub frame_count: u32,
    pub current_frame: u32,
}

pub struct SplatMessage<B: Backend> {
    pub meta: SplatMetadata,
    pub splats: Splats<B>,
}

#[derive(Debug)]
struct QuantMeta {
    mean: Vec3,
    rotation: Vec4,
    scale: Vec3,
}

pub fn load_splat_from_ply<T: AsyncRead + Unpin + 'static, B: Backend>(
    reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    // set up a reader, in this case a file.
    let mut reader = BufReader::new(reader);

    let _span = trace_span!("Read splats").entered();

    try_fn_stream(|emitter| async move {
        let gaussian_parser = Parser::<GaussianData>::new();

        let header = gaussian_parser.read_header(&mut reader).await?;

        let up_axis = header
            .comments
            .iter()
            .filter_map(|c| match c.to_lowercase().strip_prefix("vertical axis: ") {
                Some("x") => Some(Vec3::X),
                Some("y") => Some(Vec3::NEG_Y),
                Some("z") => Some(Vec3::Z),
                _ => None,
            })
            .next_back();

        let frame_count = header
            .elements
            .iter()
            .filter(|e| e.name.starts_with("delta_vertex_"))
            .count() as u32;

        let mut final_splat = None;
        let mut frame = 0;

        let mut meta_min = QuantMeta {
            mean: Vec3::ZERO,
            rotation: Vec4::ZERO,
            scale: Vec3::ZERO,
        };
        let mut meta_max = QuantMeta {
            mean: Vec3::ONE,
            rotation: Vec4::ONE,
            scale: Vec3::ONE,
        };
        for element in &header.elements {
            let properties: HashSet<_> =
                element.properties.iter().map(|x| x.name.clone()).collect();

            let n_sh_coeffs = (3 + element
                .properties
                .iter()
                .filter_map(|x| {
                    x.name
                        .strip_prefix("f_rest_")
                        .and_then(|x| x.parse::<u32>().ok())
                })
                .max()
                .unwrap_or(0)) as usize;

            let mut means = Vec::with_capacity(element.count);
            let mut log_scales = properties
                .contains("scale_0")
                .then(|| Vec::with_capacity(element.count));
            let mut rotations = properties
                .contains("rot_0")
                .then(|| Vec::with_capacity(element.count));
            let mut sh_coeffs = (properties.contains("f_dc_0") || properties.contains("red"))
                .then(|| Vec::with_capacity(element.count * n_sh_coeffs));
            let mut opacity = properties
                .contains("opacity")
                .then(|| Vec::with_capacity(element.count));

            if element.name == "vertex" {
                if ["x", "y", "z"].into_iter().any(|p| !properties.contains(p)) {
                    anyhow::bail!("Invalid splat ply. Missing properties!");
                }

                let update_every = element.count.div_ceil(25);

                for i in 0..element.count {
                    // Occasionally yield.
                    if i % 500 == 0 {
                        tokio_wasm::task::yield_now().await;
                    }

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        let splats = Splats::from_raw(
                            &means,
                            rotations.as_deref(),
                            log_scales.as_deref(),
                            sh_coeffs.as_deref(),
                            opacity.as_deref(),
                            &device,
                        );

                        emitter
                            .emit(SplatMessage {
                                meta: SplatMetadata {
                                    total_splats: element.count as u32,
                                    up_axis,
                                    frame_count,
                                    current_frame: frame,
                                },
                                splats,
                            })
                            .await;
                    }

                    // Doing this after first reading and parsing the points is quite wasteful, but
                    // we do need to advance the reader.
                    if let Some(subsample) = subsample_points {
                        if i % subsample as usize != 0 {
                            continue;
                        }
                    }

                    let splat =
                        decode_splat(&mut reader, &gaussian_parser, &header, element).await?;

                    means.push(splat.means);
                    if let Some(scales) = log_scales.as_mut() {
                        scales.push(splat.log_scale);
                    }
                    if let Some(rotation) = rotations.as_mut() {
                        // Normalize rotations, bail if 0.
                        let vec: Vec4 = splat.rotation.into();
                        let vec = vec.try_normalize().map_or(Quat::IDENTITY, Quat::from_vec4);
                        rotation.push(vec);
                    }
                    if let Some(opacity) = opacity.as_mut() {
                        opacity.push(splat.opacity);
                    }
                    if let Some(sh_coeffs) = sh_coeffs.as_mut() {
                        let sh_coeffs_interleaved =
                            interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest);
                        sh_coeffs.extend(sh_coeffs_interleaved);
                    }
                }

                let splats = Splats::from_raw(
                    &means,
                    rotations.as_deref(),
                    log_scales.as_deref(),
                    sh_coeffs.as_deref(),
                    opacity.as_deref(),
                    &device,
                );
                final_splat = Some(splats.clone());
                emitter
                    .emit(SplatMessage {
                        meta: SplatMetadata {
                            total_splats: element.count as u32,
                            up_axis,
                            frame_count,
                            current_frame: frame,
                        },
                        splats,
                    })
                    .await;
            } else if element.name.starts_with("meta_delta_min_") {
                let splat = decode_splat(&mut reader, &gaussian_parser, &header, element).await?;
                meta_min.mean = splat.means;
                meta_min.rotation = splat.rotation.into();
                meta_min.scale = splat.log_scale;
            } else if element.name.starts_with("meta_delta_max_") {
                let splat = decode_splat(&mut reader, &gaussian_parser, &header, element).await?;
                meta_max.mean = splat.means;
                meta_max.rotation = splat.rotation.into();
                meta_max.scale = splat.log_scale;
            } else if element.name.starts_with("delta_vertex_") {
                let Some(splats) = final_splat.clone() else {
                    anyhow::bail!("Need to read base splat first.");
                };

                for i in 0..element.count {
                    // Occasionally yield.
                    if i % 500 == 0 {
                        tokio_wasm::task::yield_now().await;
                    }
                    // The splat we decode is normed to 0-1 (if quantized), so rescale to
                    // actual values afterwards.
                    let splat_enc =
                        decode_splat(&mut reader, &gaussian_parser, &header, element).await?;

                    // Let's only animate transforms for now.
                    means.push(splat_enc.means * (meta_max.mean - meta_min.mean) + meta_min.mean);

                    if let Some(rotation) = rotations.as_mut() {
                        let val: Vec4 = splat_enc.rotation.into();
                        let val = val * (meta_max.rotation - meta_min.rotation) + meta_min.rotation;
                        rotation.push(Quat::from_vec4(val));
                    }

                    if let Some(log_scales) = log_scales.as_mut() {
                        log_scales.push(
                            splat_enc.log_scale * (meta_max.scale - meta_min.scale)
                                + meta_min.scale,
                        );
                    }
                    // Don't emit any intermediate states as it looks strange to have a torn state.
                }

                let n_splats = splats.num_splats() as usize;
                let means_tensor: Vec<f32> = means.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
                let means =
                    Tensor::from_data(TensorData::new(means_tensor, [n_splats, 3]), &device)
                        + splats.means.val();

                // The encoding is just delta encoding in floats - nothing fancy
                // like actually considering the quaternion transform.
                let rotations = if let Some(rotations) = rotations {
                    let rotations: Vec<f32> = rotations
                        .into_iter()
                        .flat_map(|v| [v.w, v.x, v.y, v.z])
                        .collect();
                    Tensor::from_data(TensorData::new(rotations, [n_splats, 4]), &device)
                        + splats.rotation.val()
                } else {
                    splats.rotation.val()
                };

                let log_scales = if let Some(log_scales) = log_scales {
                    let log_scales: Vec<f32> = log_scales
                        .into_iter()
                        .flat_map(|v| [v.x, v.y, v.z])
                        .collect();
                    Tensor::from_data(TensorData::new(log_scales, [n_splats, 3]), &device)
                        + splats.log_scales.val()
                } else {
                    splats.log_scales.val()
                };

                let mut new_splat = Splats::from_tensor_data(
                    means,
                    rotations,
                    log_scales,
                    splats.sh_coeffs.val(),
                    splats.raw_opacity.val(),
                );
                new_splat.norm_rotations();

                // Emit newly animated splat.
                emitter
                    .emit(SplatMessage {
                        meta: SplatMetadata {
                            total_splats: element.count as u32,
                            up_axis,
                            frame_count,
                            current_frame: frame,
                        },
                        splats: new_splat,
                    })
                    .await;

                frame += 1;
            }
        }

        Ok(())
    })
}
