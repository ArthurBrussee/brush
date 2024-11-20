use std::collections::HashSet;

use async_fn_stream::try_fn_stream;
use brush_render::{render::rgb_to_sh, Backend};
use burn::tensor::{Tensor, TensorData};
use glam::{Quat, Vec3};
use ply_rs::{
    parser::Parser,
    ply::{ElementDef, Header, Property, PropertyAccess},
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, BufReader};
use tokio_stream::Stream;
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
        GaussianData {
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

        if let Property::Float(value) = property {
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
        } else if let Property::UChar(value) = property {
            match ascii {
                b"red" => self.sh_dc[0] = rgb_to_sh(value as f32 / 255.0),
                b"green" => self.sh_dc[1] = rgb_to_sh(value as f32 / 255.0),
                b"blue" => self.sh_dc[2] = rgb_to_sh(value as f32 / 255.0),
                _ => {}
            }
        } else {
            return;
        };
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

pub fn load_splat_from_ply<T: AsyncRead + Unpin + 'static, B: Backend>(
    reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
) -> impl Stream<Item = Result<Splats<B>>> + 'static {
    // set up a reader, in this case a file.
    let mut reader = BufReader::new(reader);

    let update_every = 25000;
    let _span = trace_span!("Read splats").entered();
    let gaussian_parser = Parser::<GaussianData>::new();

    try_fn_stream(|emitter| async move {
        let header = gaussian_parser.read_header(&mut reader).await?;
        let mut final_splat = None;

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
                    Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                }

                for i in 0..element.count {
                    // Ocassionally yield.
                    if i % 500 == 0 {
                        tokio::task::yield_now().await;
                    }

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        let splats = Splats::from_raw(
                            means.clone(),
                            rotations.clone(),
                            log_scales.clone(),
                            sh_coeffs.clone(),
                            opacity.clone(),
                            &device,
                        );

                        emitter.emit(splats).await;
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
                        rotation.push(splat.rotation.normalize());
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

                let splats =
                    Splats::from_raw(means, rotations, log_scales, sh_coeffs, opacity, &device);
                final_splat = Some(splats.clone());
                emitter.emit(splats).await;
            } else if element.name.starts_with("delta_vertex_") {
                let Some(splats) = final_splat.clone() else {
                    anyhow::bail!("Need to read base splat first.");
                };

                for i in 0..element.count {
                    // Ocassionally yield.
                    if i % 500 == 0 {
                        tokio::task::yield_now().await;
                    }
                    let splat =
                        decode_splat(&mut reader, &gaussian_parser, &header, element).await?;
                    // Let's only animate transforms for now.
                    means.push(splat.means);
                    if let Some(log_scales) = log_scales.as_mut() {
                        log_scales.push(splat.log_scale);
                    }
                    if let Some(rotation) = rotations.as_mut() {
                        rotation.push(splat.rotation);
                    }
                    // Don't emit any intermediate states as it looks strange to have a torn state.
                }

                let n_splats = splats.num_splats();
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
                emitter.emit(new_splat).await;
            }
        }

        Ok(())
    })
}
