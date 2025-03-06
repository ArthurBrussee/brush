use std::collections::HashSet;

use async_fn_stream::try_fn_stream;
use brush_render::{gaussian_splats::inverse_sigmoid, sh::rgb_to_sh};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use glam::{Quat, Vec3, Vec4};
use ply_rs::{
    parser::Parser,
    ply::{DefaultElement, ElementDef, Encoding, Header, Property, PropertyAccess},
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, BufReader};
use tokio_stream::{Stream, StreamExt};
use tokio_with_wasm::alias as tokio_wasm;
use tracing::trace_span;

use anyhow::{Context, Result};
use brush_render::gaussian_splats::Splats;

use crate::parsed_gaussian::ParsedGaussian;

pub struct ParseMetadata {
    pub up_axis: Option<Vec3>,
    pub total_splats: u32,
    pub frame_count: u32,
    pub current_frame: u32,
}

pub struct SplatMessage<B: Backend> {
    pub meta: ParseMetadata,
    pub splats: Splats<B>,
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

async fn parse_elem<T: AsyncBufRead + Unpin + 'static, E: PropertyAccess>(
    reader: &mut T,
    parser: &Parser<E>,
    encoding: Encoding,
    element: &ElementDef,
) -> tokio::io::Result<E> {
    match encoding {
        Encoding::Ascii => {
            let mut ascii_line = String::new();
            reader.read_line(&mut ascii_line).await?;
            let elem = parser.read_ascii_element(&ascii_line, element)?;
            ascii_line.clear();
            Ok(elem)
        }
        Encoding::BinaryBigEndian => parser.read_big_endian_element(reader, element).await,
        Encoding::BinaryLittleEndian => parser.read_little_endian_element(reader, element).await,
    }
}

/// Call this in a loop to yield every once in a while.
async fn try_yield(i: usize) {
    if i % 50000 == 0 {
        tokio_wasm::task::yield_now().await;
    }
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
        let header = Parser::<DefaultElement>::new()
            .read_header(&mut reader)
            .await?;

        // Parse some metadata.
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

        // Check whether there is a vertex header that has at least XYZ.
        let has_vertex = header.elements.iter().any(|el| el.name == "vertex");

        let ply_type = if has_vertex && header.elements.first().is_some_and(|el| el.name == "chunk")
        {
            PlyFormat::SuperSplatCompressed
        } else if has_vertex && header.elements.iter().any(|el| el.name == "delta_vertex_") {
            PlyFormat::Brush4DCompressed
        } else if has_vertex {
            PlyFormat::Ply
        } else {
            anyhow::bail!("Couldn't decide format of Ply file. Unknown Ply format.")
        };

        match ply_type {
            PlyFormat::Ply => {
                let mut stream =
                    std::pin::pin!(parse_ply(reader, subsample_points, device, header, up_axis));
                while let Some(splat) = stream.next().await {
                    emitter.emit(splat?).await;
                }
            }
            PlyFormat::Brush4DCompressed => {
                let mut stream = std::pin::pin!(parse_delta_ply(
                    reader,
                    subsample_points,
                    device,
                    header,
                    up_axis
                ));
                while let Some(splat) = stream.next().await {
                    emitter.emit(splat?).await;
                }
            }
            PlyFormat::SuperSplatCompressed => {
                let mut stream = std::pin::pin!(parse_compressed_ply(
                    reader,
                    subsample_points,
                    device,
                    header,
                    up_axis
                ));
                while let Some(splat) = stream.next().await {
                    emitter.emit(splat?).await;
                }
            }
        };

        Ok(())
    })
}

fn parse_ply<T: AsyncBufRead + Unpin + 'static, B: Backend>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
    header: Header,
    up_axis: Option<Vec3>,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    try_fn_stream(|emitter| async move {
        let vertex = header.elements.first().context("No elements in header")?;
        if vertex.name != "vertex" {
            anyhow::bail!("First element must be 'vertex'")
        }

        let parser = Parser::<ParsedGaussian<false>>::new();

        let properties: HashSet<_> = vertex.properties.iter().map(|x| x.name.clone()).collect();
        let mut means = Vec::with_capacity(vertex.count);
        let mut log_scales = properties
            .contains("scale_0")
            .then(|| Vec::with_capacity(vertex.count));
        let mut rotations = properties
            .contains("rot_0")
            .then(|| Vec::with_capacity(vertex.count));
        let mut sh_coeffs = (properties.contains("f_dc_0") || properties.contains("red"))
            .then(|| Vec::with_capacity(vertex.count * 24));
        let mut opacity = properties
            .contains("opacity")
            .then(|| Vec::with_capacity(vertex.count));

        let update_every = vertex.count.div_ceil(20);

        let mut last_update = 0;

        for i in 0..vertex.count {
            try_yield(i).await;

            // Doing this after first reading and parsing the points is quite wasteful, but
            // we do need to advance the reader.
            if let Some(subsample) = subsample_points {
                if i % subsample as usize != 0 {
                    continue;
                }
            }

            let splat = parse_elem(&mut reader, &parser, header.encoding, vertex).await?;

            if !splat.is_finite() {
                continue;
            }

            means.push(splat.mean);
            if let Some(scales) = &mut log_scales {
                scales.push(splat.log_scale);
            }
            if let Some(rotation) = &mut rotations {
                rotation.push(splat.rotation);
            }
            if let Some(opacity) = &mut opacity {
                opacity.push(splat.opacity);
            }
            if let Some(sh_coeffs) = &mut sh_coeffs {
                interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest, sh_coeffs);
            }

            if (i - last_update) >= update_every || i == vertex.count - 1 {
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
                        meta: ParseMetadata {
                            total_splats: vertex.count as u32,
                            up_axis,
                            frame_count: 0,
                            current_frame: 0,
                        },
                        splats,
                    })
                    .await;

                last_update = i;
            }
        }

        Ok(())
    })
}

fn parse_compressed_ply<T: AsyncBufRead + Unpin + 'static, B: Backend>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
    header: Header,
    up_axis: Option<Vec3>,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    #[derive(Default)]
    struct MinMax {
        min: Vec3,
        max: Vec3,
    }

    impl MinMax {
        fn dequant(&self, raw: Vec3) -> Vec3 {
            self.min + raw * (self.max - self.min)
        }
    }

    #[derive(Default)]
    struct QuantMeta {
        mean: MinMax,
        scale: MinMax,
        color: MinMax,
    }

    impl PropertyAccess for QuantMeta {
        fn new() -> Self {
            Self::default()
        }

        fn set_property(&mut self, key: &str, property: Property) {
            let ascii = key.as_bytes();
            let Property::Float(val) = property else {
                return;
            };
            match ascii {
                b"min_x" => self.mean.min.x = val,
                b"min_y" => self.mean.min.y = val,
                b"min_z" => self.mean.min.z = val,

                b"max_x" => self.mean.max.x = val,
                b"max_y" => self.mean.max.y = val,
                b"max_z" => self.mean.max.z = val,

                b"min_scale_x" => self.scale.min.x = val,
                b"min_scale_y" => self.scale.min.y = val,
                b"min_scale_z" => self.scale.min.z = val,

                b"max_scale_x" => self.scale.max.x = val,
                b"max_scale_y" => self.scale.max.y = val,
                b"max_scale_z" => self.scale.max.z = val,

                b"min_r" => self.color.min.x = val,
                b"min_g" => self.color.min.y = val,
                b"min_b" => self.color.min.z = val,

                b"max_r" => self.color.max.x = val,
                b"max_g" => self.color.max.y = val,
                b"max_b" => self.color.max.z = val,
                _ => {}
            }
        }
    }

    try_fn_stream(|emitter| async move {
        let quant_elem = header
            .elements
            .first()
            .context("Not enough elements in header")?;
        if quant_elem.name != "chunk" {
            anyhow::bail!("First element should be chunk compression metadata!");
        }

        let parser = Parser::<QuantMeta>::new();
        let mut quant_metas = vec![];
        for i in 0..quant_elem.count {
            try_yield(i).await;
            let quant_meta = parse_elem(&mut reader, &parser, header.encoding, quant_elem).await?;
            quant_metas.push(quant_meta);
        }

        let vertex = header
            .elements
            .get(1)
            .context("Not enough elements in header")?;
        if vertex.name != "vertex" {
            anyhow::bail!("Second element should be vertex compression metadata!");
        }

        let parser = Parser::<ParsedGaussian<true>>::new();
        let mut means = Vec::with_capacity(vertex.count);
        // Atm, unlike normal plys, these values aren't optional.
        let mut log_scales = Vec::with_capacity(vertex.count);
        let mut rotations = Vec::with_capacity(vertex.count);
        let mut sh_coeffs = Vec::with_capacity(vertex.count);
        let mut opacity = Vec::with_capacity(vertex.count);

        let update_every = vertex.count.div_ceil(20);
        let mut last_update = 0;

        let mut valid = vec![true; vertex.count];

        for i in 0..vertex.count {
            // Occasionally yield.
            try_yield(i).await;

            // Doing this after first reading and parsing the points is quite wasteful, but
            // we do need to advance the reader.
            if let Some(subsample) = subsample_points {
                if i % subsample as usize != 0 {
                    continue;
                }
            }

            let quant_data = quant_metas
                .get(i / 256)
                .context("not enough quantization data to parse ply")?;

            let splat = parse_elem(&mut reader, &parser, header.encoding, vertex).await?;

            // Don't add invalid splats.
            if !splat.is_finite() {
                valid[i] = false;
                continue;
            }

            means.push(quant_data.mean.dequant(splat.mean));

            log_scales.push(quant_data.scale.dequant(splat.log_scale));
            rotations.push(splat.rotation);

            // Compressed ply specifies things in post-activated values. Convert to pre-activated values.
            opacity.push(inverse_sigmoid(splat.opacity));

            // These come in as RGB colors. Convert to base SH coeffecients.
            let sh_dc = rgb_to_sh(quant_data.color.dequant(splat.sh_dc));
            sh_coeffs.extend([sh_dc.x, sh_dc.y, sh_dc.z]);

            // Occasionally send some updated splats.
            if (i - last_update) >= update_every || i == vertex.count - 1 {
                emitter
                    .emit(SplatMessage {
                        meta: ParseMetadata {
                            total_splats: vertex.count as u32,
                            up_axis,
                            frame_count: 0,
                            current_frame: 0,
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
                last_update = i;
            }
        }

        if let Some(sh_vals) = header.elements.get(2) {
            // Bit of a hack - use the unquantized parser as that handles SH values. Really we don't need
            // the entire splat parser though.
            let parser = Parser::<ParsedGaussian<false>>::new();

            if sh_vals.name != "sh" {
                anyhow::bail!("Second element should be SH compression metadata!");
            }

            let mut splat_index = 0;

            let mut total_coeffs = vec![];
            for i in 0..sh_vals.count {
                try_yield(i).await;

                if !valid[i] {
                    continue;
                }

                // Parse a splat - though nb only SH values will be used.
                let mut splat = parse_elem(&mut reader, &parser, header.encoding, sh_vals).await?;
                for coeff in &mut splat.sh_coeffs_rest {
                    *coeff = 8.0 * (*coeff - 0.5);
                }

                let dc = glam::vec3(
                    sh_coeffs[splat_index * 3],
                    sh_coeffs[splat_index * 3 + 1],
                    sh_coeffs[splat_index * 3 + 2],
                );
                interleave_coeffs(dc, &splat.sh_coeffs_rest, &mut total_coeffs);
                splat_index += 1;
            }

            emitter
                .emit(SplatMessage {
                    meta: ParseMetadata {
                        total_splats: vertex.count as u32,
                        up_axis,
                        frame_count: 0,
                        current_frame: 0,
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
    })
}

fn parse_delta_ply<T: AsyncBufRead + Unpin + 'static, B: Backend>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
    header: Header,
    up_axis: Option<Vec3>,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    try_fn_stream(|emitter| async move {
        let parser = Parser::<ParsedGaussian<false>>::new();

        // Check for frame count.
        let frame_count = header
            .elements
            .iter()
            .filter(|e| e.name.starts_with("delta_vertex_"))
            .count() as u32;

        let mut final_splat = None;
        let mut frame = 0;

        #[derive(Debug)]
        struct QuantMeta {
            mean: Vec3,
            rotation: Vec4,
            scale: Vec3,
        }

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

            let mut means = Vec::with_capacity(element.count);
            let mut log_scales = properties
                .contains("scale_0")
                .then(|| Vec::with_capacity(element.count));
            let mut rotations = properties
                .contains("rot_0")
                .then(|| Vec::with_capacity(element.count));
            let mut sh_coeffs = (properties.contains("f_dc_0") || properties.contains("red"))
                .then(|| Vec::with_capacity(element.count * 24));
            let mut opacity = properties
                .contains("opacity")
                .then(|| Vec::with_capacity(element.count));

            if element.name == "vertex" {
                let update_every = element.count.div_ceil(20);

                for i in 0..element.count {
                    try_yield(i).await;

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        emitter
                            .emit(SplatMessage {
                                meta: ParseMetadata {
                                    total_splats: element.count as u32,
                                    up_axis,
                                    frame_count,
                                    current_frame: frame,
                                },
                                splats: Splats::from_raw(
                                    &means,
                                    rotations.as_deref(),
                                    log_scales.as_deref(),
                                    sh_coeffs.as_deref(),
                                    opacity.as_deref(),
                                    &device,
                                ),
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

                    let splat = parse_elem(&mut reader, &parser, header.encoding, element).await?;

                    means.push(splat.mean);
                    if let Some(scales) = &mut log_scales {
                        scales.push(splat.log_scale);
                    }
                    if let Some(rotation) = &mut rotations {
                        rotation.push(splat.rotation);
                    }
                    if let Some(opacity) = &mut opacity {
                        opacity.push(splat.opacity);
                    }
                    if let Some(sh_coeffs) = &mut sh_coeffs {
                        interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest, sh_coeffs);
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
                        meta: ParseMetadata {
                            total_splats: element.count as u32,
                            up_axis,
                            frame_count,
                            current_frame: frame,
                        },
                        splats,
                    })
                    .await;
            } else if element.name.starts_with("meta_delta_min_") {
                let splat = parse_elem(&mut reader, &parser, header.encoding, element).await?;
                meta_min.mean = splat.mean;
                meta_min.rotation = splat.rotation.into();
                meta_min.scale = splat.log_scale;
            } else if element.name.starts_with("meta_delta_max_") {
                let splat = parse_elem(&mut reader, &parser, header.encoding, element).await?;
                meta_max.mean = splat.mean;
                meta_max.rotation = splat.rotation.into();
                meta_max.scale = splat.log_scale;
            } else if element.name.starts_with("delta_vertex_") {
                let Some(splats) = final_splat.clone() else {
                    anyhow::bail!("Need to read base splat first.");
                };

                for i in 0..element.count {
                    try_yield(i).await;
                    // The splat we decode is normed to 0-1 (if quantized), so rescale to
                    // actual values afterwards.
                    let splat_enc =
                        parse_elem(&mut reader, &parser, header.encoding, element).await?;

                    // Let's only animate transforms for now.
                    means.push(splat_enc.mean * (meta_max.mean - meta_min.mean) + meta_min.mean);

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

                // Emit newly animated splat.
                emitter
                    .emit(SplatMessage {
                        meta: ParseMetadata {
                            total_splats: element.count as u32,
                            up_axis,
                            frame_count,
                            current_frame: frame,
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
    })
}
