use async_fn_stream::try_fn_stream;
use brush_render::{render::sh_coeffs_for_degree, Backend};
use burn::{
    module::{Param, ParamId},
    tensor::{Tensor, TensorData},
};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio_stream::Stream;
use tracing::trace_span;

use anyhow::{Context, Result};
use brush_render::gaussian_splats::Splats;

pub(crate) struct GaussianData {
    pub(crate) means: [f32; 3],
    pub(crate) scale: [f32; 3],
    pub(crate) opacity: f32,
    pub(crate) rotation: [f32; 4],
    pub(crate) sh_dc: [f32; 3],
    // NB: This is in the inria format, aka [channels, coeffs]
    // not [coeffs, channels].
    pub(crate) sh_coeffs_rest: Vec<f32>,
}

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData {
            means: [0.0; 3],
            scale: [0.0; 3],
            opacity: 0.0,
            rotation: [0.0; 4],
            sh_dc: [0.0, 0.0, 0.0],
            sh_coeffs_rest: Vec::new(),
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        if let Property::Float(v) = property {
            match key {
                "x" => self.means[0] = v,
                "y" => self.means[1] = v,
                "z" => self.means[2] = v,
                "scale_0" => self.scale[0] = v,
                "scale_1" => self.scale[1] = v,
                "scale_2" => self.scale[2] = v,
                "opacity" => self.opacity = v,
                "rot_0" => self.rotation[0] = v,
                "rot_1" => self.rotation[1] = v,
                "rot_2" => self.rotation[2] = v,
                "rot_3" => self.rotation[3] = v,
                "f_dc_0" => self.sh_dc[0] = v,
                "f_dc_1" => self.sh_dc[1] = v,
                "f_dc_2" => self.sh_dc[2] = v,
                _ if key.starts_with("f_rest_") => {
                    if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                        if idx >= self.sh_coeffs_rest.len() as u32 {
                            self.sh_coeffs_rest.resize(idx as usize + 1, 0.0);
                        }
                        self.sh_coeffs_rest[idx as usize] = v;
                    }
                }
                _ => (),
            }
        }
    }

    fn get_float(&self, key: &str) -> Option<f32> {
        match key {
            "x" => Some(self.means[0]),
            "y" => Some(self.means[1]),
            "z" => Some(self.means[2]),
            "scale_0" => Some(self.scale[0]),
            "scale_1" => Some(self.scale[1]),
            "scale_2" => Some(self.scale[2]),
            "opacity" => Some(self.opacity),
            "rot_0" => Some(self.rotation[0]),
            "rot_1" => Some(self.rotation[1]),
            "rot_2" => Some(self.rotation[2]),
            "rot_3" => Some(self.rotation[3]),
            "f_dc_0" => Some(self.sh_dc[0]),
            "f_dc_1" => Some(self.sh_dc[1]),
            "f_dc_2" => Some(self.sh_dc[2]),
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

fn update_splats<B: Backend>(
    splats: &mut Option<Splats<B>>,
    means: Vec<f32>,
    sh_coeffs: Vec<f32>,
    rotation: Vec<f32>,
    raw_opacities: Vec<f32>,
    log_scales: Vec<f32>,
    device: &B::Device,
) {
    let n_splats = means.len() / 3;
    let n_coeffs = sh_coeffs.len() / n_splats;

    let means = Tensor::from_data(TensorData::new(means, [n_splats, 3]), device).require_grad();
    let sh_coeffs = Tensor::from_data(
        TensorData::new(sh_coeffs, [n_splats, n_coeffs / 3, 3]),
        device,
    )
    .require_grad();
    let rotations =
        Tensor::from_data(TensorData::new(rotation, [n_splats, 4]), device).require_grad();
    let raw_opacities =
        Tensor::from_data(TensorData::new(raw_opacities, [n_splats]), device).require_grad();
    let log_scales =
        Tensor::from_data(TensorData::new(log_scales, [n_splats, 3]), device).require_grad();

    if let Some(splats) = splats.as_mut() {
        Splats::map_param(&mut splats.means, |x| {
            Tensor::cat(vec![x, means.clone()], 0)
        });
        Splats::map_param(&mut splats.rotation, |x| {
            Tensor::cat(vec![x, rotations.clone()], 0)
        });
        Splats::map_param(&mut splats.sh_coeffs, |x| {
            Tensor::cat(vec![x, sh_coeffs.clone()], 0)
        });
        Splats::map_param(&mut splats.raw_opacity, |x| {
            Tensor::cat(vec![x, raw_opacities.clone()], 0)
        });
        Splats::map_param(&mut splats.log_scales, |x| {
            Tensor::cat(vec![x, log_scales.clone()], 0)
        });
        splats.norm_rotations();
    } else {
        let mut init = Splats {
            means: Param::initialized(ParamId::new(), means),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs),
            rotation: Param::initialized(ParamId::new(), rotations),
            raw_opacity: Param::initialized(ParamId::new(), raw_opacities),
            log_scales: Param::initialized(ParamId::new(), log_scales),
            xys_dummy: Tensor::zeros([n_splats, 2], device).require_grad(),
        };
        init.norm_rotations();
        // Create a new splat instance if it hasn't been initialzized yet.
        *splats = Some(init);
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

pub fn load_splat_from_ply<T: AsyncRead + Unpin + 'static, B: Backend>(
    reader: T,
    device: B::Device,
) -> impl Stream<Item = Result<Splats<B>>> + 'static {
    // set up a reader, in this case a file.
    let mut reader = BufReader::new(reader);
    let mut splats: Option<Splats<B>> = None;

    let update_every = 10000;
    let _span = trace_span!("Read splats").entered();
    let gaussian_parser = Parser::<GaussianData>::new();

    try_fn_stream(|emitter| async move {
        let header = gaussian_parser.read_header(&mut reader).await?;

        for element in &header.elements {
            if element.name == "vertex" {
                let min_props = [
                    "x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1",
                    "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
                ];

                if !min_props
                    .iter()
                    .all(|p| element.properties.iter().any(|x| &x.name == p))
                {
                    Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                }

                let n_sh_coeffs = (1 + element
                    .properties
                    .iter()
                    .filter_map(|x| {
                        x.name
                            .strip_prefix("f_rest_")
                            .and_then(|x| x.parse::<u32>().ok())
                    })
                    .max()
                    .unwrap_or(0)) as usize;

                let mut means = Vec::with_capacity(update_every * 3);
                let mut sh_coeffs = Vec::with_capacity(update_every * n_sh_coeffs);
                let mut rotation = Vec::with_capacity(update_every * 4);
                let mut opacity = Vec::with_capacity(update_every);
                let mut scales = Vec::with_capacity(update_every * 3);

                for i in 0..element.count {
                    let splat = match header.encoding {
                        ply_rs::ply::Encoding::Ascii => {
                            let mut line = String::new();
                            reader.read_line(&mut line).await?;
                            gaussian_parser.read_ascii_element(&line, element)?
                        }
                        ply_rs::ply::Encoding::BinaryBigEndian => {
                            gaussian_parser
                                .read_big_endian_element(&mut reader, element)
                                .await?
                        }
                        ply_rs::ply::Encoding::BinaryLittleEndian => {
                            gaussian_parser
                                .read_little_endian_element(&mut reader, element)
                                .await?
                        }
                    };

                    let mut sh_coeffs_interleaved =
                        interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest);

                    // Limit the number of imported SH channels for now.
                    let max_sh_len = sh_coeffs_for_degree(3) as usize * 3;

                    if sh_coeffs_interleaved.len() > max_sh_len {
                        sh_coeffs_interleaved.truncate(max_sh_len);
                    }

                    means.extend(splat.means);
                    sh_coeffs.extend(sh_coeffs_interleaved);
                    rotation.extend(splat.rotation);
                    opacity.push(splat.opacity);
                    scales.extend(splat.scale);

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        update_splats(
                            &mut splats,
                            std::mem::take(&mut means),
                            std::mem::take(&mut sh_coeffs),
                            std::mem::take(&mut rotation),
                            std::mem::take(&mut opacity),
                            std::mem::take(&mut scales),
                            &device,
                        );

                        emitter
                            .emit(splats.clone().context("Failed to update splats")?)
                            .await;
                    }

                    // Ocassionally yield.
                    if i % 250 == 0 {
                        tokio::task::yield_now().await;
                    }
                }

                update_splats(
                    &mut splats,
                    means,
                    sh_coeffs,
                    rotation,
                    opacity,
                    scales,
                    &device,
                );

                if let Some(splats) = splats.as_ref() {
                    if splats.num_splats() == 0 {
                        Err(anyhow::anyhow!("No splats found"))?;
                    }
                }

                emitter
                    .emit(splats.clone().context("Invalid ply file.")?)
                    .await;
            }
        }

        Ok(())
    })
}
