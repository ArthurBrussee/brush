//! True training checkpoints (issue #176): everything needed to stop training
//! and later continue as if uninterrupted — splat parameters, Adam momentum,
//! and trainer counters — in a single `.safetensors` file.
//!
//! PLY exports stay plain viewable files; this is a separate, denser artifact
//! purely for resuming. All tensors are stored as f32; scalar state rides in
//! the safetensors metadata map as strings.
//!
//! Not captured (and re-derived or re-accumulated on resume):
//! - The refine-statistics window (re-accumulates within one `refine_every`).
//! - RNG state (reseeded deterministically from the seed and the resume step).
//! - The dataloader shuffle position (restarts from the seed's order).

use std::collections::HashMap;

use burn::tensor::{DType, Device, Tensor, TensorData, s};

use brush_render::bounding_box::BoundingBox;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use crate::{
    adam_scaled::MomentumState,
    config::TrainConfig,
    train::{SplatOptim, SplatTrainer},
};

/// Format version written into the metadata; bump on breaking layout changes.
const CHECKPOINT_VERSION: &str = "1";

/// Canonical checkpoint file name. A file with this name inside a dataset is
/// picked up automatically for resuming.
pub const CHECKPOINT_FILE_NAME: &str = "checkpoint.safetensors";

const META_VERSION: &str = "brush.version";
const META_NEXT_ITER: &str = "brush.next-iter";
const META_STEP_COUNT: &str = "brush.step-count";
const META_MAX_SH_DEGREE: &str = "brush.max-sh-degree";
const META_RENDER_MIP: &str = "brush.render-mip";
const META_BOUNDS: &str = "brush.bounds";
const META_ADAM_TIME_TRANSFORMS: &str = "brush.adam-time.transforms";
const META_ADAM_TIME_SH: &str = "brush.adam-time.sh-coeffs";
const META_ADAM_TIME_OPAC: &str = "brush.adam-time.opacities";
/// Free-form entry holding the training arguments the checkpoint was written
/// with, so a resume can warn when the configuration changed.
pub const META_CONFIG_ARGS: &str = "brush.config-args";

const T_TRANSFORMS: &str = "splats.transforms";
const T_SH_COEFFS: &str = "splats.sh_coeffs";
const T_OPACITIES: &str = "splats.raw_opacities";
const T_MIN_SCALE: &str = "splats.min_scale";
const T_OPT_TRANSFORMS_M1: &str = "optim.transforms.moment_1";
const T_OPT_TRANSFORMS_M2: &str = "optim.transforms.moment_2";
const T_OPT_SH_M1: &str = "optim.sh_coeffs.moment_1";
const T_OPT_SH_M2: &str = "optim.sh_coeffs.moment_2";
const T_OPT_OPAC_M1: &str = "optim.raw_opacities.moment_1";
const T_OPT_OPAC_M2: &str = "optim.raw_opacities.moment_2";

/// A checkpoint loaded back into live training state.
pub struct LoadedCheckpoint {
    pub splats: Splats,
    pub trainer: SplatTrainer,
    /// The next training iteration to run (the loop resumes at this value).
    pub next_iter: u32,
    /// The training arguments stored at save time, if any.
    pub config_args: Option<String>,
}

async fn host_data<const D: usize>(t: Tensor<D>) -> anyhow::Result<TensorData> {
    let data = t
        .into_data_async()
        .await
        .map_err(|e| anyhow::anyhow!("Reading tensor for checkpoint: {e:?}"))?;
    anyhow::ensure!(
        data.dtype == DType::F32,
        "Checkpoint tensors must be f32, got {:?}",
        data.dtype
    );
    Ok(data)
}

/// Serialize the full training state to safetensors bytes. `next_iter` is the
/// iteration the training loop should resume at (i.e. iterations completed so
/// far). `extra_metadata` entries are stored verbatim — see
/// [`META_CONFIG_ARGS`].
pub async fn checkpoint_to_bytes(
    trainer: &SplatTrainer,
    splats: &Splats,
    next_iter: u32,
    extra_metadata: &[(String, String)],
) -> anyhow::Result<Vec<u8>> {
    let mut entries: Vec<(&str, TensorData)> = vec![
        (T_TRANSFORMS, host_data(splats.transforms.val()).await?),
        (T_SH_COEFFS, host_data(splats.sh_coeffs.val()).await?),
        (T_OPACITIES, host_data(splats.raw_opacities.val()).await?),
    ];
    if let Some(min_scale) = &splats.min_scale {
        entries.push((T_MIN_SCALE, host_data(min_scale.clone()).await?));
    }

    let mut metadata = HashMap::new();
    metadata.insert(META_VERSION.to_owned(), CHECKPOINT_VERSION.to_owned());
    metadata.insert(META_NEXT_ITER.to_owned(), next_iter.to_string());
    metadata.insert(META_STEP_COUNT.to_owned(), trainer.step_count.to_string());
    metadata.insert(
        META_MAX_SH_DEGREE.to_owned(),
        trainer.max_sh_degree.to_string(),
    );
    metadata.insert(
        META_RENDER_MIP.to_owned(),
        u32::from(splats.render_mip).to_string(),
    );
    let b = trainer.bounds;
    metadata.insert(
        META_BOUNDS.to_owned(),
        format!(
            "{} {} {} {} {} {}",
            b.center.x, b.center.y, b.center.z, b.extent.x, b.extent.y, b.extent.z
        ),
    );

    if let Some(optim) = &trainer.optim {
        if let Some(m) = &optim.transforms.momentum {
            entries.push((T_OPT_TRANSFORMS_M1, host_data(m.moment_1.clone()).await?));
            entries.push((T_OPT_TRANSFORMS_M2, host_data(m.moment_2.clone()).await?));
            metadata.insert(META_ADAM_TIME_TRANSFORMS.to_owned(), m.time.to_string());
        }
        if let Some(m) = &optim.sh_coeffs.momentum {
            entries.push((T_OPT_SH_M1, host_data(m.moment_1.clone()).await?));
            entries.push((T_OPT_SH_M2, host_data(m.moment_2.clone()).await?));
            metadata.insert(META_ADAM_TIME_SH.to_owned(), m.time.to_string());
        }
        if let Some(m) = &optim.opacities.momentum {
            entries.push((T_OPT_OPAC_M1, host_data(m.moment_1.clone()).await?));
            entries.push((T_OPT_OPAC_M2, host_data(m.moment_2.clone()).await?));
            metadata.insert(META_ADAM_TIME_OPAC.to_owned(), m.time.to_string());
        }
    }

    for (key, value) in extra_metadata {
        metadata.insert(key.clone(), value.clone());
    }

    let views = entries
        .iter()
        .map(|(name, data)| {
            TensorView::new(Dtype::F32, data.shape.to_vec(), data.as_bytes())
                .map(|view| (*name, view))
                .map_err(|e| anyhow::anyhow!("Building checkpoint tensor {name}: {e:?}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    safetensors::serialize(views, Some(metadata))
        .map_err(|e| anyhow::anyhow!("Serializing checkpoint: {e:?}"))
}

fn tensor_from<const D: usize>(
    tensors: &SafeTensors,
    name: &str,
    device: &Device,
) -> anyhow::Result<Tensor<D>> {
    let view = tensors
        .tensor(name)
        .map_err(|e| anyhow::anyhow!("Checkpoint is missing tensor {name}: {e:?}"))?;
    anyhow::ensure!(
        view.dtype() == Dtype::F32,
        "Checkpoint tensor {name} must be f32, got {:?}",
        view.dtype()
    );
    let floats: Vec<f32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let data = TensorData::new(floats, view.shape().to_vec());
    Ok(Tensor::from_data(data, device))
}

fn opt_tensor_from<const D: usize>(
    tensors: &SafeTensors,
    name: &str,
    device: &Device,
) -> anyhow::Result<Option<Tensor<D>>> {
    if tensors.tensor(name).is_err() {
        return Ok(None);
    }
    tensor_from(tensors, name, device).map(Some)
}

fn parse_meta<T: std::str::FromStr>(
    metadata: &HashMap<String, String>,
    key: &str,
) -> anyhow::Result<T> {
    metadata
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Checkpoint is missing metadata entry {key}"))?
        .parse::<T>()
        .map_err(|_e| anyhow::anyhow!("Checkpoint metadata entry {key} is malformed"))
}

/// Rebuild live training state from checkpoint bytes. The trainer is seeded
/// deterministically from `seed` and the resume step, so two resumes of the
/// same checkpoint behave identically.
pub fn checkpoint_from_bytes(
    bytes: &[u8],
    config: &TrainConfig,
    device: &Device,
    seed: u64,
) -> anyhow::Result<LoadedCheckpoint> {
    let (_, meta) = SafeTensors::read_metadata(bytes)
        .map_err(|e| anyhow::anyhow!("Reading checkpoint header: {e:?}"))?;
    let metadata = meta
        .metadata()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Checkpoint has no metadata — not a brush checkpoint?"))?;

    let version = metadata
        .get(META_VERSION)
        .ok_or_else(|| anyhow::anyhow!("Checkpoint has no version — not a brush checkpoint?"))?;
    anyhow::ensure!(
        version == CHECKPOINT_VERSION,
        "Unsupported checkpoint version {version} (this build reads version {CHECKPOINT_VERSION})"
    );

    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|e| anyhow::anyhow!("Reading checkpoint tensors: {e:?}"))?;

    let next_iter: u32 = parse_meta(&metadata, META_NEXT_ITER)?;
    let step_count: u32 = parse_meta(&metadata, META_STEP_COUNT)?;
    let max_sh_degree: u32 = parse_meta(&metadata, META_MAX_SH_DEGREE)?;
    let render_mip: u32 = parse_meta(&metadata, META_RENDER_MIP)?;

    let bounds_str: String = parse_meta(&metadata, META_BOUNDS)?;
    let bounds_vals = bounds_str
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_e| anyhow::anyhow!("Checkpoint bounds are malformed"))?;
    anyhow::ensure!(
        bounds_vals.len() == 6,
        "Checkpoint bounds must have 6 values, got {}",
        bounds_vals.len()
    );
    let bounds = BoundingBox {
        center: glam::Vec3::new(bounds_vals[0], bounds_vals[1], bounds_vals[2]),
        extent: glam::Vec3::new(bounds_vals[3], bounds_vals[4], bounds_vals[5]),
    };

    // Splat parameters live on the main device; the transforms tensor is
    // split back into the pieces `from_tensor_data` expects.
    let transforms: Tensor<2> = tensor_from(&tensors, T_TRANSFORMS, device)?;
    anyhow::ensure!(
        transforms.dims()[1] == 10,
        "Checkpoint transforms must be [N, 10]"
    );
    let sh_coeffs: Tensor<3> = tensor_from(&tensors, T_SH_COEFFS, device)?;
    let raw_opacities: Tensor<1> = tensor_from(&tensors, T_OPACITIES, device)?;
    let mode = if render_mip != 0 {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };
    let mut splats = Splats::from_tensor_data(
        transforms.clone().slice(s![.., 0..3]),
        transforms.clone().slice(s![.., 3..7]),
        transforms.slice(s![.., 7..10]),
        sh_coeffs,
        raw_opacities,
        mode,
    );
    // The 3D-filter floor lives on the inner backend.
    let opt_device = device.clone().inner();
    if let Some(min_scale) = opt_tensor_from::<1>(&tensors, T_MIN_SCALE, &opt_device)? {
        splats = splats.with_min_scale(min_scale);
    }

    // Reseed deterministically but distinctly per resume point, so a resumed
    // run doesn't replay the exact noise sequence from step 0.
    let resume_seed = seed.wrapping_add(u64::from(next_iter).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut trainer = SplatTrainer::new_seeded(config, device, bounds, resume_seed);
    trainer.step_count = step_count;
    trainer.max_sh_degree = max_sh_degree;

    // Optimizer momentum, when the checkpoint was taken after at least one
    // step per parameter. The LR scalings are rebuilt from the config.
    let mut optim = SplatOptim::new(config, splats.sh_degree(), &opt_device);
    let mut any_momentum = false;
    if let Some(m1) = opt_tensor_from::<2>(&tensors, T_OPT_TRANSFORMS_M1, &opt_device)? {
        let m2 = tensor_from::<2>(&tensors, T_OPT_TRANSFORMS_M2, &opt_device)?;
        let time: usize = parse_meta(&metadata, META_ADAM_TIME_TRANSFORMS)?;
        optim.transforms.momentum = Some(MomentumState {
            moment_1: m1,
            moment_2: m2,
            time,
        });
        any_momentum = true;
    }
    if let Some(m1) = opt_tensor_from::<3>(&tensors, T_OPT_SH_M1, &opt_device)? {
        let m2 = tensor_from::<3>(&tensors, T_OPT_SH_M2, &opt_device)?;
        let time: usize = parse_meta(&metadata, META_ADAM_TIME_SH)?;
        optim.sh_coeffs.momentum = Some(MomentumState {
            moment_1: m1,
            moment_2: m2,
            time,
        });
        any_momentum = true;
    }
    if let Some(m1) = opt_tensor_from::<1>(&tensors, T_OPT_OPAC_M1, &opt_device)? {
        let m2 = tensor_from::<1>(&tensors, T_OPT_OPAC_M2, &opt_device)?;
        let time: usize = parse_meta(&metadata, META_ADAM_TIME_OPAC)?;
        optim.opacities.momentum = Some(MomentumState {
            moment_1: m1,
            moment_2: m2,
            time,
        });
        any_momentum = true;
    }
    if any_momentum {
        trainer.optim = Some(optim);
    }

    Ok(LoadedCheckpoint {
        splats,
        trainer,
        next_iter,
        config_args: metadata.get(META_CONFIG_ARGS).cloned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data_async()
            .await
            .expect("tensor readback")
            .try_into_vec::<f32>()
            .expect("f32 tensor")
    }

    fn test_splats(device: &Device, n: usize) -> Splats {
        let coeffs = 4; // SH degree 1
        Splats::from_tensor_data(
            Tensor::random([n, 3], burn::tensor::Distribution::Default, device),
            Tensor::random([n, 4], burn::tensor::Distribution::Default, device),
            Tensor::random([n, 3], burn::tensor::Distribution::Default, device),
            Tensor::random([n, coeffs, 3], burn::tensor::Distribution::Default, device),
            Tensor::random([n], burn::tensor::Distribution::Default, device),
            SplatRenderMode::Default,
        )
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn checkpoint_round_trips_full_state() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let opt_device = device.clone().inner();
        let config = TrainConfig::default();
        let n = 8;
        let splats = test_splats(&device, n).with_min_scale(Tensor::random(
            [n],
            burn::tensor::Distribution::Default,
            &opt_device,
        ));

        let bounds = BoundingBox {
            center: glam::Vec3::new(1.0, 2.0, 3.0),
            extent: glam::Vec3::new(4.0, 5.0, 6.0),
        };
        let mut trainer = SplatTrainer::new_seeded(&config, &device, bounds, 7);
        trainer.step_count = 1234;
        trainer.max_sh_degree = 1;

        // Fabricate momentum as a few steps of training would leave it.
        let mut optim = SplatOptim::new(&config, splats.sh_degree(), &opt_device);
        let dist = burn::tensor::Distribution::Default;
        optim.transforms.momentum = Some(MomentumState {
            moment_1: Tensor::random([n, 10], dist, &opt_device),
            moment_2: Tensor::random([n, 10], dist, &opt_device),
            time: 1234,
        });
        optim.sh_coeffs.momentum = Some(MomentumState {
            moment_1: Tensor::random([n, 4, 3], dist, &opt_device),
            // Reduced second moment: size-1 trailing dims.
            moment_2: Tensor::random([n, 1, 1], dist, &opt_device),
            time: 1234,
        });
        optim.opacities.momentum = Some(MomentumState {
            moment_1: Tensor::random([n], dist, &opt_device),
            moment_2: Tensor::random([n], dist, &opt_device),
            time: 1200,
        });
        trainer.optim = Some(optim);

        let bytes = checkpoint_to_bytes(
            &trainer,
            &splats,
            5000,
            &[(
                META_CONFIG_ARGS.to_owned(),
                "--total-train-iters 30000".to_owned(),
            )],
        )
        .await
        .expect("serialize checkpoint");

        let loaded =
            checkpoint_from_bytes(&bytes, &config, &device, 7).expect("deserialize checkpoint");

        assert_eq!(loaded.next_iter, 5000, "next_iter survives");
        assert_eq!(
            loaded.config_args.as_deref(),
            Some("--total-train-iters 30000"),
            "config args survive"
        );
        assert_eq!(loaded.trainer.step_count, 1234, "step count survives");
        assert_eq!(loaded.trainer.max_sh_degree, 1, "sh degree survives");
        let b = loaded.trainer.bounds;
        assert!(
            (b.center - bounds.center).length() < 1e-6
                && (b.extent - bounds.extent).length() < 1e-6,
            "bounds survive"
        );

        // Splat params round-trip bit-exactly (f32 end to end).
        assert_eq!(
            to_vec(splats.transforms.val()).await,
            to_vec(loaded.splats.transforms.val()).await,
            "transforms survive"
        );
        assert_eq!(
            to_vec(splats.sh_coeffs.val()).await,
            to_vec(loaded.splats.sh_coeffs.val()).await,
            "sh coeffs survive"
        );
        assert_eq!(
            to_vec(splats.raw_opacities.val()).await,
            to_vec(loaded.splats.raw_opacities.val()).await,
            "opacities survive"
        );
        assert_eq!(
            to_vec(splats.min_scale.clone().expect("floor saved")).await,
            to_vec(loaded.splats.min_scale.clone().expect("floor loaded")).await,
            "min-scale floor survives"
        );

        let orig = trainer.optim.as_ref().expect("optim saved");
        let restored = loaded.trainer.optim.as_ref().expect("optim loaded");
        let om = orig.transforms.momentum.as_ref().expect("momentum saved");
        let rm = restored
            .transforms
            .momentum
            .as_ref()
            .expect("momentum loaded");
        assert_eq!(om.time, rm.time, "adam time survives");
        assert_eq!(
            to_vec(om.moment_1.clone()).await,
            to_vec(rm.moment_1.clone()).await,
            "transforms moment 1 survives"
        );
        assert_eq!(
            to_vec(om.moment_2.clone()).await,
            to_vec(rm.moment_2.clone()).await,
            "transforms moment 2 survives"
        );
        let om = orig.sh_coeffs.momentum.as_ref().expect("momentum saved");
        let rm = restored
            .sh_coeffs
            .momentum
            .as_ref()
            .expect("momentum loaded");
        assert_eq!(
            rm.moment_2.dims(),
            [n, 1, 1],
            "reduced second moment keeps its shape"
        );
        assert_eq!(
            to_vec(om.moment_1.clone()).await,
            to_vec(rm.moment_1.clone()).await,
            "sh moment 1 survives"
        );
        let om = orig.opacities.momentum.as_ref().expect("momentum saved");
        let rm = restored
            .opacities
            .momentum
            .as_ref()
            .expect("momentum loaded");
        assert_eq!(om.time, rm.time, "opacity adam time survives");
        // The sh LR scaling is rebuilt from config, not stored.
        assert!(
            restored.sh_coeffs.scaling.is_some(),
            "sh scaling is rebuilt on load"
        );
        assert!(
            restored.sh_coeffs.reduce_moment_2,
            "reduce flag is rebuilt on load"
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn checkpoint_without_momentum_loads_fresh_optim() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let config = TrainConfig::default();
        let splats = test_splats(&device, 4);
        let bounds = BoundingBox {
            center: glam::Vec3::ZERO,
            extent: glam::Vec3::ONE,
        };
        let trainer = SplatTrainer::new_seeded(&config, &device, bounds, 7);

        let bytes = checkpoint_to_bytes(&trainer, &splats, 0, &[])
            .await
            .expect("serialize checkpoint");
        let loaded = checkpoint_from_bytes(&bytes, &config, &device, 7).expect("deserialize");

        assert!(
            loaded.trainer.optim.is_none(),
            "no momentum saved means the optimizer re-initializes lazily"
        );
        assert!(
            loaded.splats.min_scale.is_none(),
            "no floor saved means no floor loaded"
        );
        assert_eq!(loaded.next_iter, 0, "iter zero round-trips");
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn garbage_bytes_are_rejected() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let config = TrainConfig::default();
        // Not a safetensors file at all.
        let result = checkpoint_from_bytes(&[1, 2, 3, 4], &config, &device, 7);
        assert!(result.is_err(), "garbage must not load");
    }
}
