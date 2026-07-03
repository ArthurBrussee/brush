//! `DiG` (DINO-embedded Gaussians) training state.
//!
//! Port of the `DiG` model from Robot See Robot Do (kerrj/dig): every
//! gaussian carries a learnable feature (dim set by `--dino-feature-dim`,
//! default 64), decoded by a
//! small shared no-bias MLP to the cached `DINOv2` (PCA-reduced) feature
//! space and supervised with MSE against rendered feature maps. Kept
//! separate from [`brush_render::gaussian_splats::Splats`] so the export
//! / viewer / FFI surfaces stay untouched — the trainer owns this state
//! and keeps it in lockstep with the splats through refine.

use burn::{
    Tensor,
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    optim::{Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, Int, TensorData, activation::relu},
};

use crate::adam_scaled::{AdamScaled, AdamScaledConfig};

/// MLP hidden width (fixed by the reference architecture).
const MLP_HIDDEN: usize = 64;
/// Feature/MLP LR decay horizon: `lr` → `lr_end` over this many steps,
/// then held (matches the reference `ExponentialDecayScheduler`).
const DIG_LR_DECAY_STEPS: f64 = 6000.0;
/// Warmup before the neighbor feature-variance regularizer kicks in.
pub const NN_REG_START_STEP: u32 = 1000;
pub const NN_K: usize = 3;

pub fn dig_lr(step: u32, lr_start: f64, lr_end: f64) -> f64 {
    let t = (f64::from(step) / DIG_LR_DECAY_STEPS).min(1.0);
    lr_start * (lr_end / lr_start).powf(t)
}

/// The learnable `DiG` parameters: per-gaussian features + shared decoder.
#[derive(Module, Debug)]
pub struct DigModule {
    /// `[N, feature_dim]` per-gaussian feature table.
    pub features: Param<Tensor<2>>,
    l1: Linear,
    l2: Linear,
    l3: Linear,
    l4: Linear,
}

impl DigModule {
    pub fn new(num_splats: u32, feature_dim: usize, out_dim: usize, device: &Device) -> Self {
        let features = Tensor::random(
            [num_splats as usize, feature_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let linear = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out).with_bias(false).init(device)
        };
        Self {
            features: Param::initialized(ParamId::new(), features.require_grad()),
            l1: linear(feature_dim, MLP_HIDDEN),
            l2: linear(MLP_HIDDEN, MLP_HIDDEN),
            l3: linear(MLP_HIDDEN, MLP_HIDDEN),
            l4: linear(MLP_HIDDEN, out_dim),
        }
    }

    /// The per-gaussian stored feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.features.dims()[1]
    }

    /// Decode rendered (alpha-normalized) raw features to the GT DINO
    /// space: `Linear → ReLU ×3 → Linear`, all bias-free.
    pub fn decode<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let x = relu(self.l1.forward(x));
        let x = relu(self.l2.forward(x));
        let x = relu(self.l3.forward(x));
        self.l4.forward(x)
    }

    pub fn mlp_param_ids(&self) -> Vec<ParamId> {
        vec![
            self.l1.weight.id,
            self.l2.weight.id,
            self.l3.weight.id,
            self.l4.weight.id,
        ]
    }

    pub async fn export(&self) -> DigExport {
        async fn read2(t: Tensor<2>) -> ([usize; 2], Vec<f32>) {
            let dims = t.dims();
            let data = t
                .into_data_async()
                .await
                .expect("Failed to read DiG tensor")
                .to_vec()
                .expect("Failed to read DiG tensor");
            (dims, data)
        }
        let ([num_splats, feat_dim], features) = read2(self.features.val()).await;
        let mut mlp = Vec::new();
        for (name, layer) in [
            ("l1", &self.l1),
            ("l2", &self.l2),
            ("l3", &self.l3),
            ("l4", &self.l4),
        ] {
            let (dims, data) = read2(layer.weight.val()).await;
            mlp.push((name.to_owned(), dims, data));
        }
        DigExport {
            features,
            num_splats,
            feat_dim,
            mlp,
        }
    }
}

/// CPU-side snapshot of the `DiG` state for export. Feature rows are in
/// the splats' current row order (matching a PLY exported at the same
/// step).
pub struct DigExport {
    /// `[num_splats * feat_dim]` row-major.
    pub features: Vec<f32>,
    pub num_splats: usize,
    pub feat_dim: usize,
    /// Decoder layers in forward order: (name, `[d_in, d_out]`, row-major weights).
    pub mlp: Vec<(String, [usize; 2], Vec<f32>)>,
}

pub(crate) type DigOptimizer = OptimizerAdaptor<AdamScaled, DigModule>;

pub(crate) fn create_dig_optimizer() -> DigOptimizer {
    AdamScaledConfig::new().with_epsilon(1e-15).init()
}

/// Trainer-owned `DiG` state: the module, its optimizer, and a cached
/// 3-nearest-neighbor index for the feature-smoothness regularizer.
pub struct DigTrainState {
    pub module: DigModule,
    pub(crate) optim: DigOptimizer,
    /// `[N, NN_K]` neighbor indices; invalidated whenever the splat count
    /// changes (refine).
    nn_indices: Option<Tensor<2, Int>>,
}

impl DigTrainState {
    pub fn new(num_splats: u32, feature_dim: usize, out_dim: usize, device: &Device) -> Self {
        Self {
            module: DigModule::new(num_splats, feature_dim, out_dim, device),
            optim: create_dig_optimizer(),
            nn_indices: None,
        }
    }

    pub fn invalidate_neighbors(&mut self) {
        self.nn_indices = None;
    }

    /// Remap after a prune: keep only `valid_inds` rows of the feature
    /// table and its Adam state. Every splat-count change must go through
    /// [`Self::keep`] / [`Self::split`] so the table can't silently desync
    /// from the splats.
    pub fn keep(&mut self, valid_inds: &Tensor<1, Int>) {
        self.module.features = self
            .module
            .features
            .clone()
            .map(|x| x.select(0, valid_inds.clone()));
        let mut record = self.optim.to_record();
        if record.contains_key(&self.module.features.id) {
            crate::train::map_opt(self.module.features.id, &mut record, &|x: Tensor<2>| {
                x.select(0, valid_inds.clone())
            });
            self.optim = create_dig_optimizer().load_record(record);
        }
        self.invalidate_neighbors();
    }

    /// Remap after a split: children copy the parent's feature vector, and
    /// (like every other refined param) both halves restart with zero Adam
    /// moments. `refine_inds_opt` is `refine_inds` on the optimizer device.
    pub fn split(
        &mut self,
        refine_inds: &Tensor<1, Int>,
        refine_inds_opt: &Tensor<1, Int>,
        opt_device: &Device,
    ) {
        let refine_count = refine_inds.dims()[0];
        let cur_feats = self.module.features.val().select(0, refine_inds.clone());
        self.module.features = self
            .module
            .features
            .clone()
            .map(|x| Tensor::cat(vec![x, cur_feats], 0));
        let mut record = self.optim.to_record();
        if record.contains_key(&self.module.features.id) {
            let inds_opt = refine_inds_opt.clone();
            let opt_device = opt_device.clone();
            crate::train::map_opt(self.module.features.id, &mut record, &move |x: Tensor<
                2,
            >| {
                let d1 = x.dims()[1];
                let neg_parent = -x.clone().select(0, inds_opt.clone());
                let inds: Tensor<2, Int> = inds_opt.clone().unsqueeze_dim(1).repeat_dim(1, d1);
                let x = x.scatter(0, inds, neg_parent, burn::tensor::IndexingUpdateOp::Add);
                Tensor::cat(vec![x, Tensor::zeros([refine_count, d1], &opt_device)], 0)
            });
            self.optim = create_dig_optimizer().load_record(record);
        }
        self.invalidate_neighbors();
    }

    /// Neighbor indices for the feature-variance regularizer, recomputed
    /// (CPU-side, approximate grid search) when stale. "Stale" means the
    /// splat *count* changed (refine/prune) — means drift continuously
    /// between refines, so the neighbor set lags geometry by up to
    /// `refine_every` steps. That matches the reference (periodic KNN),
    /// and the self-neighbor fallback keeps mismatches a zero-variance
    /// no-op; do not assume neighbors track positions step-to-step.
    pub async fn neighbor_indices(&mut self, means: &Tensor<2>, device: &Device) -> Tensor<2, Int> {
        let n = means.dims()[0];
        if let Some(cached) = &self.nn_indices
            && cached.dims()[0] == n
        {
            return cached.clone();
        }
        let pos: Vec<f32> = means
            .clone()
            .into_data_async()
            .await
            .expect("Failed to read means")
            .to_vec()
            .expect("Failed to read means");
        let inds = grid_knn(&pos, NN_K);
        let inds = Tensor::from_data(TensorData::new(inds, [n, NN_K]), device);
        self.nn_indices = Some(inds.clone());
        inds
    }
}

/// Approximate k-nearest-neighbors on a uniform grid hash. Points are
/// bucketed at a cell size that puts a handful of points per cell; each
/// query scans its 3×3×3 cell neighborhood, expanding once if that finds
/// fewer than `k`. Falls back to the point's own index when isolated —
/// self-neighbors contribute zero variance, which is the right no-op for
/// the regularizer this feeds.
fn grid_knn(pos: &[f32], k: usize) -> Vec<i64> {
    let n = pos.len() / 3;
    if n == 0 {
        return Vec::new();
    }
    let p = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];

    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        let q = p(i);
        for d in 0..3 {
            if q[d].is_finite() {
                mn[d] = mn[d].min(q[d]);
                mx[d] = mx[d].max(q[d]);
            }
        }
    }
    let extent = (0..3)
        .map(|d| (mx[d] - mn[d]).max(1e-6))
        .fold(0.0, f32::max);
    // ~4 points per cell on average.
    let cells_per_axis = ((n as f32 / 4.0).cbrt().ceil() as i64).max(1);
    let cell = extent / cells_per_axis as f32;

    let key = |q: [f32; 3]| -> (i64, i64, i64) {
        (
            ((q[0] - mn[0]) / cell) as i64,
            ((q[1] - mn[1]) / cell) as i64,
            ((q[2] - mn[2]) / cell) as i64,
        )
    };

    let mut grid: hashbrown::HashMap<(i64, i64, i64), Vec<u32>> = hashbrown::HashMap::new();
    for i in 0..n {
        let q = p(i);
        if q.iter().all(|v| v.is_finite()) {
            grid.entry(key(q)).or_default().push(i as u32);
        }
    }

    let mut out = Vec::with_capacity(n * k);
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(64);
    for i in 0..n {
        let q = p(i);
        best.clear();
        for radius in 1..=2i64 {
            let (cx, cy, cz) = key(q);
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        // Only the new shell on the second pass.
                        if radius > 1 && dx.abs() < radius && dy.abs() < radius && dz.abs() < radius
                        {
                            continue;
                        }
                        if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in ids {
                                if j as usize == i {
                                    continue;
                                }
                                let r = p(j as usize);
                                let d2 = (0..3).map(|d| (q[d] - r[d]).powi(2)).sum::<f32>();
                                best.push((d2, j));
                            }
                        }
                    }
                }
            }
            if best.len() >= k {
                break;
            }
        }
        best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for slot in 0..k {
            let idx = best.get(slot).map_or(i as i64, |&(_, j)| i64::from(j));
            out.push(idx);
        }
    }
    out
}
