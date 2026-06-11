//! Training-side state for the appearance models: per-step lifting onto the
//! autodiff graph and Adam updates.
//!
//! Two pipelines share this state:
//!
//! - **Hybrid** (`ppisp_grid`, the recommended mode): per-camera PPISP
//!   vignetting → per-view PPISP grid (exposure + optional color/CRF payload)
//!   → optional per-camera CRF.
//! - **Legacy** comparison modes: full per-frame PPISP and/or the affine
//!   bilateral grid.
//!
//! The grids deliberately bypass Burn's module/optimizer plumbing: only the
//! *active view's* `[1, C, L, H, W]` slice is lifted as an autodiff leaf
//! each step, so the gradient, the regularisers and the optimizer update
//! all run on one view's grid. A naive full-tensor setup (what the
//! reference implementations do) allocates an `[N, ...]` gradient and
//! Adam-updates every view's grid every step, which scales per-step cost
//! with the dataset size.
//!
//! The sparse update is *dense-Adam equivalent*: each visit consolidates
//! every step a dense Adam would have applied to the slice since the
//! view's last visit (see `GridTrainState::step`). Plain
//! `torch.optim.SparseAdam` semantics (per-view bias-correction
//! timesteps) are deliberately NOT used: their bias correction makes
//! every visit a `~lr`-sized step, capping a view's total movement at
//! `visits * lr` — on a large dataset (say 1.5k views / 12k iters, ~8
//! visits per view) that is ~100x less learning than the references,
//! whose dense Adam amplifies rare-visit steps by `~1/sqrt(v_hat)` as
//! the second moment decays toward zero between visits.
//!
//! PPISP params are tiny (a few dozen floats per frame/camera), so they get
//! a plain dense Adam.

use burn::tensor::{Device, Gradients, Tensor, s};

use crate::bilagrid::bilagrid_apply_inner;
use crate::ppisp::{PpispModel, PpispStages, ppisp_apply_inner};
use crate::ppisp_grid::{GridPayload, ppisp_grid_apply, ppisp_grid_apply_inner};
use crate::{AppearanceConfig, BilagridModel, GradSubsample, bilagrid_apply, bilagrid_tv_loss};
use brush_render::burn_glue::{detach_autodiff, lift_to_autodiff};

const ADAM_EPS: f64 = 1e-15;

/// Warmup steps for the appearance LR schedules (`LichtFeld` / PPISP
/// reference defaults), with linear warmup from 1% and exponential decay
/// toward 1% of the base LR at `total_iters`.
const BILAGRID_LR_WARMUP: u32 = 1000;
const PPISP_LR_WARMUP: u32 = 500;
const LR_START_FACTOR: f64 = 0.01;
const LR_FINAL_FACTOR: f64 = 0.01;

/// EMA decay length (in steps) for the grid mean-to-identity anchor;
/// roughly one epoch on typical datasets (spirulae uses the same horizon
/// for its color-shift regulariser).
const MEAN_EMA_PERIOD: f64 = 750.0;

/// One dense Adam update. `t` is the 1-based timestep for bias correction.
fn adam_update<const D: usize>(
    param: Tensor<D>,
    grad: Tensor<D>,
    m1: Tensor<D>,
    m2: Tensor<D>,
    t: i32,
    lr: f64,
    betas: (f64, f64),
) -> (Tensor<D>, Tensor<D>, Tensor<D>) {
    let (b1, b2) = betas;
    let m1 = m1 * b1 + grad.clone() * (1.0 - b1);
    let m2 = m2 * b2 + grad.powi_scalar(2) * (1.0 - b2);
    let m1_hat = m1.clone() / (1.0 - b1.powi(t));
    let m2_hat = m2.clone() / (1.0 - b2.powi(t));
    let param = param - m1_hat / (m2_hat.sqrt() + ADAM_EPS) * lr;
    (param, m1, m2)
}

/// Warmup + exponential decay (see module docs).
fn scheduled_lr(step: u32, base: f64, warmup: u32, total_iters: u32) -> f64 {
    crate::warmup_exp_lr(
        step,
        base,
        warmup,
        LR_START_FACTOR,
        LR_FINAL_FACTOR,
        total_iters,
    )
}

/// Per-view grids plus sparse-Adam state, all on the inner (non-autodiff)
/// backend. Tensor fields are `Option` only so updates can take ownership
/// and write back without cloning the handle (a second live handle would
/// force `slice_assign` out-of-place over the full tensor).
struct GridTrainState {
    grids: Option<Tensor<5>>,
    m1: Option<Tensor<5>>,
    m2: Option<Tensor<5>>,
    /// Global step of each view's last visit (-1 = never), for the
    /// dense-equivalent moment decay over the gap.
    last_step: Vec<i64>,
    betas: (f64, f64),
    /// EMA of per-view payload-channel means `[C]`, the anchor for the
    /// mean-to-identity regulariser (zero-identity payloads only).
    mean_ema: Option<Tensor<1>>,
}

impl GridTrainState {
    /// Affine 3x4 grid (identity-initialised diagonal).
    fn new_affine(
        num_views: usize,
        dims: (usize, usize, usize),
        betas: (f64, f64),
        device: &Device,
    ) -> Self {
        let (gx, gy, guidance) = dims;
        let grids = BilagridModel::new(num_views, gx, gy, guidance, device)
            .grids
            .into_value();
        let zeros = || Tensor::zeros(grids.dims(), device);
        Self {
            m1: Some(zeros()),
            m2: Some(zeros()),
            grids: Some(grids),
            last_step: vec![-1; num_views],
            betas,
            mean_ema: None,
        }
    }

    /// PPISP-payload grid (all-zeros = identity).
    fn new_payload(
        num_views: usize,
        dims: (usize, usize, usize),
        payload: GridPayload,
        betas: (f64, f64),
        mean_reg: bool,
        device: &Device,
    ) -> Self {
        let (gx, gy, guidance) = dims;
        let c = payload.channels();
        let shape = [num_views, c, guidance, gy, gx];
        Self {
            grids: Some(Tensor::zeros(shape, device)),
            m1: Some(Tensor::zeros(shape, device)),
            m2: Some(Tensor::zeros(shape, device)),
            last_step: vec![-1; num_views],
            betas,
            mean_ema: mean_reg.then(|| Tensor::zeros([c], device)),
        }
    }

    fn grids_ref(&self) -> &Tensor<5> {
        self.grids
            .as_ref()
            .expect("grids always present between steps")
    }

    fn view_grid(&self, view_idx: usize) -> Tensor<5> {
        self.grids_ref().clone().slice(s![view_idx..view_idx + 1])
    }

    /// One lazy, dense-Adam-equivalent update for `view_idx`'s slice.
    ///
    /// Reproduces exactly what a dense Adam over the full `[N, ...]`
    /// tensor — the reference implementations' optimizer — would have
    /// applied since this view's last visit (up to `eps` placement, and
    /// the LR schedule sampled at visits instead of every step): first
    /// the momentum tail of zero-gradient steps dense Adam kept taking
    /// over the gap while `m` decayed (each is bias-corrected
    /// `m_hat/sqrt(v_hat)`; they share the per-element direction
    /// `m/sqrt(v)`, so their coefficients fold into one host-side scalar
    /// sum), then the visit's own Adam step on the gap-decayed moments
    /// with *global*-timestep bias correction. For consecutive visits
    /// this reduces to standard Adam. `SparseAdam` semantics (per-view
    /// bias-correction timesteps) are deliberately NOT used: they turn
    /// every visit into a `~lr`-sized step, capping a view's total
    /// movement at `visits * lr` — orders of magnitude too slow on
    /// many-view datasets, where dense Adam's between-visit second-moment
    /// decay amplifies each visit's effect instead.
    fn step(&mut self, view_idx: usize, grad: Tensor<5>, lr: f64, global_step: u32) {
        let r = s![view_idx..view_idx + 1];
        let (b1, b2) = self.betas;
        // 1-based dense-Adam timesteps of the previous visit's update and
        // this one's (`t_old == 0`: never visited).
        let t_old = self.last_step[view_idx] + 1;
        let t_new = i64::from(global_step) + 1;
        self.last_step[view_idx] = i64::from(global_step);
        let dt = i32::try_from((t_new - t_old).max(1)).expect("step gap fits i32");

        let grids = self.grids.take().expect("grids present");
        let m1 = self.m1.take().expect("m1 present");
        let m2 = self.m2.take().expect("m2 present");

        let mut pv = grids.clone().slice(r);
        let m1v = m1.clone().slice(r);
        let m2v = m2.clone().slice(r);

        // Catch up the gap's zero-gradient steps.
        if t_old > 0 && dt > 1 {
            let t_old = i32::try_from(t_old).expect("step fits i32");
            let mut tail = 0.0f64;
            for k in 1..dt {
                let decay = b1.powi(k) / b2.powi(k).sqrt();
                if decay < 1e-14 {
                    break;
                }
                tail += decay * (1.0 - b2.powi(t_old + k)).sqrt() / (1.0 - b1.powi(t_old + k));
            }
            if tail > 0.0 {
                pv = pv - m1v.clone() / (m2v.clone().sqrt() + ADAM_EPS) * (lr * tail);
            }
        }

        // The visit's own step, on the gap-decayed moments.
        let t = i32::try_from(t_new).expect("step fits i32");
        let m1v = m1v * b1.powi(dt) + grad.clone() * (1.0 - b1);
        let m2v = m2v * b2.powi(dt) + grad.powi_scalar(2) * (1.0 - b2);
        let m_hat = m1v.clone() / (1.0 - b1.powi(t));
        let v_hat = m2v.clone() / (1.0 - b2.powi(t));
        let pv = pv - m_hat / (v_hat.sqrt() + ADAM_EPS) * lr;

        // Track the dataset-wide payload mean for the identity anchor.
        if let Some(ema) = self.mean_ema.take() {
            let c = pv.dims()[1];
            let cells = pv.dims()[2] * pv.dims()[3] * pv.dims()[4];
            let view_mean = pv
                .clone()
                .reshape([c as i32, cells as i32])
                .mean_dim(1)
                .reshape([c]);
            let beta = 1.0 - 1.0 / MEAN_EMA_PERIOD;
            self.mean_ema = Some(ema * beta + view_mean * (1.0 - beta));
        }

        self.grids = Some(grids.slice_assign(r, pv));
        self.m1 = Some(m1.slice_assign(r, m1v));
        self.m2 = Some(m2.slice_assign(r, m2v));
    }
}

/// Dense Adam state for the four PPISP parameter tensors. In hybrid mode
/// the per-frame tensors (exposure/color) receive exactly-zero gradients
/// (the kernel runs with `with_frame = false`) and stay at identity.
struct PpispTrainState {
    model: PpispModel,
    m_exposure: (Tensor<1>, Tensor<1>),
    m_vignetting: (Tensor<3>, Tensor<3>),
    m_color: (Tensor<2>, Tensor<2>),
    m_crf: (Tensor<3>, Tensor<3>),
    step: i32,
}

impl PpispTrainState {
    fn new(
        num_cameras: usize,
        num_views: usize,
        camera_indices: Vec<u32>,
        device: &Device,
    ) -> Self {
        let model = PpispModel::new(num_cameras, num_views, camera_indices, device);
        let zeros_like1 = |t: &Tensor<1>| {
            (
                Tensor::zeros(t.dims(), device),
                Tensor::zeros(t.dims(), device),
            )
        };
        let zeros_like2 = |t: &Tensor<2>| {
            (
                Tensor::zeros(t.dims(), device),
                Tensor::zeros(t.dims(), device),
            )
        };
        let zeros_like3 = |t: &Tensor<3>| {
            (
                Tensor::zeros(t.dims(), device),
                Tensor::zeros(t.dims(), device),
            )
        };
        Self {
            m_exposure: zeros_like1(&model.exposure.val()),
            m_vignetting: zeros_like3(&model.vignetting.val()),
            m_color: zeros_like2(&model.color.val()),
            m_crf: zeros_like3(&model.crf.val()),
            model,
            step: 0,
        }
    }

    /// Lift the stored (inner) params into a fresh tracked model for one
    /// training step.
    fn lifted(&self) -> PpispModel {
        use burn::module::Param;
        PpispModel {
            exposure: Param::from_tensor(
                lift_to_autodiff(self.model.exposure.val()).require_grad(),
            ),
            vignetting: Param::from_tensor(
                lift_to_autodiff(self.model.vignetting.val()).require_grad(),
            ),
            color: Param::from_tensor(lift_to_autodiff(self.model.color.val()).require_grad()),
            crf: Param::from_tensor(lift_to_autodiff(self.model.crf.val()).require_grad()),
            camera_indices: self.model.camera_indices.clone(),
        }
    }

    fn step(&mut self, lifted: &PpispModel, grads: &mut Gradients, lr: f64) {
        use burn::module::Param;
        self.step += 1;
        let t = self.step;

        fn update<const D: usize>(
            param: &mut Param<Tensor<D>>,
            moments: &mut (Tensor<D>, Tensor<D>),
            lifted: &Param<Tensor<D>>,
            grads: &mut Gradients,
            t: i32,
            lr: f64,
        ) {
            let Some(grad) = lifted.val().grad_remove(grads) else {
                return;
            };
            let grad = detach_autodiff(grad);
            let (p, m1, m2) = adam_update(
                param.val(),
                grad,
                moments.0.clone(),
                moments.1.clone(),
                t,
                lr,
                (0.9, 0.999),
            );
            *moments = (m1, m2);
            *param = Param::from_tensor(p);
        }

        update(
            &mut self.model.exposure,
            &mut self.m_exposure,
            &lifted.exposure,
            grads,
            t,
            lr,
        );
        update(
            &mut self.model.vignetting,
            &mut self.m_vignetting,
            &lifted.vignetting,
            grads,
            t,
            lr,
        );
        update(
            &mut self.model.color,
            &mut self.m_color,
            &lifted.color,
            grads,
            t,
            lr,
        );
        update(
            &mut self.model.crf,
            &mut self.m_crf,
            &lifted.crf,
            grads,
            t,
            lr,
        );
    }
}

/// Which correction pipeline is active.
#[derive(Debug, Clone, Copy)]
enum PipelineMode {
    /// Comparison modes: optional full per-frame PPISP, then optional
    /// affine bilateral grid.
    Legacy,
    /// Per-camera vignetting → PPISP grid → optional per-camera CRF.
    Hybrid {
        payload: GridPayload,
        crf_per_camera: bool,
    },
}

/// Trainer-facing appearance state: holds the params + optimizer state on
/// the inner backend, and hands out per-step [`ActiveAppearance`] handles
/// with the lifted (tracked) tensors.
pub struct AppearanceTrainState {
    config: AppearanceConfig,
    mode: PipelineMode,
    total_iters: u32,
    grid: Option<GridTrainState>,
    ppisp: Option<PpispTrainState>,
}

/// The lifted appearance tensors for one training step. Apply to the
/// rendered image, add [`Self::reg_loss`] to the training loss, then hand
/// back to [`AppearanceTrainState::end_step`] after `loss.backward()`.
pub struct ActiveAppearance {
    view_idx: usize,
    mode: PipelineMode,
    /// Tracked `[1, C, L, H, W]` leaf for the active view.
    view_grid: Option<Tensor<5>>,
    ppisp: Option<PpispModel>,
    /// Detached EMA anchor `[C]` for the mean regulariser.
    mean_ema: Option<Tensor<1>>,
    tv_weight: f32,
    mean_reg_weight: f32,
    reg_scale: f32,
    subsample: GradSubsample,
}

impl AppearanceTrainState {
    /// Returns `None` when no appearance model is enabled. `device` may be
    /// the autodiff device; state is kept on its inner counterpart.
    pub fn new(
        config: AppearanceConfig,
        camera_indices: Vec<u32>,
        total_iters: u32,
        device: &Device,
    ) -> Option<Self> {
        if !config.bilagrid && !config.ppisp && !config.ppisp_grid {
            return None;
        }
        assert!(
            !(config.ppisp_grid && (config.bilagrid || config.ppisp)),
            "--ppisp-grid replaces the separate --bilateral-grid / --ppisp modes"
        );
        let device = device.clone().inner();
        let num_views = camera_indices.len();
        let num_cameras = camera_indices.iter().copied().max().unwrap_or(0) as usize + 1;

        let (mode, grid, ppisp) = if config.ppisp_grid {
            let payload = GridPayload {
                color: config.grid_color,
                crf: config.grid_crf,
                vignetting: true,
            };
            let grid = GridTrainState::new_payload(
                num_views,
                config.bilagrid_dims,
                payload,
                config.bilagrid_betas,
                config.bilagrid_mean_reg > 0.0,
                &device,
            );
            // Per-camera vignetting (+ optional CRF); frame params stay at
            // identity (the grid owns exposure/color).
            let ppisp = PpispTrainState::new(num_cameras, num_views, camera_indices, &device);
            (
                PipelineMode::Hybrid {
                    payload,
                    crf_per_camera: config.crf_per_camera,
                },
                Some(grid),
                Some(ppisp),
            )
        } else {
            let grid = config.bilagrid.then(|| {
                GridTrainState::new_affine(
                    num_views,
                    config.bilagrid_dims,
                    config.bilagrid_betas,
                    &device,
                )
            });
            let ppisp = config
                .ppisp
                .then(|| PpispTrainState::new(num_cameras, num_views, camera_indices, &device));
            (PipelineMode::Legacy, grid, ppisp)
        };

        Some(Self {
            mode,
            grid,
            ppisp,
            config,
            total_iters,
        })
    }

    /// Lift the active view's appearance params onto the autodiff graph.
    /// `step` rotates the gradient-subsample offset.
    pub fn begin_step(&self, view_idx: usize, step: u32) -> ActiveAppearance {
        let every = self.config.grad_subsample.max(1);
        ActiveAppearance {
            view_idx,
            mode: self.mode,
            view_grid: self
                .grid
                .as_ref()
                .map(|b| lift_to_autodiff(b.view_grid(view_idx)).require_grad()),
            ppisp: self.ppisp.as_ref().map(PpispTrainState::lifted),
            mean_ema: self
                .grid
                .as_ref()
                .and_then(|g| g.mean_ema.as_ref())
                .map(|e| lift_to_autodiff(e.clone())),
            tv_weight: self.config.bilagrid_tv_weight,
            mean_reg_weight: self.config.bilagrid_mean_reg,
            reg_scale: self.config.ppisp_reg_scale,
            subsample: GradSubsample::for_step(every, step),
        }
    }

    /// Consume the step's gradients and run the Adam updates. `step` is the
    /// global training iteration (for the LR schedules). Takes `active` by
    /// value: the lifted tensors are stale once the update ran.
    #[allow(clippy::needless_pass_by_value)]
    pub fn end_step(&mut self, active: ActiveAppearance, grads: &mut Gradients, step: u32) {
        if let (Some(state), Some(view_grid)) = (self.grid.as_mut(), &active.view_grid)
            && let Some(grad) = view_grid.clone().grad_remove(grads)
        {
            let lr = scheduled_lr(
                step,
                self.config.bilagrid_lr,
                BILAGRID_LR_WARMUP,
                self.total_iters,
            );
            state.step(active.view_idx, detach_autodiff(grad), lr, step);
        }
        if let (Some(state), Some(lifted)) = (self.ppisp.as_mut(), &active.ppisp) {
            let lr = scheduled_lr(
                step,
                self.config.ppisp_lr,
                PPISP_LR_WARMUP,
                self.total_iters,
            );
            state.step(lifted, grads, lr);
        }
    }

    /// Magnitude summary of the learned appearance parameters, for
    /// periodic logging — the cheapest way to see whether the model is
    /// actually learning (all-zero payloads mean identity).
    pub async fn stats(&self) -> Option<String> {
        let read = |t: Tensor<1>| async move {
            t.into_scalar_async::<f32>()
                .await
                .expect("appearance stats readback")
        };
        let mut parts = Vec::new();
        if let Some(g) = &self.grid {
            let grids = g.grids_ref().clone();
            match self.mode {
                PipelineMode::Hybrid { .. } => {
                    // Channel 0 is log2 exposure; zero payload = identity.
                    let exposure = grids.clone().slice(s![.., 0..1]);
                    let lo = read(exposure.clone().min()).await;
                    let hi = read(exposure.max()).await;
                    let payload = read(grids.abs().max()).await;
                    parts.push(format!(
                        "grid exposure [{lo:+.3}, {hi:+.3}] stops, payload |max| {payload:.3}"
                    ));
                }
                PipelineMode::Legacy => {
                    let lo = read(grids.clone().min()).await;
                    let hi = read(grids.max()).await;
                    parts.push(format!("grid range [{lo:.3}, {hi:.3}]"));
                }
            }
        }
        if let Some(p) = &self.ppisp {
            let vig = read(p.model.vignetting.val().abs().max()).await;
            parts.push(format!("vignetting |max| {vig:.3}"));
            if matches!(self.mode, PipelineMode::Legacy) {
                let e = p.model.exposure.val();
                let lo = read(e.clone().min()).await;
                let hi = read(e.max()).await;
                parts.push(format!("exposure [{lo:+.3}, {hi:+.3}] stops"));
            }
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }

    /// Forward-only correction for evaluation renders of *training* views
    /// (used with `--train-on-eval`, where eval views keep their learned
    /// per-view parameters). `img` is `[H, W, 3|4]` on the inner backend.
    pub fn apply_eval(&self, img: Tensor<3>, view_idx: usize) -> Tensor<3> {
        let p = self.ppisp.as_ref().map(|s| &s.model);
        let cam = |p: &PpispModel| p.camera_indices[view_idx] as usize;
        match self.mode {
            PipelineMode::Legacy => {
                let img = match p {
                    Some(p) => ppisp_apply_inner(
                        p.exposure.val(),
                        p.vignetting.val(),
                        p.color.val(),
                        p.crf.val(),
                        img,
                        cam(p),
                        view_idx,
                        PpispStages::ALL,
                    ),
                    None => img,
                };
                match &self.grid {
                    Some(g) => bilagrid_apply_inner(g.grids_ref().clone(), img, view_idx),
                    None => img,
                }
            }
            PipelineMode::Hybrid {
                payload,
                crf_per_camera,
            } => {
                let p = p.expect("hybrid always has ppisp state");
                let g = self.grid.as_ref().expect("hybrid always has a grid");
                let img = ppisp_grid_apply_inner(
                    g.grids_ref().clone(),
                    p.vignetting.val(),
                    img,
                    view_idx,
                    cam(p),
                    payload,
                );
                if crf_per_camera {
                    ppisp_apply_inner(
                        p.exposure.val(),
                        p.vignetting.val(),
                        p.color.val(),
                        p.crf.val(),
                        img,
                        cam(p),
                        view_idx,
                        PpispStages::CRF_ONLY,
                    )
                } else {
                    img
                }
            }
        }
    }
}

impl ActiveAppearance {
    /// Apply the enabled corrections to a rendered `[H, W, 3|4]` image.
    /// Alpha passes through untouched.
    pub fn apply(&self, img: Tensor<3>) -> Tensor<3> {
        match self.mode {
            PipelineMode::Legacy => {
                let img = match &self.ppisp {
                    Some(p) => p.apply(img, self.view_idx),
                    None => img,
                };
                match &self.view_grid {
                    Some(g) => bilagrid_apply(g.clone(), img, 0, self.subsample),
                    None => img,
                }
            }
            PipelineMode::Hybrid {
                payload,
                crf_per_camera,
            } => {
                // Vignetting is fused into the grid pass; an optional
                // per-camera CRF runs as a separate pass after it.
                let p = self.ppisp.as_ref().expect("hybrid always has ppisp");
                let camera_idx = p.camera_indices[self.view_idx] as usize;
                let g = self.view_grid.as_ref().expect("hybrid always has a grid");
                let img = ppisp_grid_apply(
                    g.clone(),
                    p.vignetting.val(),
                    img,
                    0,
                    camera_idx,
                    payload,
                    self.subsample,
                );
                if crf_per_camera {
                    p.apply_stages(img, self.view_idx, PpispStages::CRF_ONLY)
                } else {
                    img
                }
            }
        }
    }

    /// Regularisation for this step: TV on the active view's grid, the
    /// EMA-anchored mean-to-identity term (hybrid payloads), plus the PPISP
    /// parameter priors.
    ///
    /// The TV/mean terms cover only the sampled view (the references
    /// regularise all `N` grids every step with an extra `1/N` factor —
    /// per-epoch that applies the same total pressure per view as these
    /// per-step single-view terms, while keeping step cost independent of
    /// the dataset size).
    pub fn reg_loss(&self) -> Option<Tensor<1>> {
        let mut loss: Option<Tensor<1>> = None;
        let add = |loss: &mut Option<Tensor<1>>, term: Tensor<1>| {
            *loss = Some(match loss.take() {
                Some(l) => l + term,
                None => term,
            });
        };
        if let Some(g) = &self.view_grid {
            if self.tv_weight > 0.0 {
                add(&mut loss, bilagrid_tv_loss(g.clone()) * self.tv_weight);
            }
            // Mean anchor: gradient of `w * mse(EMA, 0)` w.r.t. this view's
            // contribution, with the EMA itself held constant — drives the
            // dataset-mean transform toward identity without touching the
            // other views' grids. Channel-mean (not sum) normalization so
            // the weight is directly comparable to spirulae's
            // `bilagrid_mean_reg_weight` (both default 10).
            if let Some(ema) = &self.mean_ema
                && self.mean_reg_weight > 0.0
            {
                let dims = g.dims();
                let c = dims[1];
                let cells = dims[2] * dims[3] * dims[4];
                let view_mean = g
                    .clone()
                    .reshape([c as i32, cells as i32])
                    .mean_dim(1)
                    .reshape([c]);
                let term =
                    (ema.clone() * view_mean).sum() * (2.0 * self.mean_reg_weight / c as f32);
                add(&mut loss, term);
            }
        }
        if let Some(p) = &self.ppisp
            && self.reg_scale > 0.0
        {
            add(&mut loss, p.reg_loss() * self.reg_scale);
        }
        loss
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use crate::ppisp_grid::GridPayload;

    /// The consolidated per-visit update must reproduce what a dense Adam
    /// over the full `[N, ...]` grid tensor (the reference implementations'
    /// optimizer) applies in total. `SparseAdam` semantics would end this
    /// schedule at `visits * lr ~ 0.02` — dense Adam's between-visit
    /// second-moment decay reaches several times that.
    #[tokio::test]
    async fn sparse_step_matches_dense_adam() {
        let device = Device::from(brush_cube::test_helpers::test_device().await);
        let payload = GridPayload {
            color: false,
            crf: false,
            vignetting: true,
        };
        let betas = (0.9f64, 0.999f64);
        let mut state = GridTrainState::new_payload(3, (1, 1, 1), payload, betas, false, &device);

        let lr = 2e-3;
        let gap = 40u32;
        let visits = 10;
        // Dense Adam reference for view 0's single cell, in host f64. The
        // view sees a gradient every `gap` steps and zeros in between, but
        // dense Adam still applies an update every step. The lazy update
        // defers each gap's zero-gradient tail to the next visit, so
        // parity holds right after a visit — stop both at the last one.
        let (mut p, mut m, mut v) = (0.0f64, 0.0f64, 0.0f64);
        for step in 0..=gap * (visits - 1) {
            let g = if step % gap == 0 {
                let g = 1.0 + 0.5 * f64::sin(f64::from(step));
                let grad = Tensor::<5>::full([1, 1, 1, 1, 1], g as f32, &device);
                state.step(0, grad, lr, step);
                g
            } else {
                0.0
            };
            m = betas.0 * m + (1.0 - betas.0) * g;
            v = betas.1 * v + (1.0 - betas.1) * g * g;
            let t = f64::from(step + 1);
            let m_hat = m / (1.0 - betas.0.powf(t));
            let v_hat = v / (1.0 - betas.1.powf(t));
            p -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
        }

        let got = f64::from(
            state
                .view_grid(0)
                .into_data_async()
                .await
                .expect("readback")
                .to_vec::<f32>()
                .expect("vec")[0],
        );
        // Prove we're in the amplification regime SparseAdam can't reach...
        assert!(
            p.abs() > 3.0 * f64::from(visits) * lr,
            "dense reference moved only {p:.4}; test schedule too easy"
        );
        // ...and that the lazy update tracks it.
        let rel = (got - p).abs() / p.abs();
        assert!(
            rel < 0.02,
            "lazy update {got:.5} vs dense Adam {p:.5} (rel err {rel:.3})"
        );
        // Untouched views must stay at identity.
        let other = state
            .view_grid(1)
            .abs()
            .max()
            .into_scalar_async::<f32>()
            .await
            .expect("readback");
        assert_eq!(other, 0.0);
    }
}
