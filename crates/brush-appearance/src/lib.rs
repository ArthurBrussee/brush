//! Per-view appearance compensation for Brush training.
//!
//! Two independently-toggleable models, both applied to the *rendered* image
//! before the photometric loss so the splats themselves learn canonical
//! (appearance-free) colors:
//!
//! - [`bilagrid`]: per-view 3D bilateral grids ("Bilateral Guided Radiance
//!   Field Processing", `BilaRF`). Each training view owns a `[12, L, H, W]`
//!   grid of 3x4 affine color transforms, sliced per pixel by screen position
//!   and grayscale guidance. Fused forward/backward `CubeCL` kernels (ported
//!   from `LichtFeld` Studio / gsplat's fused implementation) plus a fused
//!   total-variation regulariser.
//!
//! - [`ppisp`]: physically-plausible ISP compensation (NVIDIA PPISP). Models
//!   per-frame exposure + color homography and per-camera vignetting + CRF
//!   with a handful of parameters each, applied per pixel by a fused kernel.
//!   The parameter regularisation is plain tensor ops (the params are tiny).
//!
//! Both follow the `brush-loss` pattern: a backend trait implemented for the
//! raw `CubeCL` backend and the Fusion backend, plus custom Burn autodiff ops
//! so gradients flow to both the appearance params and the rendered image.

pub mod bilagrid;
mod bilagrid_kernels;
pub mod ppisp;
pub mod ppisp_grid;
mod ppisp_grid_kernels;
mod ppisp_kernels;
mod ppisp_math;
pub mod train_state;

use burn::backend::wgpu::WgpuRuntime;
use burn::tensor::{DType, Shape};
use burn_cubecl::{CubeRuntime, fusion::FusionCubeRuntime, tensor::CubeTensor};
use burn_fusion::{
    FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

pub(crate) use brush_cube::{AtomicAddF32, CasAtomicAdd, HfAtomicAdd};

pub(crate) fn alloc_zeros<R: CubeRuntime>(
    template: &CubeTensor<R>,
    shape: Shape,
    dtype: DType,
) -> CubeTensor<R> {
    burn_cubecl::ops::numeric::zeros_client::<R>(
        template.client.clone(),
        template.device.clone(),
        shape,
        dtype,
    )
}

/// Wraps a closure as a fusion `Operation` (same pattern as `brush-loss`).
struct ClosureOp<F> {
    desc: CustomOpIr,
    op: F,
}

impl<F> std::fmt::Debug for ClosureOp<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClosureOp({:?})", self.desc)
    }
}

impl<F> Operation<FusionCubeRuntime<WgpuRuntime>> for ClosureOp<F>
where
    F: Fn(&CustomOpIr, &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>)
        + Send
        + Sync
        + 'static,
{
    fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        (self.op)(&self.desc, h);
    }
}

pub(crate) type FusionTensor = burn_fusion::FusionTensor<FusionCubeRuntime<WgpuRuntime>>;

/// Register a custom op with `N` inputs and `M` outputs on the Fusion
/// stream. Generalises `brush-loss`'s single-output helper: each output is
/// described by `(shape, dtype)`; `op` runs against the inner backend when
/// fusion executes the queued op.
pub(crate) fn dispatch_custom<const N: usize, const M: usize, F>(
    name: &'static str,
    inputs: [FusionTensor; N],
    outputs: [(Shape, DType); M],
    op: F,
) -> [FusionTensor; M]
where
    F: Fn(&CustomOpIr, &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>)
        + Send
        + Sync
        + 'static,
{
    let client = inputs[0].client.clone();
    let outs =
        outputs.map(|(shape, dtype)| TensorIr::uninit(client.create_empty_handle(), shape, dtype));
    let stream = StreamId::current();
    let desc = CustomOpIr::new(name, &inputs.map(|t| t.into_ir()), &outs);
    let wrapped = ClosureOp {
        desc: desc.clone(),
        op,
    };
    client
        .register(stream, OperationIr::Custom(desc), wrapped)
        .outputs()
}

/// Resolve a possibly-fused float tensor into a contiguous `CubeTensor`.
pub(crate) fn contiguous<R: CubeRuntime>(t: CubeTensor<R>) -> CubeTensor<R> {
    burn_cubecl::kernel::into_contiguous(t)
}

// Re-export the user-facing API at the crate root.
/// Grid-gradient subsampling: each plane (subgroup) independently elects
/// itself with probability `1/every` per step — a Bernoulli draw from a
/// PCG hash of the plane index and the per-step `seed` — and elected
/// planes scatter grid gradients scaled by `every` to stay unbiased. The
/// image gradient is always exact. `every = 1` disables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradSubsample {
    pub every: u32,
    pub seed: u32,
}

impl GradSubsample {
    /// The subsample schedule for a global training step.
    pub fn for_step(every: u32, step: u32) -> Self {
        // PCG-style mix of the step index; any fixed integer hash works.
        let mut s = step.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        s = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
        Self {
            every: every.max(1),
            seed: (s >> 22) ^ s,
        }
    }
}

impl Default for GradSubsample {
    fn default() -> Self {
        Self { every: 1, seed: 0 }
    }
}

pub use bilagrid::{BilagridModel, bilagrid_apply, bilagrid_tv_loss};
pub use ppisp::{PpispModel, PpispStages, ppisp_apply};
pub use ppisp_grid::{GridPayload, ppisp_grid_apply};
pub use train_state::{ActiveAppearance, AppearanceTrainState};

/// Static configuration for the appearance models.
#[derive(Debug, Clone)]
pub struct AppearanceConfig {
    /// Hybrid mode: per-camera vignetting + per-view PPISP grid
    /// (+ optional per-camera CRF). Replaces the two legacy modes.
    pub ppisp_grid: bool,
    /// Hybrid grid payload includes the 8 color-homography latents
    /// (`false` = exposure-only grid, one log2 scalar per cell).
    pub grid_color: bool,
    /// Hybrid grid payload includes per-cell CRF offsets.
    pub grid_crf: bool,
    /// Learn a per-camera CRF applied after the hybrid grid.
    pub crf_per_camera: bool,
    /// Legacy comparison mode: per-view affine bilateral grids.
    pub bilagrid: bool,
    /// Legacy comparison mode: full per-frame PPISP.
    pub ppisp: bool,
    /// Grid dims `(x, y, guidance)` — spatial width/height and the
    /// grayscale guidance dimension (both grid kinds).
    pub bilagrid_dims: (usize, usize, usize),
    /// Weight of the grid total-variation regulariser.
    pub bilagrid_tv_weight: f32,
    /// Weight of the EMA-anchored mean-to-identity regulariser (hybrid
    /// payload grids; 0 disables).
    pub bilagrid_mean_reg: f32,
    /// Grid learning rate (warmup + exponential decay applied).
    pub bilagrid_lr: f64,
    /// Adam betas for the sparse per-view grid updates.
    pub bilagrid_betas: (f64, f64),
    /// Grid-gradient subsampling: only every Nth subgroup scatters grid
    /// gradients (1 disables). See [`GradSubsample`]. Defaults to 4 —
    /// quality-neutral for the heavily regularised grids, and removes most
    /// of the grid backward's atomic cost.
    pub grad_subsample: u32,
    /// PPISP learning rate (warmup + exponential decay applied).
    pub ppisp_lr: f64,
    /// Scale on all PPISP regularisation terms.
    pub ppisp_reg_scale: f32,
}

impl Default for AppearanceConfig {
    fn default() -> Self {
        Self {
            ppisp_grid: false,
            grid_color: true,
            grid_crf: false,
            crf_per_camera: false,
            bilagrid: false,
            ppisp: false,
            bilagrid_dims: (16, 16, 8),
            bilagrid_tv_weight: 10.0,
            bilagrid_mean_reg: 10.0,
            bilagrid_lr: 2e-3,
            bilagrid_betas: (0.9, 0.999),
            grad_subsample: 4,
            ppisp_lr: 2e-3,
            ppisp_reg_scale: 1.0,
        }
    }
}

/// Warmup + exponential-decay LR schedule used for both appearance models
/// (matches the PPISP / `LichtFeld` bilateral-grid schedules): linear warmup
/// from `start_factor * base` over `warmup_steps`, then exponential decay
/// toward `final_factor * base` at `decay_steps`.
pub fn warmup_exp_lr(
    step: u32,
    base: f64,
    warmup_steps: u32,
    start_factor: f64,
    final_factor: f64,
    decay_steps: u32,
) -> f64 {
    if step < warmup_steps {
        let t = (step as f64 + 1.0) / warmup_steps as f64;
        base * (start_factor + (1.0 - start_factor) * t)
    } else {
        let decay_step = (step - warmup_steps) as f64;
        base * final_factor.powf(decay_step / decay_steps.max(1) as f64)
    }
}
