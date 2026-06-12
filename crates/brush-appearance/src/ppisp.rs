//! Host side of PPISP: backend trait + kernel launches, Fusion dispatch,
//! Burn autodiff op, the `PpispModel` module and its (tensor-op based)
//! parameter regularisation.

use brush_cube::{MainBackend, MainBackendBase};
use brush_render::burn_glue::{
    AutodiffMain, unwrap_ad_wgpu_float, wrap_ad_wgpu_float, wrap_wgpu_float,
};
use burn::{
    backend::{
        Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::FloatTensor,
    },
    module::{Module, Param},
    tensor::{DType, Device, Shape, Tensor, TensorData},
};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use burn_fusion::Fusion;

use crate::ppisp_kernels as kernels;
use crate::{alloc_zeros, contiguous, dispatch_custom};

/// Number of scalar parameter gradients produced per pixel
/// (see `ppisp_kernels` for the slot layout).
const NUM_PARAM_GRADS: usize = kernels::NUM_PARAM_GRADS as usize;

/// Which PPISP stages a pass applies. The full model is all three; the
/// hybrid grid mode runs camera-only passes (vignetting before the grid,
/// optionally CRF after it) with the per-frame stages handled by the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PpispStages {
    /// Per-frame exposure + color homography.
    pub frame: bool,
    /// Per-camera vignetting.
    pub vignetting: bool,
    /// Per-camera CRF tone curve.
    pub crf: bool,
}

impl PpispStages {
    pub const ALL: Self = Self {
        frame: true,
        vignetting: true,
        crf: true,
    };
    pub const VIGNETTING_ONLY: Self = Self {
        frame: false,
        vignetting: true,
        crf: false,
    };
    pub const CRF_ONLY: Self = Self {
        frame: false,
        vignetting: false,
        crf: true,
    };
}

/// Backend hooks for the PPISP kernels.
pub trait PpispOps<B: Backend> {
    /// Apply the full pipeline to `rgb` `[h, w, 3|4]` (alpha untouched).
    #[allow(clippy::too_many_arguments)]
    fn ppisp_fwd(
        exposure: FloatTensor<B>,
        vignetting: FloatTensor<B>,
        color: FloatTensor<B>,
        crf: FloatTensor<B>,
        rgb: FloatTensor<B>,
        camera_idx: usize,
        frame_idx: usize,
        stages: PpispStages,
    ) -> FloatTensor<B>;

    /// Raw backward: returns `(partials [num_cubes, 36], dL/drgb)`. The
    /// caller reduces `partials` and scatters into full-shape param grads.
    #[allow(clippy::too_many_arguments)]
    fn ppisp_bwd_raw(
        exposure: FloatTensor<B>,
        vignetting: FloatTensor<B>,
        color: FloatTensor<B>,
        crf: FloatTensor<B>,
        rgb: FloatTensor<B>,
        v_out: FloatTensor<B>,
        camera_idx: usize,
        frame_idx: usize,
        stages: PpispStages,
    ) -> (FloatTensor<B>, FloatTensor<B>);
}

fn ppisp_dims<R: CubeRuntime>(rgb: &CubeTensor<R>) -> (u32, u32, u32) {
    let dims = rgb.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "rgb must be [h, w, c]");
    let ch = dims[2] as u32;
    assert!(ch == 3 || ch == 4, "rgb must have 3 or 4 channels");
    (dims[0] as u32, dims[1] as u32, ch)
}

#[allow(clippy::too_many_arguments)]
fn launch_fwd<R: CubeRuntime>(
    exposure: CubeTensor<R>,
    vignetting: CubeTensor<R>,
    color: CubeTensor<R>,
    crf: CubeTensor<R>,
    rgb: CubeTensor<R>,
    camera_idx: usize,
    frame_idx: usize,
    stages: PpispStages,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};

    let exposure = contiguous(exposure);
    let vignetting = contiguous(vignetting);
    let color = contiguous(color);
    let crf = contiguous(crf);
    let rgb = contiguous(rgb);
    let (h, w, ch) = ppisp_dims(&rgb);

    let out = alloc_zeros(&rgb, rgb.shape(), DType::F32);
    let client = rgb.client.clone();
    kernels::ppisp_fwd_kernel::launch::<R>(
        &client,
        CubeCount::Static((h * w).div_ceil(kernels::BLOCK_SIZE), 1, 1),
        CubeDim::new_1d(kernels::BLOCK_SIZE),
        exposure.into_tensor_arg(),
        vignetting.into_tensor_arg(),
        color.into_tensor_arg(),
        crf.into_tensor_arg(),
        rgb.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        h,
        w,
        camera_idx as u32,
        frame_idx as u32,
        ch,
        ch == 4,
        stages.frame,
        stages.vignetting,
        stages.crf,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn launch_bwd<R: CubeRuntime>(
    exposure: CubeTensor<R>,
    vignetting: CubeTensor<R>,
    color: CubeTensor<R>,
    crf: CubeTensor<R>,
    rgb: CubeTensor<R>,
    v_out: CubeTensor<R>,
    camera_idx: usize,
    frame_idx: usize,
    stages: PpispStages,
) -> (CubeTensor<R>, CubeTensor<R>) {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};

    let exposure = contiguous(exposure);
    let vignetting = contiguous(vignetting);
    let color = contiguous(color);
    let crf = contiguous(crf);
    let rgb = contiguous(rgb);
    let v_out = contiguous(v_out);
    let (h, w, ch) = ppisp_dims(&rgb);

    let num_cubes = (h * w).div_ceil(kernels::BLOCK_SIZE);
    let grad_rgb = alloc_zeros(&rgb, rgb.shape(), DType::F32);
    let partials = alloc_zeros(
        &rgb,
        Shape::new([num_cubes as usize, NUM_PARAM_GRADS]),
        DType::F32,
    );
    let client = rgb.client.clone();
    kernels::ppisp_bwd_kernel::launch::<R>(
        &client,
        CubeCount::Static(num_cubes, 1, 1),
        CubeDim::new_1d(kernels::BLOCK_SIZE),
        exposure.into_tensor_arg(),
        vignetting.into_tensor_arg(),
        color.into_tensor_arg(),
        crf.into_tensor_arg(),
        rgb.into_tensor_arg(),
        v_out.into_tensor_arg(),
        grad_rgb.clone().into_tensor_arg(),
        partials.clone().into_tensor_arg(),
        h,
        w,
        camera_idx as u32,
        frame_idx as u32,
        ch,
        ch == 4,
        stages.frame,
        stages.vignetting,
        stages.crf,
    );
    #[allow(clippy::tuple_array_conversions)]
    (partials, grad_rgb)
}

impl PpispOps<Self> for MainBackendBase {
    fn ppisp_fwd(
        exposure: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        color: FloatTensor<Self>,
        crf: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        camera_idx: usize,
        frame_idx: usize,
        stages: PpispStages,
    ) -> FloatTensor<Self> {
        launch_fwd(
            exposure, vignetting, color, crf, rgb, camera_idx, frame_idx, stages,
        )
    }

    fn ppisp_bwd_raw(
        exposure: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        color: FloatTensor<Self>,
        crf: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        v_out: FloatTensor<Self>,
        camera_idx: usize,
        frame_idx: usize,
        stages: PpispStages,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        launch_bwd(
            exposure, vignetting, color, crf, rgb, v_out, camera_idx, frame_idx, stages,
        )
    }
}

impl PpispOps<Self> for Fusion<MainBackendBase> {
    fn ppisp_fwd(
        exposure: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        color: FloatTensor<Self>,
        crf: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        camera_idx: usize,
        frame_idx: usize,
        stages: PpispStages,
    ) -> FloatTensor<Self> {
        let shape = rgb.shape();
        let [out] = dispatch_custom(
            "ppisp_fwd",
            [exposure, vignetting, color, crf, rgb],
            [(shape, DType::F32)],
            move |desc, h| {
                let ([exposure, vignetting, color, crf, rgb], [out]) = desc.as_fixed();
                let res = MainBackendBase::ppisp_fwd(
                    h.get_float_tensor::<MainBackendBase>(exposure),
                    h.get_float_tensor::<MainBackendBase>(vignetting),
                    h.get_float_tensor::<MainBackendBase>(color),
                    h.get_float_tensor::<MainBackendBase>(crf),
                    h.get_float_tensor::<MainBackendBase>(rgb),
                    camera_idx,
                    frame_idx,
                    stages,
                );
                h.register_float_tensor::<MainBackendBase>(&out.id, res);
            },
        );
        out
    }

    fn ppisp_bwd_raw(
        exposure: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        color: FloatTensor<Self>,
        crf: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        v_out: FloatTensor<Self>,
        camera_idx: usize,
        frame_idx: usize,
        stages: PpispStages,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let rgb_shape = rgb.shape();
        let [h_dim, w_dim, _] = rgb_shape.dims();
        let num_cubes = ((h_dim * w_dim) as u32).div_ceil(kernels::BLOCK_SIZE) as usize;
        let [partials, grad_rgb] = dispatch_custom(
            "ppisp_bwd_raw",
            [exposure, vignetting, color, crf, rgb, v_out],
            [
                (Shape::new([num_cubes, NUM_PARAM_GRADS]), DType::F32),
                (rgb_shape, DType::F32),
            ],
            move |desc, h| {
                let ([exposure, vignetting, color, crf, rgb, v_out], [partials, grad_rgb]) =
                    desc.as_fixed();
                let (p, g) = MainBackendBase::ppisp_bwd_raw(
                    h.get_float_tensor::<MainBackendBase>(exposure),
                    h.get_float_tensor::<MainBackendBase>(vignetting),
                    h.get_float_tensor::<MainBackendBase>(color),
                    h.get_float_tensor::<MainBackendBase>(crf),
                    h.get_float_tensor::<MainBackendBase>(rgb),
                    h.get_float_tensor::<MainBackendBase>(v_out),
                    camera_idx,
                    frame_idx,
                    stages,
                );
                h.register_float_tensor::<MainBackendBase>(&partials.id, p);
                h.register_float_tensor::<MainBackendBase>(&grad_rgb.id, g);
            },
        );
        #[allow(clippy::tuple_array_conversions)]
        (partials, grad_rgb)
    }
}

/// Full backward: reduce the per-cube partials and scatter the 36 summed
/// parameter gradients into full-shape gradient tensors at the active
/// frame/camera rows. Returns `(d_exposure, d_vignetting, d_color, d_crf,
/// d_rgb)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ppisp_backward<B: Backend + PpispOps<B>>(
    exposure: FloatTensor<B>,
    vignetting: FloatTensor<B>,
    color: FloatTensor<B>,
    crf: FloatTensor<B>,
    rgb: FloatTensor<B>,
    v_out: FloatTensor<B>,
    camera_idx: usize,
    frame_idx: usize,
    stages: PpispStages,
) -> (
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
) {
    use burn::tensor::Slice;
    let sl = |r: std::ops::Range<usize>| -> Slice { r.into() };

    let device = B::float_device(&rgb);
    let num_frames = exposure.shape().dims::<1>()[0];
    let num_cameras = vignetting.shape().dims::<3>()[0];

    let (partials, grad_rgb) = B::ppisp_bwd_raw(
        exposure, vignetting, color, crf, rgb, v_out, camera_idx, frame_idx, stages,
    );

    // [num_cubes, 36] → [1, 36] (deterministic on-GPU sum).
    let summed = B::float_sum_dim(partials, 0);

    let slice_1d =
        |t: FloatTensor<B>, lo: usize, hi: usize| B::float_slice(t, &[sl(0..1), sl(lo..hi)]);

    // Exposure: [F] with slot 0 at `frame_idx`.
    let g_exp = B::float_zeros(
        Shape::new([num_frames]),
        &device,
        burn::tensor::FloatDType::F32,
    );
    let g_exp = B::float_slice_assign(
        g_exp,
        &[sl(frame_idx..frame_idx + 1)],
        B::float_reshape(slice_1d(summed.clone(), 0, 1), Shape::new([1])),
    );

    // Vignetting: [C, 3, 5] with slots 1..16 at `camera_idx`.
    let g_vig = B::float_zeros(
        Shape::new([num_cameras, 3, 5]),
        &device,
        burn::tensor::FloatDType::F32,
    );
    let g_vig = B::float_slice_assign(
        g_vig,
        &[sl(camera_idx..camera_idx + 1), sl(0..3), sl(0..5)],
        B::float_reshape(slice_1d(summed.clone(), 1, 16), Shape::new([1, 3, 5])),
    );

    // Color: [F, 8] with slots 16..24 at `frame_idx`.
    let g_color = B::float_zeros(
        Shape::new([num_frames, 8]),
        &device,
        burn::tensor::FloatDType::F32,
    );
    let g_color = B::float_slice_assign(
        g_color,
        &[sl(frame_idx..frame_idx + 1), sl(0..8)],
        B::float_reshape(slice_1d(summed.clone(), 16, 24), Shape::new([1, 8])),
    );

    // CRF: [C, 3, 4] with slots 24..36 at `camera_idx`.
    let g_crf = B::float_zeros(
        Shape::new([num_cameras, 3, 4]),
        &device,
        burn::tensor::FloatDType::F32,
    );
    let g_crf = B::float_slice_assign(
        g_crf,
        &[sl(camera_idx..camera_idx + 1), sl(0..3), sl(0..4)],
        B::float_reshape(slice_1d(summed, 24, 36), Shape::new([1, 3, 4])),
    );

    (g_exp, g_vig, g_color, g_crf, grad_rgb)
}

// ---------------------------------------------------------------------------
// Autodiff op
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct PpispBackward;

#[derive(Debug, Clone)]
struct PpispState<B: Backend> {
    exposure: FloatTensor<B>,
    vignetting: FloatTensor<B>,
    color: FloatTensor<B>,
    crf: FloatTensor<B>,
    rgb: FloatTensor<B>,
    camera_idx: usize,
    frame_idx: usize,
    stages: PpispStages,
}

impl<B: Backend + PpispOps<B>> Backward<B, 5> for PpispBackward {
    type State = PpispState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let v_out = grads.consume::<B>(&ops.node);
        let [exp_parent, vig_parent, color_parent, crf_parent, rgb_parent] = ops.parents;
        let (g_exp, g_vig, g_color, g_crf, g_rgb) = ppisp_backward::<B>(
            state.exposure,
            state.vignetting,
            state.color,
            state.crf,
            state.rgb,
            v_out,
            state.camera_idx,
            state.frame_idx,
            state.stages,
        );
        if let Some(node) = exp_parent {
            grads.register::<B>(node.id, g_exp);
        }
        if let Some(node) = vig_parent {
            grads.register::<B>(node.id, g_vig);
        }
        if let Some(node) = color_parent {
            grads.register::<B>(node.id, g_color);
        }
        if let Some(node) = crf_parent {
            grads.register::<B>(node.id, g_crf);
        }
        if let Some(node) = rgb_parent {
            grads.register::<B>(node.id, g_rgb);
        }
    }
}

/// Apply PPISP to a rendered `[H, W, 3|4]` image (alpha untouched).
/// Differentiable w.r.t. all four parameter tensors and `rgb`. Inputs must
/// be on an autodiff-enabled Wgpu device.
///
/// - `exposure`: `[num_frames]` log2-exposure offsets.
/// - `vignetting`: `[num_cameras, 3, 5]` per-channel `cx, cy, a0, a1, a2`.
/// - `color`: `[num_frames, 8]` latent homography offsets.
/// - `crf`: `[num_cameras, 3, 4]` raw `toe, shoulder, gamma, center`.
pub fn ppisp_apply(
    exposure: Tensor<1>,
    vignetting: Tensor<3>,
    color: Tensor<2>,
    crf: Tensor<3>,
    rgb: Tensor<3>,
    camera_idx: usize,
    frame_idx: usize,
    stages: PpispStages,
) -> Tensor<3> {
    let exp_ad = unwrap_ad_wgpu_float(exposure);
    let vig_ad = unwrap_ad_wgpu_float(vignetting);
    let color_ad = unwrap_ad_wgpu_float(color);
    let crf_ad = unwrap_ad_wgpu_float(crf);
    let rgb_ad = unwrap_ad_wgpu_float(rgb);

    let prep = PpispBackward
        .prepare::<NoCheckpointing>([
            exp_ad.node.clone(),
            vig_ad.node.clone(),
            color_ad.node.clone(),
            crf_ad.node.clone(),
            rgb_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();

    let exp_p = exp_ad.primitive;
    let vig_p = vig_ad.primitive;
    let color_p = color_ad.primitive;
    let crf_p = crf_ad.primitive;
    let rgb_p = rgb_ad.primitive;

    let out = <MainBackend as PpispOps<MainBackend>>::ppisp_fwd(
        exp_p.clone(),
        vig_p.clone(),
        color_p.clone(),
        crf_p.clone(),
        rgb_p.clone(),
        camera_idx,
        frame_idx,
        stages,
    );

    let out_ad: FloatTensor<AutodiffMain> = match prep {
        OpsKind::Tracked(prep) => prep.finish(
            PpispState {
                exposure: exp_p,
                vignetting: vig_p,
                color: color_p,
                crf: crf_p,
                rgb: rgb_p,
                camera_idx,
                frame_idx,
                stages,
            },
            out,
        ),
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    wrap_ad_wgpu_float::<3>(out_ad)
}

/// Forward-only version of [`ppisp_apply`] for non-autodiff tensors.
pub fn ppisp_apply_inner(
    exposure: Tensor<1>,
    vignetting: Tensor<3>,
    color: Tensor<2>,
    crf: Tensor<3>,
    rgb: Tensor<3>,
    camera_idx: usize,
    frame_idx: usize,
    stages: PpispStages,
) -> Tensor<3> {
    use brush_render::burn_glue::unwrap_wgpu_float;
    let out = <MainBackend as PpispOps<MainBackend>>::ppisp_fwd(
        unwrap_wgpu_float(exposure),
        unwrap_wgpu_float(vignetting),
        unwrap_wgpu_float(color),
        unwrap_wgpu_float(crf),
        unwrap_wgpu_float(rgb),
        camera_idx,
        frame_idx,
        stages,
    );
    wrap_wgpu_float::<3>(out)
}

// ---------------------------------------------------------------------------
// Module + regularisation
// ---------------------------------------------------------------------------

/// Inverse of `min_value + softplus(x)` at `y`, for identity-CRF init.
fn softplus_inverse(y: f32, min_value: f32) -> f32 {
    let x = (y - min_value).max(1e-5);
    (x.exp_m1()).ln()
}

/// PPISP parameters: per-frame exposure + color, per-camera vignetting +
/// CRF, plus the (frozen) view → camera-group mapping.
#[derive(Module, Debug)]
pub struct PpispModel {
    /// `[num_frames]` log2-exposure offsets.
    pub exposure: Param<Tensor<1>>,
    /// `[num_cameras, 3, 5]` per-channel vignetting.
    pub vignetting: Param<Tensor<3>>,
    /// `[num_frames, 8]` latent color-homography offsets.
    pub color: Param<Tensor<2>>,
    /// `[num_cameras, 3, 4]` raw CRF params.
    pub crf: Param<Tensor<3>>,
    /// Per-view camera-group index (frozen metadata, not optimized).
    #[module(skip)]
    pub camera_indices: Vec<u32>,
}

/// Fixed regularisation weights from the PPISP reference implementation.
/// Scaled as a whole by the trainer's `ppisp_reg_scale`.
const W_EXPOSURE_MEAN: f32 = 1.0;
const W_VIG_CENTER: f32 = 0.02;
const W_VIG_CHANNEL: f32 = 0.1;
const W_VIG_NON_POS: f32 = 0.01;
const W_COLOR_MEAN: f32 = 1.0;
const W_CRF_CHANNEL: f32 = 0.1;

/// ZCA pinv blocks as a `[8, 8]` block-diagonal matrix (latent → real
/// chromaticity offsets) for the color-mean regulariser.
#[rustfmt::skip]
const COLOR_PINV_BLOCK_DIAG: [f32; 64] = [
    0.0480542, -0.0043631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.0043631, 0.0481283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0580570, -0.0179872, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.0179872, 0.0431061, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0433336, -0.0180537, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, -0.0180537, 0.0580500, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0128369, -0.0034654,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0034654, 0.0128158,
];

/// `smooth_l1(x, beta)` elementwise.
fn smooth_l1<const D: usize>(x: Tensor<D>, beta: f32) -> Tensor<D> {
    let ax = x.clone().abs();
    let quad = x.powi_scalar(2) * (0.5 / beta);
    let lin = ax.clone() - 0.5 * beta;
    lin.mask_where(ax.lower_elem(beta), quad)
}

impl PpispModel {
    pub fn new(
        num_cameras: usize,
        num_frames: usize,
        camera_indices: Vec<u32>,
        device: &Device,
    ) -> Self {
        assert_eq!(
            camera_indices.len(),
            num_frames,
            "need one camera-group index per training view"
        );
        // Identity-like CRF init (toe = shoulder = 1, gamma = 1, center = 0.5).
        let crf_row = [
            softplus_inverse(1.0, 0.3),
            softplus_inverse(1.0, 0.3),
            softplus_inverse(1.0, 0.1),
            0.0,
        ];
        let mut crf_data = Vec::with_capacity(num_cameras * 3 * 4);
        for _ in 0..num_cameras * 3 {
            crf_data.extend_from_slice(&crf_row);
        }
        let crf = Tensor::from_data(TensorData::new(crf_data, [num_cameras, 3, 4]), device);

        Self {
            exposure: Param::from_tensor(Tensor::zeros([num_frames], device)),
            vignetting: Param::from_tensor(Tensor::zeros([num_cameras, 3, 5], device)),
            color: Param::from_tensor(Tensor::zeros([num_frames, 8], device)),
            crf: Param::from_tensor(crf),
            camera_indices,
        }
    }

    /// Apply the full pipeline to a rendered `[H, W, 3|4]` image for
    /// training view `view_idx`.
    pub fn apply(&self, img: Tensor<3>, view_idx: usize) -> Tensor<3> {
        self.apply_stages(img, view_idx, PpispStages::ALL)
    }

    /// Apply a subset of the pipeline (e.g. the per-camera-only passes of
    /// the hybrid grid mode).
    pub fn apply_stages(&self, img: Tensor<3>, view_idx: usize, stages: PpispStages) -> Tensor<3> {
        let camera_idx = self.camera_indices[view_idx] as usize;
        ppisp_apply(
            self.exposure.val(),
            self.vignetting.val(),
            self.color.val(),
            self.crf.val(),
            img,
            camera_idx,
            view_idx,
            stages,
        )
    }

    /// Weighted parameter regularisation (the reference's six terms). Tiny
    /// tensors, so plain Burn ops on the autodiff backend.
    pub fn reg_loss(&self) -> Tensor<1> {
        let exposure = self.exposure.val();
        let vig = self.vignetting.val();
        let color = self.color.val();
        let crf = self.crf.val();
        let device = exposure.device();
        let [num_cameras, _, _] = vig.dims();

        // Exposure mean ~ 0 (resolves SH <-> exposure ambiguity).
        let loss = smooth_l1(exposure.mean(), 0.1) * W_EXPOSURE_MEAN;

        // Color mean ~ 0 across frames, in real-offset space.
        let pinv: Tensor<2> =
            Tensor::<1>::from_floats(COLOR_PINV_BLOCK_DIAG.as_slice(), &device).reshape([8, 8]);
        let offsets = color.matmul(pinv); // [F, 8]
        let color_mean = offsets.mean_dim(0); // [1, 8]
        let loss = loss + smooth_l1(color_mean, 0.005).mean() * W_COLOR_MEAN;

        // Vignetting center near image center.
        let centers = vig.clone().slice(burn::tensor::s![.., .., 0..2]);
        let loss =
            loss + centers.powi_scalar(2).sum() * (W_VIG_CENTER / (num_cameras as f32 * 3.0));

        // Vignetting alphas should be <= 0.
        let alphas = vig.clone().slice(burn::tensor::s![.., .., 2..5]);
        let loss = loss
            + burn::tensor::activation::relu(alphas).sum()
                * (W_VIG_NON_POS / (num_cameras as f32 * 9.0));

        // Similar vignetting across RGB channels (variance penalty).
        let vig_mean = vig.clone().mean_dim(1); // [C, 1, 5]
        let vig_dev = vig - vig_mean;
        let loss = loss
            + vig_dev.powi_scalar(2).sum() * (W_VIG_CHANNEL / (num_cameras as f32 * 5.0 * 3.0));

        // Similar CRF across RGB channels.
        let crf_mean = crf.clone().mean_dim(1); // [C, 1, 4]
        let crf_dev = crf - crf_mean;
        loss + crf_dev.powi_scalar(2).sum() * (W_CRF_CHANNEL / (num_cameras as f32 * 4.0 * 3.0))
    }
}
