//! Host side of the hybrid PPISP grid: backend trait + kernel launches,
//! Fusion dispatch and Burn autodiff op. The grid tensor layout matches the
//! affine bilateral grid (`[N, C, L, H, W]`), with `C` determined by the
//! payload; zero-filled grids (and zero vignetting params) are the identity
//! transform.

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
    tensor::{DType, Shape, Tensor},
};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use burn_fusion::Fusion;

use crate::bilagrid::grid_dims5;
use crate::ppisp_grid_kernels as kernels;
use crate::{CasAtomicAdd, GradSubsample, HfAtomicAdd, alloc_zeros, contiguous, dispatch_custom};

const NUM_VIG_GRADS: usize = kernels::NUM_VIG_GRADS as usize;

/// Which PPISP stages the fused grid pass applies (per-cell exposure is
/// always present).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridPayload {
    /// Latent color-homography offsets in the grid (8 channels).
    pub color: bool,
    /// CRF raw-param offsets from the identity tone curve in the grid
    /// (12 channels).
    pub crf: bool,
    /// Per-camera vignetting fused before the grid.
    pub vignetting: bool,
}

impl GridPayload {
    pub fn channels(self) -> usize {
        kernels::payload_channels(self.color, self.crf) as usize
    }
}

/// Backend hooks for the hybrid PPISP-grid kernels.
pub trait PpispGridOps<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn ppisp_grid_fwd(
        grids: FloatTensor<B>,
        vignetting: FloatTensor<B>,
        rgb: FloatTensor<B>,
        view_idx: usize,
        camera_idx: usize,
        payload: GridPayload,
    ) -> FloatTensor<B>;

    /// Returns `(dL/dgrids, vignetting partials [num_cubes, 15], dL/drgb)`;
    /// the grid gradient is full-size with only the active view's slice
    /// populated, and the caller reduces the vignetting partials.
    #[allow(clippy::too_many_arguments)]
    fn ppisp_grid_bwd_raw(
        grids: FloatTensor<B>,
        vignetting: FloatTensor<B>,
        rgb: FloatTensor<B>,
        v_out: FloatTensor<B>,
        view_idx: usize,
        camera_idx: usize,
        payload: GridPayload,
        subsample: GradSubsample,
    ) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>);
}

fn img_dims<R: CubeRuntime>(rgb: &CubeTensor<R>) -> (u32, u32, u32) {
    let dims = rgb.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "rgb must be [h, w, c]");
    let ch = dims[2] as u32;
    assert!(ch == 3 || ch == 4, "rgb must have 3 or 4 channels");
    (dims[0] as u32, dims[1] as u32, ch)
}

#[allow(clippy::too_many_arguments)]
fn launch_fwd<R: CubeRuntime>(
    grids: CubeTensor<R>,
    vignetting: CubeTensor<R>,
    rgb: CubeTensor<R>,
    view_idx: usize,
    camera_idx: usize,
    payload: GridPayload,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};

    let grids = contiguous(grids);
    let vignetting = contiguous(vignetting);
    let rgb = contiguous(rgb);
    let (n, gc, gl, gh, gw) = grid_dims5(&grids);
    assert_eq!(gc as usize, payload.channels(), "grid payload mismatch");
    let (h, w, ch) = img_dims(&rgb);
    assert!((view_idx as u32) < n, "view index out of range");

    let out = alloc_zeros(&rgb, rgb.shape(), DType::F32);
    let client = rgb.client.clone();
    kernels::ppisp_grid_fwd_kernel::launch::<R>(
        &client,
        CubeCount::Static((h * w).div_ceil(kernels::BLOCK_SIZE), 1, 1),
        CubeDim::new_1d(kernels::BLOCK_SIZE),
        grids.into_tensor_arg(),
        vignetting.into_tensor_arg(),
        rgb.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        gl,
        gh,
        gw,
        h,
        w,
        view_idx as u32 * gc * gl * gh * gw,
        ch,
        camera_idx as u32,
        ch == 4,
        payload.vignetting,
        payload.color,
        payload.crf,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn launch_bwd<R: CubeRuntime>(
    grids: CubeTensor<R>,
    vignetting: CubeTensor<R>,
    rgb: CubeTensor<R>,
    v_out: CubeTensor<R>,
    view_idx: usize,
    camera_idx: usize,
    payload: GridPayload,
    subsample: GradSubsample,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};

    let grids = contiguous(grids);
    let vignetting = contiguous(vignetting);
    let rgb = contiguous(rgb);
    let v_out = contiguous(v_out);
    let (n, gc, gl, gh, gw) = grid_dims5(&grids);
    assert_eq!(gc as usize, payload.channels(), "grid payload mismatch");
    let (h, w, ch) = img_dims(&rgb);
    assert!((view_idx as u32) < n, "view index out of range");
    let every = subsample.every.max(1);
    let num_cubes = (h * w).div_ceil(kernels::BLOCK_SIZE);

    let grad_grids = alloc_zeros(&grids, grids.shape(), DType::F32);
    let grad_rgb = alloc_zeros(&rgb, rgb.shape(), DType::F32);
    let vig_partials = alloc_zeros(
        &rgb,
        Shape::new([num_cubes as usize, NUM_VIG_GRADS]),
        DType::F32,
    );
    let client = rgb.client.clone();

    let cube_count = CubeCount::Static(num_cubes, 1, 1);
    let cube_dim = CubeDim::new_1d(kernels::BLOCK_SIZE);
    let grid_offset = view_idx as u32 * gc * gl * gh * gw;

    macro_rules! launch {
        ($atomic:ty) => {
            kernels::ppisp_grid_bwd_kernel::launch::<$atomic, R>(
                &client,
                cube_count,
                cube_dim,
                grids.into_tensor_arg(),
                vignetting.into_tensor_arg(),
                rgb.into_tensor_arg(),
                v_out.into_tensor_arg(),
                grad_grids.clone().into_tensor_arg(),
                grad_rgb.clone().into_tensor_arg(),
                vig_partials.clone().into_tensor_arg(),
                gl,
                gh,
                gw,
                h,
                w,
                grid_offset,
                ch,
                camera_idx as u32,
                every,
                subsample.seed,
                ch == 4,
                payload.vignetting,
                payload.color,
                payload.crf,
            )
        };
    }
    if brush_cube::supports_float_atomics::<R>(&client) {
        launch!(HfAtomicAdd);
    } else {
        launch!(CasAtomicAdd);
    }
    (grad_grids, vig_partials, grad_rgb)
}

impl PpispGridOps<Self> for MainBackendBase {
    fn ppisp_grid_fwd(
        grids: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        view_idx: usize,
        camera_idx: usize,
        payload: GridPayload,
    ) -> FloatTensor<Self> {
        launch_fwd(grids, vignetting, rgb, view_idx, camera_idx, payload)
    }

    fn ppisp_grid_bwd_raw(
        grids: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        v_out: FloatTensor<Self>,
        view_idx: usize,
        camera_idx: usize,
        payload: GridPayload,
        subsample: GradSubsample,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        launch_bwd(
            grids, vignetting, rgb, v_out, view_idx, camera_idx, payload, subsample,
        )
    }
}

impl PpispGridOps<Self> for Fusion<MainBackendBase> {
    fn ppisp_grid_fwd(
        grids: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        view_idx: usize,
        camera_idx: usize,
        payload: GridPayload,
    ) -> FloatTensor<Self> {
        let shape = rgb.shape();
        let [out] = dispatch_custom(
            "ppisp_grid_fwd",
            [grids, vignetting, rgb],
            [(shape, DType::F32)],
            move |desc, h| {
                let ([grids, vignetting, rgb], [out]) = desc.as_fixed();
                let res = MainBackendBase::ppisp_grid_fwd(
                    h.get_float_tensor::<MainBackendBase>(grids),
                    h.get_float_tensor::<MainBackendBase>(vignetting),
                    h.get_float_tensor::<MainBackendBase>(rgb),
                    view_idx,
                    camera_idx,
                    payload,
                );
                h.register_float_tensor::<MainBackendBase>(&out.id, res);
            },
        );
        out
    }

    fn ppisp_grid_bwd_raw(
        grids: FloatTensor<Self>,
        vignetting: FloatTensor<Self>,
        rgb: FloatTensor<Self>,
        v_out: FloatTensor<Self>,
        view_idx: usize,
        camera_idx: usize,
        payload: GridPayload,
        subsample: GradSubsample,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let grids_shape = grids.shape();
        let rgb_shape = rgb.shape();
        let [h_dim, w_dim, _] = rgb_shape.dims();
        let num_cubes = ((h_dim * w_dim) as u32).div_ceil(kernels::BLOCK_SIZE) as usize;
        let [grad_grids, vig_partials, grad_rgb] = dispatch_custom(
            "ppisp_grid_bwd",
            [grids, vignetting, rgb, v_out],
            [
                (grids_shape, DType::F32),
                (Shape::new([num_cubes, NUM_VIG_GRADS]), DType::F32),
                (rgb_shape, DType::F32),
            ],
            move |desc, h| {
                let ([grids, vignetting, rgb, v_out], [grad_grids, vig_partials, grad_rgb]) =
                    desc.as_fixed();
                let (gg, vp, gr) = MainBackendBase::ppisp_grid_bwd_raw(
                    h.get_float_tensor::<MainBackendBase>(grids),
                    h.get_float_tensor::<MainBackendBase>(vignetting),
                    h.get_float_tensor::<MainBackendBase>(rgb),
                    h.get_float_tensor::<MainBackendBase>(v_out),
                    view_idx,
                    camera_idx,
                    payload,
                    subsample,
                );
                h.register_float_tensor::<MainBackendBase>(&grad_grids.id, gg);
                h.register_float_tensor::<MainBackendBase>(&vig_partials.id, vp);
                h.register_float_tensor::<MainBackendBase>(&grad_rgb.id, gr);
            },
        );
        #[allow(clippy::tuple_array_conversions)]
        (grad_grids, vig_partials, grad_rgb)
    }
}

/// Full backward: kernel launch + reduce the vignetting partials into a
/// full-shape `[num_cameras, 3, 5]` gradient at the active camera row.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ppisp_grid_backward<B: Backend + PpispGridOps<B>>(
    grids: FloatTensor<B>,
    vignetting: FloatTensor<B>,
    rgb: FloatTensor<B>,
    v_out: FloatTensor<B>,
    view_idx: usize,
    camera_idx: usize,
    payload: GridPayload,
    subsample: GradSubsample,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
    use burn::tensor::Slice;
    let sl = |r: std::ops::Range<usize>| -> Slice { r.into() };

    let device = B::float_device(&rgb);
    let num_cameras = vignetting.shape().dims::<3>()[0];

    let (grad_grids, vig_partials, grad_rgb) = B::ppisp_grid_bwd_raw(
        grids, vignetting, rgb, v_out, view_idx, camera_idx, payload, subsample,
    );

    let summed = B::float_sum_dim(vig_partials, 0); // [1, 15]
    let g_vig = B::float_zeros(
        Shape::new([num_cameras, 3, 5]),
        &device,
        burn::tensor::FloatDType::F32,
    );
    let g_vig = B::float_slice_assign(
        g_vig,
        &[sl(camera_idx..camera_idx + 1), sl(0..3), sl(0..5)],
        B::float_reshape(summed, Shape::new([1, 3, 5])),
    );

    (grad_grids, g_vig, grad_rgb)
}

// ---------------------------------------------------------------------------
// Autodiff op
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct PpispGridBackward;

#[derive(Debug, Clone)]
struct PpispGridState<B: Backend> {
    grids: FloatTensor<B>,
    vignetting: FloatTensor<B>,
    rgb: FloatTensor<B>,
    view_idx: usize,
    camera_idx: usize,
    payload: GridPayload,
    subsample: GradSubsample,
}

impl<B: Backend + PpispGridOps<B>> Backward<B, 3> for PpispGridBackward {
    type State = PpispGridState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let v_out = grads.consume::<B>(&ops.node);
        let [grids_parent, vig_parent, rgb_parent] = ops.parents;
        let (grad_grids, grad_vig, grad_rgb) = ppisp_grid_backward::<B>(
            state.grids,
            state.vignetting,
            state.rgb,
            v_out,
            state.view_idx,
            state.camera_idx,
            state.payload,
            state.subsample,
        );
        if let Some(node) = grids_parent {
            grads.register::<B>(node.id, grad_grids);
        }
        if let Some(node) = vig_parent {
            grads.register::<B>(node.id, grad_vig);
        }
        if let Some(node) = rgb_parent {
            grads.register::<B>(node.id, grad_rgb);
        }
    }
}

/// Apply per-camera vignetting + the `view_idx`-th PPISP grid to a rendered
/// `[H, W, 3|4]` image (alpha untouched). Differentiable w.r.t. `grids`,
/// `vignetting` (`[num_cameras, 3, 5]`) and `rgb`.
#[allow(clippy::too_many_arguments)]
pub fn ppisp_grid_apply(
    grids: Tensor<5>,
    vignetting: Tensor<3>,
    rgb: Tensor<3>,
    view_idx: usize,
    camera_idx: usize,
    payload: GridPayload,
    subsample: GradSubsample,
) -> Tensor<3> {
    let grids_ad = unwrap_ad_wgpu_float(grids);
    let vig_ad = unwrap_ad_wgpu_float(vignetting);
    let rgb_ad = unwrap_ad_wgpu_float(rgb);

    let prep = PpispGridBackward
        .prepare::<NoCheckpointing>([
            grids_ad.node.clone(),
            vig_ad.node.clone(),
            rgb_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();

    let grids_p = grids_ad.primitive;
    let vig_p = vig_ad.primitive;
    let rgb_p = rgb_ad.primitive;
    let out = <MainBackend as PpispGridOps<MainBackend>>::ppisp_grid_fwd(
        grids_p.clone(),
        vig_p.clone(),
        rgb_p.clone(),
        view_idx,
        camera_idx,
        payload,
    );

    let out_ad: FloatTensor<AutodiffMain> = match prep {
        OpsKind::Tracked(prep) => prep.finish(
            PpispGridState {
                grids: grids_p,
                vignetting: vig_p,
                rgb: rgb_p,
                view_idx,
                camera_idx,
                payload,
                subsample,
            },
            out,
        ),
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    wrap_ad_wgpu_float::<3>(out_ad)
}

/// Forward-only version of [`ppisp_grid_apply`] for non-autodiff tensors
/// (eval-time correction).
pub fn ppisp_grid_apply_inner(
    grids: Tensor<5>,
    vignetting: Tensor<3>,
    rgb: Tensor<3>,
    view_idx: usize,
    camera_idx: usize,
    payload: GridPayload,
) -> Tensor<3> {
    use brush_render::burn_glue::unwrap_wgpu_float;
    let out = <MainBackend as PpispGridOps<MainBackend>>::ppisp_grid_fwd(
        unwrap_wgpu_float(grids),
        unwrap_wgpu_float(vignetting),
        unwrap_wgpu_float(rgb),
        view_idx,
        camera_idx,
        payload,
    );
    wrap_wgpu_float::<3>(out)
}
