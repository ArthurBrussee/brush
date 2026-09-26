#![allow(clippy::match_wildcard_for_single_variants)]

use crate::{
    SplatOps,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    sh::sh_coeffs_for_degree,
    shaders::helpers::ProjectUniforms,
};
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::{Autodiff, ExtensionType, fusion::custom::TensorSpec};
use burn::{
    backend::{
        AutodiffBackend, Backend,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{DType, Shape, Tensor},
};
use glam::Vec3;

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone, ExtensionType)]
#[extension_type(fusion)]
pub(crate) struct SplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Concrete backward kernels behind [`SplatOps::render`]. Deliberately not
/// `: SplatOps`: fewer backends have backward kernels than can render, and
/// these run on concrete tensors from the inner backend's `Backward` impl.
/// Wrapped so the allow reaches the generated Fusion impl, which binds every
/// ordinary argument whether its metadata expression reads it or not.
mod bwd_ops {
    #![allow(unused_variables)]
    use super::{ProjectUniforms, SplatGrads, SplatRenderMode, project_bwd_metadata};
    use burn::backend::{
        Backend,
        tensor::{FloatTensor, IntTensor},
    };
    use burn::tensor::Shape;
    use glam::Vec3;

    #[burn::backend::backend_extension(Fusion)]
    pub(crate) trait SplatBwdOps: Backend {
        /// Returns sparse `v_combined` `[num_visible, 10]` indexed by
        /// `compact_gid`: eight projected-splat gradients, then the raw opacity
        /// gradient and the refinement weight.
        #[allow(clippy::too_many_arguments)]
        #[fusion(dtype = v_output, shape = Shape::new([projected_splats[0], 10]))]
        fn rasterize_bwd(
            out_img: FloatTensor<Self>,
            projected_splats: FloatTensor<Self>,
            compact_gid_from_isect: IntTensor<Self>,
            tile_offsets: IntTensor<Self>,
            background: Vec3,
            img_size: glam::UVec2,
            v_output: FloatTensor<Self>,
            smooth_cutoff: bool,
        ) -> FloatTensor<Self>;

        /// Writes a zero row, then one row per visible splat in `compact_gid`
        /// order; the caller gathers those per global splat through
        /// `compact_from_global`. `sh_coeffs` is the forward's input, so the
        /// kernel can backprop `v_color` through the SH basis to the mean.
        #[allow(clippy::too_many_arguments)]
        #[fusion(meta = project_bwd_metadata)]
        fn project_bwd(
            transforms: FloatTensor<Self>,
            sh_coeffs: FloatTensor<Self>,
            raw_opac: FloatTensor<Self>,
            min_scale: FloatTensor<Self>,
            has_min_scale: bool,
            global_from_compact_gid: IntTensor<Self>,
            project_uniforms: ProjectUniforms,
            render_mode: SplatRenderMode,
            v_combined: FloatTensor<Self>,
        ) -> SplatGrads<Self>;
    }
}
pub(crate) use bwd_ops::SplatBwdOps;

/// Output shapes, which Fusion needs before the kernel runs. The visible count
/// and SH degree are already host values, so none of this waits on a readback.
fn project_bwd_metadata(
    _transforms: &TensorSpec,
    _sh_coeffs: &TensorSpec,
    _raw_opac: &TensorSpec,
    _min_scale: &TensorSpec,
    _has_min_scale: &bool,
    _global_from_compact_gid: &TensorSpec,
    project_uniforms: &ProjectUniforms,
    _render_mode: &SplatRenderMode,
    _v_combined: &TensorSpec,
) -> SplatGradsMetadata {
    let rows = project_uniforms.num_visible as usize + 1;
    let coeffs = sh_coeffs_for_degree(project_uniforms.sh_degree) as usize;
    let f32_spec = |shape: Shape| TensorSpec::new(shape, DType::F32);
    SplatGradsMetadata {
        v_transforms: f32_spec(Shape::new([rows, 10])),
        v_coeffs: f32_spec(Shape::new([rows, coeffs, 3])),
        v_raw_opac: f32_spec(Shape::new([rows])),
        v_refine_weight: f32_spec(Shape::new([rows])),
    }
}

/// State saved during forward pass for backward computation.
#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    sh_coeffs: FloatTensor<B>,
    raw_opacity: FloatTensor<B>,
    min_scale: FloatTensor<B>,
    has_min_scale: bool,

    projected_splats: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
    global_from_compact_gid: IntTensor<B>,
    compact_from_global: IntTensor<B>,

    out_img: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,

    render_mode: SplatRenderMode,
    pass: crate::gaussian_splats::RasterPass,
    background: Vec3,
    img_size: glam::UVec2,
}

#[derive(Debug)]
struct RenderBackwards;

const NUM_BWD_ARGS: usize = 5;

// Implement gradient registration when rendering backwards.
impl<B: Backend + SplatBwdOps> Backward<B, NUM_BWD_ARGS> for RenderBackwards {
    type State = GaussianBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, NUM_BWD_ARGS>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_gaussians backwards").entered();

        let state = ops.state;
        let v_output = grads.consume::<B>(&ops.node);

        // Register gradients for parent nodes (This code is already skipped entirely
        // if no parent nodes require gradients).
        let [
            transforms_parent,
            refine_weight,
            coeffs_parent,
            raw_opacity_parent,
            coeffs_grad_sq_parent,
        ] = ops.parents;

        let v_combined = B::rasterize_bwd(
            state.out_img,
            state.projected_splats,
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.background,
            state.img_size,
            v_output,
            state.pass.smooth_cutoff(),
        );

        let splat_grads = B::project_bwd(
            state.transforms,
            state.sh_coeffs,
            state.raw_opacity,
            state.min_scale,
            state.has_min_scale,
            state.global_from_compact_gid,
            state.project_uniforms,
            state.render_mode,
            v_combined,
        );

        // The kernels write compact gradients with a zero row in front, and
        // `compact_from_global` points culled splats at that row, so one
        // gather expands each to the dense param shape. burn runs each as its
        // own fused kernel rather than folding it into the optimizer's, so the
        // dense gradient is still written, but only once and with no
        // zero-fill. `tests/fusion.rs` holds that shape in place.
        let inv = state.compact_from_global;
        let dense = |compact: FloatTensor<B>| B::float_select(compact, 0, inv.clone());
        let compact_coeffs = splat_grads.v_coeffs.clone();

        if let Some(node) = transforms_parent {
            grads.register::<B>(node.id, dense(splat_grads.v_transforms));
        }

        if let Some(node) = refine_weight {
            grads.register::<B>(node.id, dense(splat_grads.v_refine_weight));
        }

        if let Some(node) = coeffs_parent {
            grads.register::<B>(node.id, dense(splat_grads.v_coeffs));
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<B>(node.id, dense(splat_grads.v_raw_opac));
        }

        // The SH second moment Adam wants is a mean over each splat's
        // coefficients. Reducing the compact rows and gathering the small
        // result skips squaring the dense `[N, coeffs, 3]` gradient, which
        // burn will not fuse into the reduce that consumes it.
        if let Some(node) = coeffs_grad_sq_parent {
            let coeffs = state.project_uniforms.sh_degree;
            let trailing = (sh_coeffs_for_degree(coeffs) * 3) as f32;
            let sq = B::float_mul(compact_coeffs.clone(), compact_coeffs);
            let sq = B::float_sum_dims(sq, &[1, 2]);
            let mean = B::float_div_scalar(sq, burn::tensor::Scalar::Float(trailing as f64));
            grads.register::<B>(node.id, B::float_select(mean, 0, inv));
        }
    }
}

pub struct SplatOutputDiff {
    /// Rendered image, on the autodiff graph (this is what the loss backprops through).
    pub img: Tensor<3>,
    pub num_visible: u32,
    /// Per-splat visibility aux — on the **inner** backend (no gradients).
    pub visible: Tensor<1>,
    /// Per-splat max screen radius aux — on the **inner** backend (no gradients).
    pub max_radius: Tensor<1>,
    /// Per-splat opacity with the scale floor folded in — on the **inner**
    /// backend (no gradients). Zero for culled splats.
    pub opacities: Tensor<1>,
    pub refine_weight_holder: Tensor<1>,
    /// Catches the per-splat mean square of the SH gradient (see
    /// [`SplatOps::render`]). Its gradient is `[N, 1, 1]`, broadcasting over
    /// the coefficients, so it feeds Adam's second moment directly.
    /// This statistic is for one render/backward; it cannot be added across
    /// renders to obtain the mean square of an accumulated SH gradient.
    pub coeffs_grad_sq_holder: Tensor<3>,
}

/// Render splats on a differentiable device.
///
/// Panics if the device is not autodiff-enabled.
pub async fn render_splats(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> SplatOutputDiff {
    render_splats_with_pass(
        splats,
        camera,
        img_size,
        background,
        crate::gaussian_splats::RasterPass::Backward,
    )
    .await
}

/// Like [`render_splats`] but picks the
/// [`crate::gaussian_splats::RasterPass`]. Only the finite-diff tests need
/// this, for the C^1 smooth-cutoff surrogate.
pub async fn render_splats_with_pass(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
) -> SplatOutputDiff {
    splats.clone().validate_values().await;

    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "brush_render::bwd::render_splats requires an autodiff-enabled device"
    );

    let refine_weight_holder = Tensor::<1>::zeros([1], &device).require_grad();
    let coeffs_grad_sq_holder = Tensor::<3>::zeros([1, 1, 1], &device).require_grad();

    // The 3D-filter floor is applied inside the projection kernels. It lives
    // on the inner backend and carries no gradient, so lifting it onto the
    // autodiff device is just a wrap.
    let (min_scale, has_min_scale) = splats.min_scale_arg();

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    assert!(
        pass.bwd_info(),
        "render_splats_with_pass requires a Backward variant"
    );

    let output = <burn::backend::Dispatch as SplatOps>::render(
        camera,
        img_size,
        splats.transforms.val().into_dispatch(),
        splats.sh_coeffs.val().into_dispatch(),
        splats.raw_opacities.val().into_dispatch(),
        min_scale.autodiff().into_dispatch(),
        has_min_scale,
        refine_weight_holder.clone().into_dispatch(),
        coeffs_grad_sq_holder.clone().into_dispatch(),
        render_mode,
        background,
        pass,
    )
    .await;

    SplatOutputDiff {
        img: Tensor::from_dispatch(output.out_img),
        num_visible: output.aux.num_visible,
        visible: Tensor::from_dispatch(output.aux.visible).without_autodiff(),
        max_radius: Tensor::from_dispatch(output.aux.max_radius).without_autodiff(),
        opacities: Tensor::from_dispatch(output.aux.opacities).without_autodiff(),
        refine_weight_holder,
        coeffs_grad_sq_holder,
    }
}

impl<B: Backend + SplatOps + SplatBwdOps, C: CheckpointStrategy> SplatOps for Autodiff<B, C> {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        refine_weight: FloatTensor<Self>,
        coeffs_grad_sq: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> crate::RenderOutput<Self> {
        let prep_nodes = RenderBackwards
            .prepare::<NoCheckpointing>([
                transforms.node(),
                refine_weight.node(),
                sh_coeffs.node(),
                raw_opacities.node(),
                coeffs_grad_sq.node(),
            ])
            .compute_bound()
            .stateful();

        let transforms_inner: FloatTensor<B> = transforms.primitive().clone();
        let sh_inner: FloatTensor<B> = sh_coeffs.into_primitive();
        let raw_opac_inner: FloatTensor<B> = raw_opacities.primitive().clone();
        let min_scale_inner: FloatTensor<B> = min_scale.into_primitive();

        let output = <B as SplatOps>::render(
            camera,
            img_size,
            transforms_inner.clone(),
            sh_inner.clone(),
            raw_opac_inner.clone(),
            min_scale_inner.clone(),
            has_min_scale,
            refine_weight.into_primitive(),
            coeffs_grad_sq.into_primitive(),
            render_mode,
            background,
            pass,
        )
        .await;

        output.clone().validate().await;

        let img_ad: FloatTensor<Self> = match prep_nodes {
            OpsKind::Tracked(prep) => {
                let state = GaussianBackwardState {
                    transforms: transforms_inner,
                    sh_coeffs: sh_inner,
                    raw_opacity: raw_opac_inner,
                    min_scale: min_scale_inner,
                    has_min_scale,
                    out_img: output.out_img.clone(),
                    projected_splats: output.projected_splats.clone(),
                    project_uniforms: output.project_uniforms,
                    tile_offsets: output.aux.tile_offsets.clone(),
                    compact_gid_from_isect: output.compact_gid_from_isect.clone(),
                    render_mode,
                    pass,
                    global_from_compact_gid: output.global_from_compact_gid.clone(),
                    compact_from_global: output.compact_from_global.clone(),
                    background,
                    img_size,
                };
                prep.finish(state, output.out_img)
            }
            OpsKind::UnTracked(prep) => prep.finish(output.out_img),
        };

        // Lift the remaining float aux onto the autodiff graph. None of these
        // carry a backward — they only feed refine bookkeeping — but the
        // extension trait's output is uniformly `RenderOutput<Self>`, so they
        // ride along as untracked autodiff tensors. Int tensors share the
        // inner backend's primitive and pass through unchanged.
        let lift = <Self as AutodiffBackend>::from_inner;

        crate::RenderOutput {
            out_img: img_ad,
            aux: crate::RenderAuxInner {
                num_visible: output.aux.num_visible,
                num_intersections: output.aux.num_intersections,
                visible: lift(output.aux.visible),
                max_radius: lift(output.aux.max_radius),
                opacities: lift(output.aux.opacities),
                tile_offsets: output.aux.tile_offsets,
                img_size: output.aux.img_size,
            },
            projected_splats: lift(output.projected_splats),
            compact_gid_from_isect: output.compact_gid_from_isect,
            project_uniforms: output.project_uniforms,
            global_from_compact_gid: output.global_from_compact_gid,
            compact_from_global: output.compact_from_global,
        }
    }
}
