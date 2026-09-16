#![allow(clippy::match_wildcard_for_single_variants)]

use crate::{
    SplatOps,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    sh::sh_coeffs_for_degree,
    shaders::helpers::ProjectUniforms,
};
use brush_cube::fusion::register_custom;
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::{
    backend::{
        AutodiffBackend, Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{DType, Shape, Tensor},
};
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use glam::Vec3;

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone)]
pub(crate) struct SplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Concrete backward kernels behind [`SplatOps::render`].
///
/// Deliberately not `: SplatOps`. This is the set of backends that have
/// backward kernels, which is smaller than the set you can render on:
/// `AutodiffMain` and the generated `Dispatch` impl `SplatOps` but have no
/// meaningful `rasterize_bwd`, since these run on concrete tensors and are
/// called from the `Backward` impl on the inner backend.
pub(crate) trait SplatBwdOps: Backend {
    /// Backward pass for rasterization. Returns the sparse `v_combined`
    /// buffer, `[num_visible, 10]` indexed by `compact_gid`: slots 0..8 are
    /// projected splat gradients, slot 8 the raw opacity gradient, slot 9 the
    /// refinement weight.
    #[allow(clippy::too_many_arguments)]
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

    /// Backward pass for projection.
    /// Reads sparse `v_combined` [`num_visible`, 10] and writes compact
    /// outputs: a zero row, then one row per visible splat in `compact_gid`
    /// order. The caller gathers them per global splat through
    /// `compact_from_global`.
    /// `sh_coeffs` is the original (input) SH coefficient tensor — needed
    /// so the kernel can backprop `v_color` through the SH basis to the
    /// view direction and then to the mean.
    #[allow(clippy::too_many_arguments)]
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

const NUM_BWD_ARGS: usize = 4;

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
        // gather expands each to the dense param shape. It's left as an
        // unexecuted stream op: burn's fusion folds it into the optimizer's
        // own kernel and nothing dense is materialised here.
        let inv = state.compact_from_global;
        let dense = |compact: FloatTensor<B>| B::float_select(compact, 0, inv.clone());

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

/// Like [`render_splats`] but lets the caller pick the
/// [`crate::gaussian_splats::RasterPass`]. Used by the finite-diff
/// test suite to enable the C^1 smooth-cutoff surrogate; production code
/// should use [`render_splats`].
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

impl SplatBwdOps for Fusion<CubeBackend> {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
    ) -> FloatTensor<Self> {
        // projected_splats is [num_visible, PROJECTED_LANES].
        let num_visible = projected_splats.shape()[0];
        let client = v_output.client.clone();
        let [v_combined] = register_custom(
            &client,
            "rasterize_bwd",
            [
                v_output,
                out_img,
                projected_splats,
                compact_gid_from_isect,
                tile_offsets,
            ],
            [(Shape::new([num_visible, 10]), DType::F32)],
            move |desc, h| {
                let (
                    [
                        v_output,
                        out_img,
                        projected_splats,
                        compact_gid_from_isect,
                        tile_offsets,
                    ],
                    [v_combined],
                ) = desc.as_fixed();
                let grads = <CubeBackend as SplatBwdOps>::rasterize_bwd(
                    h.get_float_tensor::<CubeBackend>(out_img),
                    h.get_float_tensor::<CubeBackend>(projected_splats),
                    h.get_int_tensor::<CubeBackend>(compact_gid_from_isect),
                    h.get_int_tensor::<CubeBackend>(tile_offsets),
                    background,
                    img_size,
                    h.get_float_tensor::<CubeBackend>(v_output),
                    smooth_cutoff,
                );
                h.register_float_tensor::<CubeBackend>(&v_combined.id, grads);
            },
        );
        v_combined
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> SplatGrads<Self> {
        let client = transforms.client.clone();
        let rows = project_uniforms.num_visible as usize + 1;
        let coeffs = sh_coeffs_for_degree(project_uniforms.sh_degree) as usize;
        let [v_transforms, v_coeffs, v_raw_opac, v_refine_weight] = register_custom(
            &client,
            "project_bwd",
            [
                transforms,
                sh_coeffs,
                raw_opac,
                min_scale,
                global_from_compact_gid,
                v_combined,
            ],
            [
                (Shape::new([rows, 10]), DType::F32),
                (Shape::new([rows, coeffs, 3]), DType::F32),
                (Shape::new([rows]), DType::F32),
                (Shape::new([rows]), DType::F32),
            ],
            move |desc, h| {
                let (
                    [
                        transforms,
                        sh_coeffs,
                        raw_opac,
                        min_scale,
                        global_from_compact_gid,
                        v_combined,
                    ],
                    [v_transforms, v_coeffs, v_raw_opac, v_refine_weight],
                ) = desc.as_fixed();
                let grads = <CubeBackend as SplatBwdOps>::project_bwd(
                    h.get_float_tensor::<CubeBackend>(transforms),
                    h.get_float_tensor::<CubeBackend>(sh_coeffs),
                    h.get_float_tensor::<CubeBackend>(raw_opac),
                    h.get_float_tensor::<CubeBackend>(min_scale),
                    has_min_scale,
                    h.get_int_tensor::<CubeBackend>(global_from_compact_gid),
                    project_uniforms,
                    render_mode,
                    h.get_float_tensor::<CubeBackend>(v_combined),
                );
                h.register_float_tensor::<CubeBackend>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<CubeBackend>(&v_coeffs.id, grads.v_coeffs);
                h.register_float_tensor::<CubeBackend>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<CubeBackend>(&v_refine_weight.id, grads.v_refine_weight);
            },
        );
        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}
