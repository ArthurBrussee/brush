use brush_render::burn_glue::{
    AutodiffMain, FUSION_LOCK, unwrap_ad_wgpu_float, wrap_ad_wgpu_float, wrap_wgpu_float,
};
use brush_render::{
    MainBackend, MainBackendBase, SplatOps,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    sh::sh_coeffs_for_degree,
    shaders::helpers::ProjectUniforms,
};
use burn::{
    backend::{
        AutodiffBackend, Backend, BackendTensor, DispatchTensorKind, TensorMetadata,
        TensorPrimitive,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
        wgpu::WgpuRuntime,
    },
    module::Param,
    tensor::{DType, Shape, Tensor},
};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use glam::Vec3;

/// Intermediate gradients from the rasterize backward pass.
///
/// Sparse buffer of shape `[num_visible, 10]`, indexed by `compact_gid`.
/// Slots 0..8 are projected splat gradients, slot 8 is the raw opacity
/// gradient, slot 9 is the refinement weight.
#[derive(Debug, Clone)]
pub struct RasterizeGrads<B: Backend> {
    pub v_combined: FloatTensor<B>,
}

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone)]
pub struct SplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Backward pass trait mirroring [`SplatOps`].
pub trait SplatBwdOps<B: Backend>: SplatOps<B> {
    /// Backward pass for rasterization.
    /// Returns sparse `v_combined` [`num_visible`, 10] indexed by `compact_gid`.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<B>,
        projected_splats: FloatTensor<B>,
        compact_gid_from_isect: IntTensor<B>,
        tile_offsets: IntTensor<B>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<B>,
    ) -> RasterizeGrads<B>;

    /// Backward pass for projection.
    /// Reads sparse `v_combined` [`num_visible`, 9], writes dense outputs (scatter in kernel).
    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<B>,
        raw_opac: FloatTensor<B>,
        global_from_compact_gid: IntTensor<B>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<B>,
    ) -> SplatGrads<B>;
}

/// State saved during forward pass for backward computation.
#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    raw_opacity: FloatTensor<B>,

    projected_splats: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
    global_from_compact_gid: IntTensor<B>,

    out_img: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,

    render_mode: SplatRenderMode,
    background: Vec3,
    img_size: glam::UVec2,
}

#[derive(Debug)]
struct RenderBackwards;

const NUM_BWD_ARGS: usize = 4;

// Implement gradient registration when rendering backwards.
impl<B: Backend + SplatBwdOps<B>> Backward<B, NUM_BWD_ARGS> for RenderBackwards {
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

        let rasterize_grads = B::rasterize_bwd(
            state.out_img,
            state.projected_splats,
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.background,
            state.img_size,
            v_output,
        );

        let splat_grads = B::project_bwd(
            state.transforms,
            state.raw_opacity,
            state.global_from_compact_gid,
            state.project_uniforms,
            state.render_mode,
            rasterize_grads.v_combined,
        );

        if let Some(node) = transforms_parent {
            grads.register::<B>(node.id, splat_grads.v_transforms);
        }

        // v_refine_weight is already dense [num_points], written by the kernel.
        if let Some(node) = refine_weight {
            grads.register::<B>(node.id, splat_grads.v_refine_weight);
        }

        if let Some(node) = coeffs_parent {
            grads.register::<B>(node.id, splat_grads.v_coeffs);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<B>(node.id, splat_grads.v_raw_opac);
        }
    }
}

pub struct SplatOutputDiff {
    pub img: Tensor<3>,
    pub num_visible: u32,
    pub visible: Tensor<1>,
    pub max_radius: Tensor<1>,
    pub refine_weight_holder: Tensor<1>,
}

/// Lift a non-autodiff `Tensor<D>` (on a Wgpu device) into the autodiff
/// graph. Equivalent to `Tensor::from_inner` but additionally sets the
/// `checkpointing` field that upstream burn-dispatch's `from_inner` leaves
/// at `None`. Without that field set, ops on the resulting tensor hit
/// `unreachable!("Should only be called with autodiff.")`.
pub fn lift_to_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    let dispatch = t.into_primitive().tensor();
    match dispatch.kind {
        DispatchTensorKind::Wgpu(BackendTensor::Float(inner)) => {
            wrap_ad_wgpu_float(<AutodiffMain as AutodiffBackend>::from_inner(inner))
        }
        // Already autodiff — no-op.
        DispatchTensorKind::Autodiff(_) => Tensor::from_primitive(TensorPrimitive::Float(dispatch)),
        _ => panic!("expected Wgpu tensor to lift to autodiff"),
    }
}

/// Equivalent to `Module::train()` for [`Splats`], routing through
/// [`lift_to_autodiff`] so the autodiff `checkpointing` field is set. Use this
/// instead of `splats.train()` until upstream burn-dispatch fixes `from_inner`.
pub fn lift_splats_to_autodiff(splats: Splats) -> Splats {
    let mip = splats.render_mip;
    let (transforms_id, transforms, _) = splats.transforms.consume();
    let (sh_coeffs_id, sh_coeffs, _) = splats.sh_coeffs.consume();
    let (raw_opacity_id, raw_opacity, _) = splats.raw_opacities.consume();
    Splats {
        transforms: Param::initialized(transforms_id, lift_to_autodiff(transforms).require_grad()),
        sh_coeffs: Param::initialized(sh_coeffs_id, lift_to_autodiff(sh_coeffs).require_grad()),
        raw_opacities: Param::initialized(
            raw_opacity_id,
            lift_to_autodiff(raw_opacity).require_grad(),
        ),
        render_mip: mip,
    }
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
    let _lock = FUSION_LOCK.lock().await;

    splats.clone().validate_values().await;

    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "brush_render_bwd::render_splats requires an autodiff-enabled device"
    );

    let refine_weight_holder = Tensor::<1>::zeros([1], &device).require_grad();

    let transforms_ad = unwrap_ad_wgpu_float(splats.transforms.val());
    let sh_coeffs_ad = unwrap_ad_wgpu_float(splats.sh_coeffs.val());
    let raw_opac_ad = unwrap_ad_wgpu_float(splats.raw_opacities.val());
    let refine_weight_ad = unwrap_ad_wgpu_float(refine_weight_holder.clone());

    let prep_nodes = RenderBackwards
        .prepare::<NoCheckpointing>([
            transforms_ad.node.clone(),
            refine_weight_ad.node.clone(),
            sh_coeffs_ad.node.clone(),
            raw_opac_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    let transforms_inner: FloatTensor<MainBackend> = transforms_ad.primitive.clone();
    let sh_inner: FloatTensor<MainBackend> = sh_coeffs_ad.primitive;
    let raw_opac_inner: FloatTensor<MainBackend> = raw_opac_ad.primitive.clone();

    let output = <MainBackend as SplatOps<MainBackend>>::render(
        camera,
        img_size,
        transforms_inner.clone(),
        sh_inner,
        raw_opac_inner.clone(),
        render_mode,
        background,
        true,
    )
    .await;

    output.clone().validate().await;

    let num_visible = output.aux.num_visible;
    let visible_inner = output.aux.visible.clone();
    let max_radius_inner = output.aux.max_radius.clone();

    let img_ad: FloatTensor<AutodiffMain> = match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = GaussianBackwardState {
                transforms: transforms_inner,
                raw_opacity: raw_opac_inner,
                out_img: output.out_img.clone(),
                projected_splats: output.projected_splats,
                project_uniforms: output.project_uniforms,
                tile_offsets: output.aux.tile_offsets.clone(),
                compact_gid_from_isect: output.compact_gid_from_isect,
                render_mode,
                global_from_compact_gid: output.global_from_compact_gid,
                background,
                img_size,
            };
            prep.finish(state, output.out_img)
        }
        OpsKind::UnTracked(prep) => prep.finish(output.out_img),
    };

    SplatOutputDiff {
        img: wrap_ad_wgpu_float(img_ad),
        num_visible,
        visible: Tensor::from_inner(wrap_wgpu_float(visible_inner)),
        max_radius: Tensor::from_inner(wrap_wgpu_float(max_radius_inner)),
        refine_weight_holder,
    }
}

impl SplatBwdOps<Self> for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
    ) -> RasterizeGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            background: Vec3,
            img_size: glam::UVec2,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [
                    v_output,
                    out_img,
                    projected_splats,
                    compact_gid_from_isect,
                    tile_offsets,
                ] = inputs;

                let [v_combined] = outputs;

                let grads = <MainBackendBase as SplatBwdOps<MainBackendBase>>::rasterize_bwd(
                    h.get_float_tensor::<MainBackendBase>(out_img),
                    h.get_float_tensor::<MainBackendBase>(projected_splats),
                    h.get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                    h.get_int_tensor::<MainBackendBase>(tile_offsets),
                    self.background,
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(v_output),
                );

                h.register_float_tensor::<MainBackendBase>(&v_combined.id, grads.v_combined);
            }
        }

        // projected_splats is [num_visible, PROJECTED_LANES], so shape[0] gives num_visible.
        let num_visible_val = projected_splats.shape()[0] as u32;

        let client = v_output.client.clone();
        let num_visible = (num_visible_val as usize).max(1);

        let input_tensors = [
            v_output,
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
        ];

        let outputs = {
            let v_combined_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_visible, 10]),
                DType::F32,
            );
            let stream = OperationStreams::with_inputs(&input_tensors);
            let desc = CustomOpIr::new(
                "rasterize_bwd",
                &input_tensors.map(|t| t.into_ir()),
                &[v_combined_out],
            );
            let op = CustomOp {
                desc: desc.clone(),
                background,
                img_size,
            };
            client
                .register(stream, OperationIr::Custom(desc), op)
                .outputs()
        };

        let [v_combined] = outputs;

        RasterizeGrads { v_combined }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            render_mode: SplatRenderMode,
            project_uniforms: ProjectUniforms,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [transforms, raw_opac, global_from_compact_gid, v_combined_in] = inputs;

                let [v_transforms, v_coeffs, v_raw_opac, v_refine_weight] = outputs;

                let grads = <MainBackendBase as SplatBwdOps<MainBackendBase>>::project_bwd(
                    h.get_float_tensor::<MainBackendBase>(transforms),
                    h.get_float_tensor::<MainBackendBase>(raw_opac),
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    self.project_uniforms,
                    self.render_mode,
                    h.get_float_tensor::<MainBackendBase>(v_combined_in),
                );

                h.register_float_tensor::<MainBackendBase>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<MainBackendBase>(&v_coeffs.id, grads.v_coeffs);
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<MainBackendBase>(
                    &v_refine_weight.id,
                    grads.v_refine_weight,
                );
            }
        }

        let client = transforms.client.clone();
        let num_points = transforms.shape[0];
        let coeffs = sh_coeffs_for_degree(project_uniforms.sh_degree) as usize;

        let input_tensors = [transforms, raw_opac, global_from_compact_gid, v_combined];

        let outputs = {
            let v_transforms_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points, 10]),
                DType::F32,
            );
            let v_coeffs_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points, coeffs, 3]),
                DType::F32,
            );
            let v_raw_opac_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points]),
                DType::F32,
            );
            let v_refine_weight_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points]),
                DType::F32,
            );

            let stream = OperationStreams::with_inputs(&input_tensors);
            let desc = CustomOpIr::new(
                "project_bwd",
                &input_tensors.map(|t| t.into_ir()),
                &[
                    v_transforms_out,
                    v_coeffs_out,
                    v_raw_opac_out,
                    v_refine_weight_out,
                ],
            );

            client
                .register(
                    stream,
                    OperationIr::Custom(desc.clone()),
                    CustomOp {
                        desc,
                        render_mode,
                        project_uniforms,
                    },
                )
                .outputs()
        };

        let [v_transforms, v_coeffs, v_raw_opac, v_refine_weight] = outputs;

        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}
