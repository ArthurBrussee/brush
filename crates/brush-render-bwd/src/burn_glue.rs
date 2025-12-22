use brush_render::{
    MainBackendBase, SplatForward,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    render_aux::RenderAux,
    sh::{sh_coeffs_for_degree, sh_degree_from_coeffs},
};
use burn::{
    backend::{
        Autodiff,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        wgpu::WgpuRuntime,
    },
    prelude::Backend,
    tensor::{
        DType, Shape, Tensor, TensorPrimitive,
        backend::AutodiffBackend,
        ops::{FloatTensor, IntTensor},
    },
};
use burn_cubecl::{BoolElement, fusion::FusionCubeRuntime};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use glam::Vec3;

use crate::render_bwd::SplatGrads;

/// Like [`SplatForward`], but for backends that support differentiation.
///
/// This shouldn't be a separate trait, but atm is needed because of orphan trait rules.
pub trait SplatForwardDiff<B: Backend> {
    /// Render splats to a buffer.
    ///
    /// This projects the gaussians, sorts them, and rasterizes them to a buffer, in a
    /// differentiable way.
    #[allow(clippy::too_many_arguments)]
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<B>,
        log_scales: FloatTensor<B>,
        quats: FloatTensor<B>,
        sh_coeffs: FloatTensor<B>,
        raw_opacity: FloatTensor<B>,
        render_mode: SplatRenderMode,
        background: Vec3,
    ) -> SplatOutputDiff<B>;
}

pub trait SplatBackwardOps<B: Backend> {
    /// Backward pass for `render_splats`.
    ///
    /// Do not use directly, `render_splats` will use this to calculate gradients.
    #[allow(unused_variables)]
    fn render_splats_bwd(
        state: GaussianBackwardState<B>,
        v_output: FloatTensor<B>,
    ) -> SplatGrads<B>;
}

#[derive(Debug, Clone)]
pub struct GaussianBackwardState<B: Backend> {
    pub(crate) means: FloatTensor<B>,
    pub(crate) quats: FloatTensor<B>,
    pub(crate) log_scales: FloatTensor<B>,
    pub(crate) raw_opac: FloatTensor<B>,
    pub(crate) out_img: FloatTensor<B>,
    pub(crate) projected_splats: FloatTensor<B>,
    pub(crate) uniforms_buffer: IntTensor<B>,
    pub(crate) compact_gid_from_isect: IntTensor<B>,
    pub(crate) global_from_compact_gid: IntTensor<B>,
    pub(crate) tile_offsets: IntTensor<B>,
    pub(crate) render_mode: SplatRenderMode,
    pub(crate) sh_degree: u32,
}

#[derive(Debug)]
struct RenderBackwards;

const NUM_BWD_ARGS: usize = 6;

// Implement gradient registration when rendering backwards.
impl<B: Backend + SplatBackwardOps<B>> Backward<B, NUM_BWD_ARGS> for RenderBackwards {
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
            mean_parent,
            refine_weight,
            log_scales_parent,
            quats_parent,
            coeffs_parent,
            raw_opacity_parent,
        ] = ops.parents;

        let v_tens = B::render_splats_bwd(state, v_output);

        if let Some(node) = mean_parent {
            grads.register::<B>(node.id, v_tens.v_means);
        }

        // Register the gradients for the dummy xy input.
        if let Some(node) = refine_weight {
            grads.register::<B>(node.id, v_tens.v_refine_weight);
        }

        if let Some(node) = log_scales_parent {
            grads.register::<B>(node.id, v_tens.v_scales);
        }

        if let Some(node) = quats_parent {
            grads.register::<B>(node.id, v_tens.v_quats);
        }

        if let Some(node) = coeffs_parent {
            grads.register::<B>(node.id, v_tens.v_coeffs);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<B>(node.id, v_tens.v_raw_opac);
        }
    }
}

pub struct SplatOutputDiff<B: Backend> {
    pub img: FloatTensor<B>,
    pub aux: RenderAux<B>,
    pub refine_weight_holder: Tensor<B, 1>,
}

// Implement
impl<B: Backend + SplatBackwardOps<B> + SplatForward<B>, C: CheckpointStrategy>
    SplatForwardDiff<Self> for Autodiff<B, C>
{
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacity: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
    ) -> SplatOutputDiff<Self> {
        // Get backend tensors & dequantize if needed. Could try and support quantized inputs
        // in the future.
        let device =
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(means.clone())).device();
        let refine_weight_holder = Tensor::<Self, 1>::zeros([1], &device).require_grad();

        // Prepare backward pass, and check if we even need to do it. Store nodes that need gradients.
        let prep_nodes = RenderBackwards
            .prepare::<C>([
                means.node.clone(),
                refine_weight_holder.clone().into_primitive().tensor().node,
                log_scales.node.clone(),
                quats.node.clone(),
                sh_coeffs.node.clone(),
                raw_opacity.node.clone(),
            ])
            .compute_bound()
            .stateful();

        // Render complete forward pass.
        let (out_img, aux) = <B as SplatForward<B>>::render_splats(
            camera,
            img_size,
            means.clone().into_primitive(),
            log_scales.clone().into_primitive(),
            quats.clone().into_primitive(),
            sh_coeffs.clone().into_primitive(),
            raw_opacity.clone().into_primitive(),
            render_mode,
            background,
            true,
        );

        let wrapped_aux = RenderAux::<Self> {
            projected_splats: <Self as AutodiffBackend>::from_inner(aux.projected_splats.clone()),
            num_intersections: aux.num_intersections,
            tile_offsets: aux.tile_offsets.clone(),
            compact_gid_from_isect: aux.compact_gid_from_isect.clone(),
            global_from_compact_gid: aux.global_from_compact_gid.clone(),
            uniforms_buffer: aux.uniforms_buffer.clone(),
            visible: <Self as AutodiffBackend>::from_inner(aux.visible),
            img_size: aux.img_size,
        };

        match prep_nodes {
            OpsKind::Tracked(prep) => {
                // Save state needed for backward pass.
                let state = GaussianBackwardState {
                    means: means.into_primitive(),
                    log_scales: log_scales.into_primitive(),
                    quats: quats.into_primitive(),
                    raw_opac: raw_opacity.into_primitive(),
                    sh_degree: sh_degree_from_coeffs(
                        Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(sh_coeffs)).dims()
                            [1] as u32,
                    ),
                    out_img: out_img.clone(),
                    projected_splats: aux.projected_splats,
                    uniforms_buffer: aux.uniforms_buffer,
                    tile_offsets: aux.tile_offsets,
                    compact_gid_from_isect: aux.compact_gid_from_isect,
                    render_mode,
                    global_from_compact_gid: aux.global_from_compact_gid,
                };

                let out_img = prep.finish(state, out_img);

                SplatOutputDiff {
                    img: out_img,
                    aux: wrapped_aux,
                    refine_weight_holder,
                }
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just use the original operation without
                // keeping any state.
                SplatOutputDiff {
                    img: prep.finish(out_img),
                    aux: wrapped_aux,
                    refine_weight_holder,
                }
            }
        }
    }
}

impl SplatBackwardOps<Self> for Fusion<MainBackendBase> {
    fn render_splats_bwd(
        state: GaussianBackwardState<Self>,
        v_output: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            render_mode: SplatRenderMode,
            sh_degree: u32,
        }

        impl<BT: BoolElement> Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [
                    v_output,
                    means,
                    quats,
                    log_scales,
                    raw_opac,
                    out_img,
                    projected_splats,
                    uniforms_buffer,
                    tile_offsets,
                    compact_gid_from_isect,
                    global_from_compact_gid,
                ] = inputs;

                let [v_means, v_quats, v_scales, v_coeffs, v_raw_opac, v_refine] = outputs;

                let inner_state = GaussianBackwardState {
                    means: h.get_float_tensor::<MainBackendBase>(means),
                    log_scales: h.get_float_tensor::<MainBackendBase>(log_scales),
                    quats: h.get_float_tensor::<MainBackendBase>(quats),
                    raw_opac: h.get_float_tensor::<MainBackendBase>(raw_opac),
                    out_img: h.get_float_tensor::<MainBackendBase>(out_img),
                    projected_splats: h.get_float_tensor::<MainBackendBase>(projected_splats),
                    uniforms_buffer: h.get_int_tensor::<MainBackendBase>(uniforms_buffer),
                    tile_offsets: h.get_int_tensor::<MainBackendBase>(tile_offsets),
                    compact_gid_from_isect: h
                        .get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                    global_from_compact_gid: h
                        .get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    sh_degree: self.sh_degree,
                    render_mode: self.render_mode,
                };

                let grads =
                    <MainBackendBase as SplatBackwardOps<MainBackendBase>>::render_splats_bwd(
                        inner_state,
                        h.get_float_tensor::<MainBackendBase>(v_output),
                    );

                // // Register output.
                h.register_float_tensor::<MainBackendBase>(&v_means.id, grads.v_means);
                h.register_float_tensor::<MainBackendBase>(&v_quats.id, grads.v_quats);
                h.register_float_tensor::<MainBackendBase>(&v_scales.id, grads.v_scales);
                h.register_float_tensor::<MainBackendBase>(&v_coeffs.id, grads.v_coeffs);
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<MainBackendBase>(&v_refine.id, grads.v_refine_weight);
            }
        }

        let client = v_output.client.clone();
        let num_points = state.means.shape[0];
        let coeffs = sh_coeffs_for_degree(state.sh_degree) as usize;

        let v_means = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 3]),
            DType::F32,
        );
        let v_scales = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 3]),
            DType::F32,
        );
        let v_quats = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 4]),
            DType::F32,
        );
        let v_coeffs = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, coeffs, 3]),
            DType::F32,
        );
        let v_raw_opac = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points]),
            DType::F32,
        );
        let v_refine_weight = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points]),
            DType::F32,
        );

        let input_tensors = [
            v_output,
            state.means,
            state.quats,
            state.log_scales,
            state.raw_opac,
            state.out_img,
            state.projected_splats,
            state.uniforms_buffer,
            state.tile_offsets,
            state.compact_gid_from_isect,
            state.global_from_compact_gid,
        ];

        let stream = OperationStreams::with_inputs(&input_tensors);
        let desc = CustomOpIr::new(
            "render_splat_bwd",
            &input_tensors.map(|t| t.into_ir()),
            &[
                v_means,
                v_quats,
                v_scales,
                v_coeffs,
                v_raw_opac,
                v_refine_weight,
            ],
        );

        let outputs = client
            .register(
                stream,
                OperationIr::Custom(desc.clone()),
                CustomOp {
                    // state,
                    desc,
                    sh_degree: state.sh_degree,
                    render_mode: state.render_mode,
                },
            )
            .outputs();

        let [
            v_means,
            v_quats,
            v_scales,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        ] = outputs;

        SplatGrads {
            v_means,
            v_scales,
            v_quats,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}

/// Render splats on a differentiable backend.
pub fn render_splats<B>(
    splats: &Splats<B>,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> SplatOutputDiff<B>
where
    B: Backend + SplatForwardDiff<B>,
{
    #[cfg(any(feature = "debug-validation", test))]
    splats.validate_values();

    let result = B::render_splats(
        camera,
        img_size,
        splats.means.val().into_primitive().tensor(),
        splats.log_scales.val().into_primitive().tensor(),
        splats.rotations.val().into_primitive().tensor(),
        splats.sh_coeffs.val().into_primitive().tensor(),
        splats.raw_opacities.val().into_primitive().tensor(),
        splats.render_mode,
        background,
    );

    #[cfg(any(feature = "debug-validation", test))]
    result.aux.validate_values();

    result
}
