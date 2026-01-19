use brush_render::{
    MainBackendBase, RenderAux, SplatOps,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    sh::{sh_coeffs_for_degree, sh_degree_from_coeffs},
    shaders::helpers::ProjectUniforms,
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

/// Intermediate gradients from the rasterize backward pass.
#[derive(Debug, Clone)]
pub struct RasterizeGrads<B: Backend> {
    /// Gradients w.r.t. projected splat data [`num_points`, 8].
    pub v_projected_splats: FloatTensor<B>,
    /// Gradients w.r.t. raw opacity from rasterization [`num_points`].
    pub v_raw_opac: FloatTensor<B>,
    /// Refinement weights for densification [`num_points`].
    pub v_refine_weight: FloatTensor<B>,
}

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone)]
pub struct SplatGrads<B: Backend> {
    pub v_means: FloatTensor<B>,
    pub v_quats: FloatTensor<B>,
    pub v_scales: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// Backward pass trait mirroring [`SplatOps`].
pub trait SplatBwdOps<B: Backend>: SplatOps<B> {
    /// Backward pass for rasterization.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<B>,
        projected_splats: FloatTensor<B>,
        global_from_compact_gid: IntTensor<B>,
        compact_gid_from_isect: IntTensor<B>,
        tile_offsets: IntTensor<B>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<B>,
    ) -> RasterizeGrads<B>;

    /// Backward pass for projection.
    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        means: FloatTensor<B>,
        log_scales: FloatTensor<B>,
        quats: FloatTensor<B>,
        raw_opac: FloatTensor<B>,
        num_visible: IntTensor<B>,
        global_from_compact_gid: IntTensor<B>,
        project_uniforms: ProjectUniforms,
        sh_degree: u32,
        render_mode: SplatRenderMode,
        rasterize_grads: RasterizeGrads<B>,
    ) -> SplatGrads<B>;
}

/// State saved during forward pass for backward computation.
#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    means: FloatTensor<B>,
    quats: FloatTensor<B>,
    log_scales: FloatTensor<B>,
    raw_opac: FloatTensor<B>,

    projected_splats: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
    num_visible: IntTensor<B>,
    global_from_compact_gid: IntTensor<B>,

    out_img: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,

    render_mode: SplatRenderMode,
    sh_degree: u32,
    background: Vec3,
    img_size: glam::UVec2,
}

#[derive(Debug)]
struct RenderBackwards;

const NUM_BWD_ARGS: usize = 6;

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
            mean_parent,
            refine_weight,
            log_scales_parent,
            quats_parent,
            coeffs_parent,
            raw_opacity_parent,
        ] = ops.parents;

        // Step 1: Rasterize backward
        let rasterize_grads = B::rasterize_bwd(
            state.out_img,
            state.projected_splats,
            state.global_from_compact_gid.clone(),
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.background,
            state.img_size,
            v_output,
        );

        // Step 2: Project backward
        let splat_grads = B::project_bwd(
            state.means,
            state.log_scales,
            state.quats,
            state.raw_opac,
            state.num_visible,
            state.global_from_compact_gid,
            state.project_uniforms,
            state.sh_degree,
            state.render_mode,
            rasterize_grads,
        );

        if let Some(node) = mean_parent {
            grads.register::<B>(node.id, splat_grads.v_means);
        }

        // Register the gradients for the dummy xy input.
        if let Some(node) = refine_weight {
            grads.register::<B>(node.id, splat_grads.v_refine_weight);
        }

        if let Some(node) = log_scales_parent {
            grads.register::<B>(node.id, splat_grads.v_scales);
        }

        if let Some(node) = quats_parent {
            grads.register::<B>(node.id, splat_grads.v_quats);
        }

        if let Some(node) = coeffs_parent {
            grads.register::<B>(node.id, splat_grads.v_coeffs);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<B>(node.id, splat_grads.v_raw_opac);
        }
    }
}

pub struct SplatOutputDiff<B: Backend> {
    pub img: FloatTensor<B>,
    pub render_aux: RenderAux<B>,
    pub refine_weight_holder: Tensor<B, 1>,
}

/// Render splats on a differentiable backend.
///
/// This is the main entry point for differentiable rendering, wrapping
/// the forward pass with autodiff support.
pub fn render_splats<B, C>(
    splats: &Splats<Autodiff<B, C>>,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> SplatOutputDiff<Autodiff<B, C>>
where
    B: Backend + SplatBwdOps<B>,
    C: CheckpointStrategy,
{
    splats.validate_values();

    let device = Tensor::<Autodiff<B, C>, 2>::from_primitive(TensorPrimitive::Float(
        splats.means.val().into_primitive().tensor(),
    ))
    .device();
    let refine_weight_holder = Tensor::<Autodiff<B, C>, 1>::zeros([1], &device).require_grad();

    // Prepare backward pass, and check if we even need to do it.
    let prep_nodes = RenderBackwards
        .prepare::<C>([
            splats.means.val().into_primitive().tensor().node,
            refine_weight_holder.clone().into_primitive().tensor().node,
            splats.log_scales.val().into_primitive().tensor().node,
            splats.rotations.val().into_primitive().tensor().node,
            splats.sh_coeffs.val().into_primitive().tensor().node,
            splats.raw_opacities.val().into_primitive().tensor().node,
        ])
        .compute_bound()
        .stateful();

    let means = splats
        .means
        .val()
        .into_primitive()
        .tensor()
        .into_primitive();
    let log_scales = splats
        .log_scales
        .val()
        .into_primitive()
        .tensor()
        .into_primitive();
    let quats = splats
        .rotations
        .val()
        .into_primitive()
        .tensor()
        .into_primitive();
    let sh_coeffs_dims = splats.sh_coeffs.dims();
    let sh_coeffs = splats
        .sh_coeffs
        .val()
        .into_primitive()
        .tensor()
        .into_primitive();
    let raw_opacity = splats
        .raw_opacities
        .val()
        .into_primitive()
        .tensor()
        .into_primitive();

    // First pass: project
    let project_output = <B as SplatOps<B>>::project(
        camera,
        img_size,
        means.clone(),
        log_scales.clone(),
        quats.clone(),
        sh_coeffs,
        raw_opacity.clone(),
        splats.render_mode,
    );

    // Sync readback of num_intersections
    let num_intersections = project_output.read_num_intersections();

    // Second pass: rasterize (with bwd_info = true)
    let (out_img, render_aux, compact_gid_from_isect) =
        <B as SplatOps<B>>::rasterize(&project_output, num_intersections, background, true);

    // Create wrapped render_aux for Autodiff backend
    let wrapped_render_aux = RenderAux::<Autodiff<B, C>> {
        num_visible: render_aux.num_visible.clone(),
        num_intersections: render_aux.num_intersections,
        visible: <Autodiff<B, C> as AutodiffBackend>::from_inner(render_aux.visible.clone()),
        tile_offsets: render_aux.tile_offsets.clone(),
        img_size: render_aux.img_size,
    };

    let sh_degree = sh_degree_from_coeffs(sh_coeffs_dims[1] as u32);

    match prep_nodes {
        OpsKind::Tracked(prep) => {
            // Save state needed for backward pass.
            let state = GaussianBackwardState {
                means,
                log_scales,
                quats,
                raw_opac: raw_opacity,
                sh_degree,
                out_img: out_img.clone(),
                projected_splats: project_output.projected_splats,
                project_uniforms: project_output.project_uniforms,
                num_visible: project_output.num_visible,
                tile_offsets: render_aux.tile_offsets,
                compact_gid_from_isect,
                render_mode: splats.render_mode,
                global_from_compact_gid: project_output.global_from_compact_gid,
                background,
                img_size,
            };

            let out_img = prep.finish(state, out_img);

            let result = SplatOutputDiff {
                img: out_img,
                render_aux: wrapped_render_aux,
                refine_weight_holder,
            };
            result.render_aux.validate();
            result
        }
        OpsKind::UnTracked(prep) => {
            // When no node is tracked, we can just use the original operation without
            // keeping any state.
            let result = SplatOutputDiff {
                img: prep.finish(out_img),
                render_aux: wrapped_render_aux,
                refine_weight_holder,
            };
            result.render_aux.validate();
            result
        }
    }
}

impl SplatBwdOps<Self> for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
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

        impl<BT: BoolElement> Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [
                    v_output,
                    out_img,
                    projected_splats,
                    global_from_compact_gid,
                    compact_gid_from_isect,
                    tile_offsets,
                ] = inputs;

                let [v_projected_splats, v_raw_opac, v_refine_weight] = outputs;

                let grads = <MainBackendBase as SplatBwdOps<MainBackendBase>>::rasterize_bwd(
                    h.get_float_tensor::<MainBackendBase>(out_img),
                    h.get_float_tensor::<MainBackendBase>(projected_splats),
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    h.get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                    h.get_int_tensor::<MainBackendBase>(tile_offsets),
                    self.background,
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(v_output),
                );

                h.register_float_tensor::<MainBackendBase>(
                    &v_projected_splats.id,
                    grads.v_projected_splats,
                );
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<MainBackendBase>(
                    &v_refine_weight.id,
                    grads.v_refine_weight,
                );
            }
        }

        let client = v_output.client.clone();
        let num_points = projected_splats.shape[0];

        let v_projected_splats_out = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 8]),
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
            out_img,
            projected_splats,
            global_from_compact_gid,
            compact_gid_from_isect,
            tile_offsets,
        ];

        let stream = OperationStreams::with_inputs(&input_tensors);
        let desc = CustomOpIr::new(
            "rasterize_bwd",
            &input_tensors.map(|t| t.into_ir()),
            &[v_projected_splats_out, v_raw_opac, v_refine_weight],
        );
        let op = CustomOp {
            desc: desc.clone(),
            background,
            img_size,
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [v_projected_splats, v_raw_opac, v_refine_weight] = outputs;

        RasterizeGrads {
            v_projected_splats,
            v_raw_opac,
            v_refine_weight,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        num_visible: IntTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        sh_degree: u32,
        render_mode: SplatRenderMode,
        rasterize_grads: RasterizeGrads<Self>,
    ) -> SplatGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            render_mode: SplatRenderMode,
            sh_degree: u32,
            project_uniforms: ProjectUniforms,
        }

        impl<BT: BoolElement> Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [
                    means,
                    log_scales,
                    quats,
                    raw_opac,
                    num_visible,
                    global_from_compact_gid,
                    v_projected_splats,
                    v_raw_opac_in,
                    v_refine_weight_in,
                ] = inputs;

                let [v_means, v_quats, v_scales, v_coeffs, v_raw_opac, v_refine_weight] = outputs;

                let inner_rasterize_grads = RasterizeGrads {
                    v_projected_splats: h.get_float_tensor::<MainBackendBase>(v_projected_splats),
                    v_raw_opac: h.get_float_tensor::<MainBackendBase>(v_raw_opac_in),
                    v_refine_weight: h.get_float_tensor::<MainBackendBase>(v_refine_weight_in),
                };

                let grads = <MainBackendBase as SplatBwdOps<MainBackendBase>>::project_bwd(
                    h.get_float_tensor::<MainBackendBase>(means),
                    h.get_float_tensor::<MainBackendBase>(log_scales),
                    h.get_float_tensor::<MainBackendBase>(quats),
                    h.get_float_tensor::<MainBackendBase>(raw_opac),
                    h.get_int_tensor::<MainBackendBase>(num_visible),
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    self.project_uniforms,
                    self.sh_degree,
                    self.render_mode,
                    inner_rasterize_grads,
                );

                h.register_float_tensor::<MainBackendBase>(&v_means.id, grads.v_means);
                h.register_float_tensor::<MainBackendBase>(&v_quats.id, grads.v_quats);
                h.register_float_tensor::<MainBackendBase>(&v_scales.id, grads.v_scales);
                h.register_float_tensor::<MainBackendBase>(&v_coeffs.id, grads.v_coeffs);
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<MainBackendBase>(
                    &v_refine_weight.id,
                    grads.v_refine_weight,
                );
            }
        }

        let client = means.client.clone();
        let num_points = means.shape[0];
        let coeffs = sh_coeffs_for_degree(sh_degree) as usize;

        let v_means_out = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 3]),
            DType::F32,
        );
        let v_scales_out = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 3]),
            DType::F32,
        );
        let v_quats_out = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, 4]),
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

        let input_tensors = [
            means,
            log_scales,
            quats,
            raw_opac,
            num_visible,
            global_from_compact_gid,
            rasterize_grads.v_projected_splats,
            rasterize_grads.v_raw_opac,
            rasterize_grads.v_refine_weight,
        ];

        let stream = OperationStreams::with_inputs(&input_tensors);
        let desc = CustomOpIr::new(
            "project_bwd",
            &input_tensors.map(|t| t.into_ir()),
            &[
                v_means_out,
                v_quats_out,
                v_scales_out,
                v_coeffs_out,
                v_raw_opac_out,
                v_refine_weight_out,
            ],
        );

        let outputs = client
            .register(
                stream,
                OperationIr::Custom(desc.clone()),
                CustomOp {
                    desc,
                    sh_degree,
                    render_mode,
                    project_uniforms,
                },
            )
            .outputs();

        let [v_means, v_quats, v_scales, v_coeffs, v_raw_opac, v_refine_weight] = outputs;

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
