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
        DType, Shape, Tensor, TensorMetadata, TensorPrimitive,
        backend::AutodiffBackend,
        ops::{FloatTensor, IntTensor},
    },
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
/// Sparse buffer of shape [`num_visible`, 10], indexed by `compact_gid`:
///   [0..8]: projected splat gradients
///   [8]:    raw opacity gradient
///   [9]:    refinement weight
#[derive(Debug, Clone)]
pub struct RasterizeGrads<B: Backend> {
    pub v_combined: FloatTensor<B>,
}

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone)]
pub struct SplatGrads<B: Backend> {
    /// Gradients w.r.t. transforms [`num_points`, 10] = means(3) + quats(4) + log scales(3).
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
        sh_degree: u32,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<B>,
    ) -> SplatGrads<B>;
}

/// State saved during forward pass for backward computation.
#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    raw_opac: FloatTensor<B>,

    projected_splats: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
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
            state.raw_opac,
            state.global_from_compact_gid,
            state.project_uniforms,
            state.sh_degree,
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

pub struct SplatOutputDiff<B: Backend> {
    pub img: FloatTensor<B>,
    pub render_aux: RenderAux<B>,
    pub refine_weight_holder: Tensor<B, 1>,
}

/// Render splats on a differentiable backend.
///
/// This is the main entry point for differentiable rendering, wrapping
/// the forward pass with autodiff support.
///
/// Takes ownership of the splats. Clone before calling if you need to reuse them.
pub async fn render_splats<B, C>(
    splats: Splats<Autodiff<B, C>>,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> SplatOutputDiff<Autodiff<B, C>>
where
    B: Backend + SplatBwdOps<B>,
    C: CheckpointStrategy,
{
    splats.clone().validate_values().await;

    let device = Tensor::<Autodiff<B, C>, 2>::from_primitive(TensorPrimitive::Float(
        splats.transforms.val().into_primitive().tensor(),
    ))
    .device();
    let refine_weight_holder = Tensor::<Autodiff<B, C>, 1>::zeros([1], &device).require_grad();

    let prep_nodes = RenderBackwards
        .prepare::<C>([
            splats.transforms.val().into_primitive().tensor().node,
            refine_weight_holder.clone().into_primitive().tensor().node,
            splats.sh_coeffs.val().into_primitive().tensor().node,
            splats.raw_opacities.val().into_primitive().tensor().node,
        ])
        .compute_bound()
        .stateful();

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

    let transforms = splats
        .transforms
        .val()
        .into_primitive()
        .tensor()
        .into_primitive();

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    let output = <B as SplatOps<B>>::render(
        camera,
        img_size,
        transforms.clone(),
        sh_coeffs,
        raw_opacity.clone(),
        render_mode,
        background,
        true,
    )
    .await;

    output.clone().validate().await;

    let wrapped_render_aux = RenderAux::<Autodiff<B, C>> {
        num_visible: output.aux.num_visible,
        num_intersections: output.aux.num_intersections,
        visible: <Autodiff<B, C> as AutodiffBackend>::from_inner(output.aux.visible.clone()),
        tile_offsets: output.aux.tile_offsets.clone(),
        img_size: output.aux.img_size,
    };

    let sh_degree = sh_degree_from_coeffs(sh_coeffs_dims[1] as u32);

    match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = GaussianBackwardState {
                transforms,
                raw_opac: raw_opacity,
                sh_degree,
                out_img: output.out_img.clone(),
                projected_splats: output.projected_splats,
                project_uniforms: output.project_uniforms,
                tile_offsets: output.aux.tile_offsets,
                compact_gid_from_isect: output.compact_gid_from_isect,
                render_mode,
                global_from_compact_gid: output.global_from_compact_gid,
                background,
                img_size,
            };

            let out_img = prep.finish(state, output.out_img);

            SplatOutputDiff {
                img: out_img,
                render_aux: wrapped_render_aux,
                refine_weight_holder,
            }
        }
        OpsKind::UnTracked(prep) => SplatOutputDiff {
            img: prep.finish(output.out_img),
            render_aux: wrapped_render_aux,
            refine_weight_holder,
        },
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

        // projected_splats is [num_visible, proj_size], so shape[0] gives num_visible.
        let num_visible_val = projected_splats.shape()[0] as u32;

        let client = v_output.client.clone();
        let num_visible = (num_visible_val as usize).max(1);

        let v_combined_out = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_visible, 10]),
            DType::F32,
        );

        let input_tensors = [
            v_output,
            out_img,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
        ];

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

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [v_combined] = outputs;

        RasterizeGrads { v_combined }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        sh_degree: u32,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            render_mode: SplatRenderMode,
            sh_degree: u32,
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
                    self.sh_degree,
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
        let coeffs = sh_coeffs_for_degree(sh_degree) as usize;

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

        let input_tensors = [transforms, raw_opac, global_from_compact_gid, v_combined];

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

        let [v_transforms, v_coeffs, v_raw_opac, v_refine_weight] = outputs;

        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}
