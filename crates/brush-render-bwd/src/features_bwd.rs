//! Differentiable feature rendering (`DiG`).
//!
//! Wraps [`brush_render::render_features`] with a hand-rolled autodiff
//! `Backward` whose only tracked input is the `[N, feat_dim]` feature
//! tensor — geometry is detached, matching the reference `DiG` feature
//! pass. Same fusion custom-op plumbing as the RGB path in
//! [`crate::burn_glue`], but with a single differentiable input.

use brush_cube::{MainBackend, MainBackendBase, calc_cube_count_1d};
use brush_render::burn_glue::{AutodiffMain, unwrap_ad_wgpu_float, wrap_ad_wgpu_float};
use brush_render::{
    camera::Camera,
    gaussian_splats::SplatRenderMode,
    render_features::{FeatureRenderOutput, render_features_base},
    shaders::helpers::TILE_WIDTH,
};
use burn::backend::ops::FloatTensorOps;
use burn::{
    backend::{
        TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
        wgpu::WgpuRuntime,
    },
    tensor::{DType, FloatDType, Shape, Tensor},
};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::cubecl::features::AtomicUsage;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, Type};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

use crate::kernels;

/// Launch the feature backward kernel on the base backend.
#[allow(clippy::too_many_arguments)]
fn rasterize_features_bwd_base(
    projected_splats: FloatTensor<MainBackendBase>,
    compact_gid_from_isect: IntTensor<MainBackendBase>,
    tile_offsets: IntTensor<MainBackendBase>,
    global_from_compact_gid: IntTensor<MainBackendBase>,
    v_output: FloatTensor<MainBackendBase>,
    img_size: glam::UVec2,
    num_points: usize,
    feat_dim: usize,
) -> FloatTensor<MainBackendBase> {
    let _span = tracing::trace_span!("rasterize_features_bwd").entered();

    let v_output = into_contiguous(v_output);
    let device = v_output.device.clone();
    let client = v_output.client.clone();

    let v_features =
        MainBackendBase::float_zeros([num_points, feat_dim].into(), &device, FloatDType::F32);

    let tile_bounds = glam::uvec2(
        img_size.x.div_ceil(TILE_WIDTH),
        img_size.y.div_ceil(TILE_WIDTH),
    );
    let num_tiles = tile_bounds.x * tile_bounds.y;

    let hard_floats = client
        .properties()
        .atomic_type_usage(Type::atomic(Type::scalar(ElemType::Float(FloatKind::F32))))
        .contains(AtomicUsage::Add);

    let uniforms = brush_render::kernels::types::RasterizeUniformsLaunch::new(
        tile_bounds.x,
        img_size.x,
        img_size.y,
        0.0,
        0.0,
        0.0,
    );
    let cube_count = calc_cube_count_1d(
        num_tiles * (TILE_WIDTH * TILE_WIDTH),
        TILE_WIDTH * TILE_WIDTH,
    );
    let cube_dim = CubeDim::new_1d(TILE_WIDTH * TILE_WIDTH);

    tracing::trace_span!("RasterizeFeaturesBackwards").in_scope(|| {
        use kernels::rasterize_backwards::{CasAtomicAdd, HfAtomicAdd};
        use kernels::rasterize_features_backwards::rasterize_features_backwards_kernel;
        if hard_floats {
            rasterize_features_backwards_kernel::launch::<HfAtomicAdd, WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                compact_gid_from_isect.into_tensor_arg(),
                tile_offsets.into_tensor_arg(),
                projected_splats.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_output.into_tensor_arg(),
                v_features.clone().into_tensor_arg(),
                uniforms,
                feat_dim,
            );
        } else {
            rasterize_features_backwards_kernel::launch::<CasAtomicAdd, WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                compact_gid_from_isect.into_tensor_arg(),
                tile_offsets.into_tensor_arg(),
                projected_splats.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_output.into_tensor_arg(),
                v_features.clone().into_tensor_arg(),
                uniforms,
                feat_dim,
            );
        }
    });

    v_features
}

/// Run the feature forward on the fusion backend: resolve inputs, run the
/// base pipeline, and bind the outputs back into the fusion stream.
async fn render_features_fusion(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<MainBackend>,
    raw_opacities: FloatTensor<MainBackend>,
    features: FloatTensor<MainBackend>,
    render_mode: SplatRenderMode,
) -> FeatureRenderOutput<MainBackend> {
    let client = transforms.client.clone();

    let base_transforms = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(transforms);
    let base_raw_opac = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(raw_opacities);
    let base_features = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(features);

    let out = render_features_base(
        camera,
        img_size,
        base_transforms,
        base_raw_opac,
        base_features,
        render_mode,
    )
    .await;

    #[derive(Debug)]
    struct BindOp {
        desc: CustomOpIr,
        out_img: FloatTensor<MainBackendBase>,
        projected_splats: FloatTensor<MainBackendBase>,
        compact_gid_from_isect: IntTensor<MainBackendBase>,
        tile_offsets: IntTensor<MainBackendBase>,
        global_from_compact_gid: IntTensor<MainBackendBase>,
    }

    impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
        fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
            let (_, outputs) = self.desc.as_fixed::<0, 5>();
            let [
                out_img,
                projected_splats,
                compact_gid_from_isect,
                tile_offsets,
                global_from_compact_gid,
            ] = outputs;

            h.register_float_tensor::<MainBackendBase>(&out_img.id, self.out_img.clone());
            h.register_float_tensor::<MainBackendBase>(
                &projected_splats.id,
                self.projected_splats.clone(),
            );
            h.register_int_tensor::<MainBackendBase>(
                &compact_gid_from_isect.id,
                self.compact_gid_from_isect.clone(),
            );
            h.register_int_tensor::<MainBackendBase>(&tile_offsets.id, self.tile_offsets.clone());
            h.register_int_tensor::<MainBackendBase>(
                &global_from_compact_gid.id,
                self.global_from_compact_gid.clone(),
            );
        }
    }

    let out_img_ir = TensorIr::uninit(
        client.create_empty_handle(),
        out.out_img.shape(),
        DType::F32,
    );
    let projected_splats_ir = TensorIr::uninit(
        client.create_empty_handle(),
        out.projected_splats.shape(),
        DType::F32,
    );
    let compact_gid_from_isect_ir = TensorIr::uninit(
        client.create_empty_handle(),
        out.compact_gid_from_isect.shape(),
        DType::U32,
    );
    let tile_offsets_ir = TensorIr::uninit(
        client.create_empty_handle(),
        out.tile_offsets.shape(),
        DType::U32,
    );
    let global_from_compact_gid_ir = TensorIr::uninit(
        client.create_empty_handle(),
        out.global_from_compact_gid.shape(),
        DType::U32,
    );

    let stream = StreamId::current();
    let desc = CustomOpIr::new(
        "render_features_bind",
        &[],
        &[
            out_img_ir,
            projected_splats_ir,
            compact_gid_from_isect_ir,
            tile_offsets_ir,
            global_from_compact_gid_ir,
        ],
    );
    let op = BindOp {
        desc: desc.clone(),
        out_img: out.out_img,
        projected_splats: out.projected_splats,
        compact_gid_from_isect: out.compact_gid_from_isect,
        tile_offsets: out.tile_offsets,
        global_from_compact_gid: out.global_from_compact_gid,
    };

    let outputs = client
        .register(stream, OperationIr::Custom(desc), op)
        .outputs();

    let [
        out_img,
        projected_splats,
        compact_gid_from_isect,
        tile_offsets,
        global_from_compact_gid,
    ] = outputs;

    FeatureRenderOutput {
        out_img,
        projected_splats,
        compact_gid_from_isect,
        tile_offsets,
        global_from_compact_gid,
        num_visible: out.num_visible,
    }
}

/// Fusion wrapper around [`rasterize_features_bwd_base`].
#[allow(clippy::too_many_arguments)]
fn rasterize_features_bwd_fusion(
    projected_splats: FloatTensor<Fusion<MainBackendBase>>,
    compact_gid_from_isect: IntTensor<Fusion<MainBackendBase>>,
    tile_offsets: IntTensor<Fusion<MainBackendBase>>,
    global_from_compact_gid: IntTensor<Fusion<MainBackendBase>>,
    v_output: FloatTensor<Fusion<MainBackendBase>>,
    img_size: glam::UVec2,
    num_points: usize,
    feat_dim: usize,
) -> FloatTensor<Fusion<MainBackendBase>> {
    #[derive(Debug)]
    struct CustomOp {
        desc: CustomOpIr,
        img_size: glam::UVec2,
        num_points: usize,
        feat_dim: usize,
    }

    impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
        fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
            let (inputs, outputs) = self.desc.as_fixed();

            let [
                v_output,
                projected_splats,
                compact_gid_from_isect,
                tile_offsets,
                global_from_compact_gid,
            ] = inputs;
            let [v_features] = outputs;

            let grads = rasterize_features_bwd_base(
                h.get_float_tensor::<MainBackendBase>(projected_splats),
                h.get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                h.get_int_tensor::<MainBackendBase>(tile_offsets),
                h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                h.get_float_tensor::<MainBackendBase>(v_output),
                self.img_size,
                self.num_points,
                self.feat_dim,
            );

            h.register_float_tensor::<MainBackendBase>(&v_features.id, grads);
        }
    }

    let client = v_output.client.clone();

    let input_tensors = [
        v_output,
        projected_splats,
        compact_gid_from_isect,
        tile_offsets,
        global_from_compact_gid,
    ];

    let outputs = {
        let v_features_out = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, feat_dim]),
            DType::F32,
        );
        let stream = StreamId::current();
        let desc = CustomOpIr::new(
            "rasterize_features_bwd",
            &input_tensors.map(|t| t.into_ir()),
            &[v_features_out],
        );
        let op = CustomOp {
            desc: desc.clone(),
            img_size,
            num_points,
            feat_dim,
        };
        client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs()
    };

    let [v_features] = outputs;
    v_features
}

/// State saved by the forward for the feature backward.
#[derive(Debug, Clone)]
struct FeatureBackwardState {
    projected_splats: FloatTensor<MainBackend>,
    compact_gid_from_isect: IntTensor<MainBackend>,
    tile_offsets: IntTensor<MainBackend>,
    global_from_compact_gid: IntTensor<MainBackend>,
    img_size: glam::UVec2,
    num_points: usize,
    feat_dim: usize,
}

#[derive(Debug)]
struct FeaturesBackward;

impl Backward<MainBackend, 1> for FeaturesBackward {
    type State = FeatureBackwardState;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_features backwards").entered();

        let state = ops.state;
        let v_output = grads.consume::<MainBackend>(&ops.node);
        let [features_parent] = ops.parents;

        if let Some(node) = features_parent {
            let v_features = rasterize_features_bwd_fusion(
                state.projected_splats,
                state.compact_gid_from_isect,
                state.tile_offsets,
                state.global_from_compact_gid,
                v_output,
                state.img_size,
                state.num_points,
                state.feat_dim,
            );
            grads.register::<MainBackend>(node.id, v_features);
        }
    }
}

/// Differentiably rasterize `[N, feat_dim]` per-splat features to a
/// `[h, w, feat_dim + 1]` image (features + alpha in the last channel).
///
/// Only `features` is on the autodiff graph; `transforms` and
/// `raw_opacities` are detached internally (pass the same folded values
/// the RGB pass renders with). Requires an autodiff-enabled device for
/// `features`.
/// Resolve a `Tensor<D>` — autodiff-wrapped or already inner — down to the
/// inner fusion-Wgpu float primitive, discarding any autodiff tracking.
fn to_inner_float<const D: usize>(t: Tensor<D>) -> FloatTensor<MainBackend> {
    use burn::backend::DispatchTensorKind;
    let is_ad = matches!(
        t.clone().into_dispatch().kind,
        DispatchTensorKind::Autodiff(_)
    );
    if is_ad {
        unwrap_ad_wgpu_float(t).primitive
    } else {
        brush_render::burn_glue::unwrap_wgpu_float(t)
    }
}

pub async fn render_splat_features(
    transforms: Tensor<2>,
    raw_opacities: Tensor<1>,
    features: Tensor<2>,
    camera: &Camera,
    img_size: glam::UVec2,
    render_mode: SplatRenderMode,
) -> Tensor<3> {
    let [num_points, feat_dim] = features.dims();

    let features_ad = unwrap_ad_wgpu_float(features);
    let prep_nodes = FeaturesBackward
        .prepare::<NoCheckpointing>([features_ad.node.clone()])
        .compute_bound()
        .stateful();

    let features_inner: FloatTensor<MainBackend> = features_ad.primitive.clone();
    let transforms_inner = to_inner_float(transforms);
    let raw_opac_inner = to_inner_float(raw_opacities);

    let output = render_features_fusion(
        camera,
        img_size,
        transforms_inner,
        raw_opac_inner,
        features_inner,
        render_mode,
    )
    .await;

    let img_ad: FloatTensor<AutodiffMain> = match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = FeatureBackwardState {
                projected_splats: output.projected_splats,
                compact_gid_from_isect: output.compact_gid_from_isect,
                tile_offsets: output.tile_offsets,
                global_from_compact_gid: output.global_from_compact_gid,
                img_size,
                num_points,
                feat_dim,
            };
            prep.finish(state, output.out_img)
        }
        OpsKind::UnTracked(prep) => prep.finish(output.out_img),
    };

    wrap_ad_wgpu_float(img_ad)
}
