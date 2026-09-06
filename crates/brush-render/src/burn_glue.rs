#![allow(clippy::match_wildcard_for_single_variants)]

use brush_cube::{MainBackend, MainBackendBase};
use burn::backend::{
    Autodiff, BackendTensor, DispatchAutodiffContext, DispatchTensor, DispatchTensorKind,
    GradientCheckpointingStrategy, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::{
    ExecutionError, Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use glam::Vec3;

use crate::{
    RenderAuxInner, SplatOps, backend_kind, camera::Camera, gaussian_splats::SplatRenderMode,
    render_aux::RenderOutput,
};
use burn_cubecl::CubeBackend;

/// Inner Wgpu autodiff backend (same as `Autodiff<burn::backend::Wgpu>`).
/// Used as the primitive backend for autodiff `Tensor<D>` operations.
pub type AutodiffMain = Autodiff<MainBackend>;

// ---------------------------------------------------------------------------
// `Tensor<D>` ↔ backend-level primitive bridges.
//
// `Tensor<D>` is pinned to burn's `Dispatch` backend; brush only ever runs on
// a wgpu device, so every helper here assumes a `DispatchTensorKind::Cube`
// (optionally wrapped in `Autodiff`) and panics otherwise. The forward render
// now goes through the `#[backend_extension]`-generated `Dispatch` impl
// instead; these stay for the hand-rolled backward path (brush-render-bwd)
// and the LPIPS custom ops (brush-loss).
// ---------------------------------------------------------------------------

/// Extract the inner fusion-Wgpu float tensor from a non-autodiff
/// `Tensor<D>`.
pub fn unwrap_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.float(),
        other => panic!(
            "expected Wgpu tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner fusion-Wgpu int tensor from a non-autodiff
/// `Tensor<D, Int>`.
pub fn unwrap_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_wgpu_float`]: wraps a fusion-Wgpu float tensor as a
/// user-facing `Tensor<D>`.
pub fn wrap_wgpu_float<const D: usize>(t: FloatTensor<MainBackend>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: backend_kind!(BackendTensor::Float(t)),
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

/// Extract the inner `AutodiffTensor<MainBackend>` from a `Tensor<D>` on an
/// autodiff-enabled Wgpu device. Panics on any other shape.
pub fn unwrap_ad_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<AutodiffMain> {
    let prim: DispatchTensor = t.into_dispatch();
    match prim.kind {
        DispatchTensorKind::Autodiff(inner) => match *inner {
            backend_kind!(BackendTensor::Autodiff(t)) => t,
            other => panic!(
                "autodiff inner kind is not Wgpu: {:?}",
                std::mem::discriminant(&other)
            ),
        },
        other => panic!(
            "expected autodiff-enabled tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner Wgpu `IntTensor` regardless of whether the tensor is
/// wrapped in an autodiff device — ints are never autodiff-tracked.
pub fn unwrap_ad_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    match kind {
        backend_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor; got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Inverse of [`unwrap_ad_wgpu_float`]: wraps an autodiff tensor as a
/// user-facing `Tensor<D>` on the autodiff device.
pub fn wrap_ad_wgpu_float<const D: usize>(t: FloatTensor<AutodiffMain>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: DispatchTensorKind::Autodiff(Box::new(backend_kind!(BackendTensor::Autodiff(t)))),
        autodiff: DispatchAutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
    })
}

fn is_autodiff<const D: usize>(t: &Tensor<D>) -> bool {
    matches!(
        t.clone().into_dispatch().kind,
        DispatchTensorKind::Autodiff(_)
    )
}

/// Put `t` on the same autodiff/inner backend variant as `reference`. Brush
/// keeps some frozen tensors (e.g. the 3D-filter floor) on the inner backend
/// but folds them against params that may be lifted to autodiff; this aligns
/// both operands so dispatch ops don't trip a cross-backend assertion.
pub(crate) fn match_backend<const D: usize, const DR: usize>(
    t: Tensor<D>,
    reference: &Tensor<DR>,
) -> Tensor<D> {
    if is_autodiff(reference) {
        t.autodiff()
    } else {
        t.without_autodiff()
    }
}

/// Resolve pending fusion operations and return the underlying tensor.
pub fn resolve_to_cube_float<const D: usize>(tensor: Tensor<D>) -> CubeTensor {
    let fusion = unwrap_wgpu_float(tensor);
    let client = fusion.client.clone();
    client.resolve_tensor_float::<MainBackendBase>(fusion)
}

impl SplatOps for Fusion<CubeBackend> {
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        refine_weight: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> RenderOutput<Self> {
        let client = transforms.client.clone();

        // Resolve fusion inputs to MainBackendBase tensors. This
        // drains any pending fusion operations into a concrete buffer.
        let base_transforms = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(transforms);
        let base_sh_coeffs = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(sh_coeffs);
        let base_raw_opac = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(raw_opacities);
        let base_refine_weight = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(refine_weight);

        let out = <CubeBackend as SplatOps>::render(
            camera,
            img_size,
            base_transforms,
            base_sh_coeffs,
            base_raw_opac,
            base_refine_weight,
            render_mode,
            background,
            pass,
        )
        .await;

        // Bind precomputed outputs back into the fusion stream.
        #[derive(Debug)]
        struct BindOp {
            desc: CustomOpIr,
            out_img: FloatTensor<CubeBackend>,
            visible: FloatTensor<CubeBackend>,
            max_radius: FloatTensor<CubeBackend>,
            projected_splats: FloatTensor<CubeBackend>,
            tile_offsets: IntTensor<CubeBackend>,
            compact_gid_from_isect: IntTensor<CubeBackend>,
            global_from_compact_gid: IntTensor<CubeBackend>,
        }

        impl Operation<FusionCubeRuntime> for BindOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime>>,
            ) -> Result<(), ExecutionError> {
                let (_, outputs) = self.desc.as_fixed::<0, 7>();
                let [
                    out_img,
                    visible,
                    max_radius,
                    projected_splats,
                    tile_offsets,
                    compact_gid_from_isect,
                    global_from_compact_gid,
                ] = outputs;

                h.register_float_tensor::<CubeBackend>(&out_img.id, self.out_img.clone());
                h.register_float_tensor::<CubeBackend>(&visible.id, self.visible.clone());
                h.register_float_tensor::<CubeBackend>(&max_radius.id, self.max_radius.clone());
                h.register_float_tensor::<CubeBackend>(
                    &projected_splats.id,
                    self.projected_splats.clone(),
                );
                h.register_int_tensor::<CubeBackend>(&tile_offsets.id, self.tile_offsets.clone());
                h.register_int_tensor::<CubeBackend>(
                    &compact_gid_from_isect.id,
                    self.compact_gid_from_isect.clone(),
                );
                h.register_int_tensor::<CubeBackend>(
                    &global_from_compact_gid.id,
                    self.global_from_compact_gid.clone(),
                );
                Ok(())
            }
        }

        // Every output is a fresh handle the bind op fills in; only shape and
        // dtype differ.
        let new_out = |shape, dtype| TensorIr::uninit(client.create_empty_handle(), shape, dtype);
        let out_img_ir = new_out(out.out_img.shape(), DType::F32);
        let visible_ir = new_out(out.aux.visible.shape(), DType::F32);
        let max_radius_ir = new_out(out.aux.max_radius.shape(), DType::F32);
        let projected_splats_ir = new_out(out.projected_splats.shape(), DType::F32);
        let tile_offsets_ir = new_out(out.aux.tile_offsets.shape(), DType::U32);
        let compact_gid_from_isect_ir = new_out(out.compact_gid_from_isect.shape(), DType::U32);
        let global_from_compact_gid_ir = new_out(out.global_from_compact_gid.shape(), DType::U32);

        let stream = StreamId::current();
        let desc = CustomOpIr::new(
            "render_bind",
            &[],
            &[
                out_img_ir,
                visible_ir,
                max_radius_ir,
                projected_splats_ir,
                tile_offsets_ir,
                compact_gid_from_isect_ir,
                global_from_compact_gid_ir,
            ],
        );
        let op = BindOp {
            desc: desc.clone(),
            out_img: out.out_img,
            visible: out.aux.visible,
            max_radius: out.aux.max_radius,
            projected_splats: out.projected_splats,
            tile_offsets: out.aux.tile_offsets,
            compact_gid_from_isect: out.compact_gid_from_isect,
            global_from_compact_gid: out.global_from_compact_gid,
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [
            out_img,
            visible,
            max_radius,
            projected_splats,
            tile_offsets,
            compact_gid_from_isect,
            global_from_compact_gid,
        ] = outputs;

        RenderOutput {
            out_img,
            aux: RenderAuxInner {
                num_visible: out.aux.num_visible,
                num_intersections: out.aux.num_intersections,
                visible,
                max_radius,
                tile_offsets,
                img_size: out.aux.img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            project_uniforms: out.project_uniforms,
            global_from_compact_gid,
        }
    }
}
