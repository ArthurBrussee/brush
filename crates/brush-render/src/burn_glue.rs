#![allow(clippy::match_wildcard_for_single_variants)]

use brush_cube::FusionCubeRuntime;
use brush_cube::{MainBackend, MainBackendBase};
use burn::backend::{
    Autodiff, AutodiffBackend, BackendTensor, DispatchTensor, DispatchTensorKind,
    GradientCheckpointingStrategy, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{DType, Int, Tensor};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::CubeTensor;
use glam::Vec3;

use crate::{
    RenderAuxInner, SplatOps, backend_kind, camera::Camera, gaussian_splats::SplatRenderMode,
    render_aux::RenderOutput,
};
use brush_cube::{CubeBackend, CubeRuntime};

/// Inner Wgpu autodiff backend (same as `Autodiff<burn::backend::Wgpu>`).
/// Used as the primitive backend for autodiff `Tensor<D>` operations.
pub type AutodiffMain = Autodiff<MainBackend>;

// ---------------------------------------------------------------------------
// `Tensor<D>` ↔ backend-level primitive bridges.
//
// `Tensor<D>` is pinned to burn's `Dispatch` backend; brush only ever runs on
// a wgpu device, so every helper here assumes a `DispatchTensorKind::Wgpu`
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
        checkpointing: None,
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
        checkpointing: Some(GradientCheckpointingStrategy::Disabled),
    })
}

/// Strip the autodiff wrapping from a `Tensor<D>` and clear the residual
/// `checkpointing` field.
///
/// Operates directly on the `DispatchTensor` kind so it works both for an
/// autodiff input (unwrap one level) and an already-inner input (passthrough),
/// always landing with `checkpointing: None`. The high-level `.inner()` can't
/// stand in here: it panics on a non-autodiff input, and (via the Bridge path)
/// doesn't reliably normalise `checkpointing`, which downstream ops read as a
/// "came from autodiff" signal and use to re-lift — tripping cross-backend
/// asserts when mixed with a genuinely-inner tensor.
pub fn detach_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    Tensor::from_dispatch(DispatchTensor {
        kind,
        checkpointing: None,
    })
}

/// Lift a non-autodiff `Tensor<D>` into the autodiff graph as a constant.
/// A no-op if `t` is already autodiff.
///
/// Lifts at the concrete-Wgpu autodiff level and re-wraps with an explicit
/// `checkpointing`. The high-level `Tensor::from_inner` goes through the
/// Bridge/Dispatch path, which doesn't set `checkpointing` the way the mixed
/// inner/autodiff folds (e.g. `fold_min_scale`) need — a lifted constant then
/// degrades to the inner backend on the next op and trips a cross-backend
/// assert. Keep the hand-rolled lift.
pub(crate) fn lift_to_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    /// Lift within one backend, keeping the variant it came in on.
    macro_rules! lift_in {
        ($variant:ident, $backend:ty, $inner:expr) => {
            Tensor::from_dispatch(DispatchTensor {
                kind: DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::$variant(
                    BackendTensor::Autodiff(<Autodiff<$backend> as AutodiffBackend>::from_inner(
                        $inner,
                    )),
                ))),
                checkpointing: Some(GradientCheckpointingStrategy::Disabled),
            })
        };
    }

    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        DispatchTensorKind::Wgpu(BackendTensor::Float(inner)) => {
            lift_in!(Wgpu, burn::backend::Wgpu, inner)
        }
        DispatchTensorKind::Autodiff(_) => Tensor::from_dispatch(dispatch),
        _ => panic!("unsupported backend for autodiff lift"),
    }
}

/// Fully strip autodiff from a dispatch tensor: both the outer
/// `DispatchTensorKind::Autodiff` wrapper and the inner
/// `BackendTensor::Autodiff`, landing on the plain inner-backend float.
///
/// `detach_autodiff` only removes the outer level, which leaves a tensor that
/// still reports as autodiff to ops that inspect the `BackendTensor`.
pub fn strip_autodiff_float<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    macro_rules! strip_in {
        ($variant:ident, $inner:expr) => {
            DispatchTensorKind::$variant(BackendTensor::Float($inner.primitive))
        };
    }

    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    let kind = match kind {
        DispatchTensorKind::Wgpu(BackendTensor::Autodiff(ad)) => strip_in!(Wgpu, ad),
        other => other,
    };
    Tensor::from_dispatch(DispatchTensor {
        kind,
        checkpointing: None,
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
        lift_to_autodiff(t)
    } else {
        detach_autodiff(t)
    }
}

/// Like [`detach_autodiff`] for `Tensor<D, Int>`.
pub fn detach_autodiff_int<const D: usize>(t: Tensor<D, Int>) -> Tensor<D, Int> {
    let dispatch: DispatchTensor = t.into_dispatch();
    let kind = match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => *inner,
        other => other,
    };
    Tensor::from_dispatch(DispatchTensor {
        kind,
        checkpointing: None,
    })
}

/// Resolve a `Tensor<D>` down to the underlying `CubeTensor`, draining any
/// pending fusion ops. Used for direct GPU resource access, e.g. binding the
/// buffer into a wgpu pipeline, so it stays tied to the main backend rather
/// than being generic over the runtime.
pub fn resolve_to_cube_float<const D: usize>(
    tensor: Tensor<D>,
) -> CubeTensor<brush_cube::MainRuntime> {
    let fusion = unwrap_wgpu_float(tensor);
    let client = fusion.client.clone();
    client.resolve_tensor_float::<MainBackendBase>(fusion)
}

impl<R: CubeRuntime> SplatOps for Fusion<CubeBackend<R>> {
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
            .resolve_tensor_float::<CubeBackend<R>>(transforms);
        let base_sh_coeffs = client
            .clone()
            .resolve_tensor_float::<CubeBackend<R>>(sh_coeffs);
        let base_raw_opac = client
            .clone()
            .resolve_tensor_float::<CubeBackend<R>>(raw_opacities);
        let base_refine_weight = client
            .clone()
            .resolve_tensor_float::<CubeBackend<R>>(refine_weight);

        // Run the full pipeline on the concrete cube backend.
        let out = <CubeBackend<R> as SplatOps>::render(
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
        struct BindOp<R: CubeRuntime> {
            desc: CustomOpIr,
            out_img: FloatTensor<CubeBackend<R>>,
            visible: FloatTensor<CubeBackend<R>>,
            max_radius: FloatTensor<CubeBackend<R>>,
            projected_splats: FloatTensor<CubeBackend<R>>,
            tile_offsets: IntTensor<CubeBackend<R>>,
            compact_gid_from_isect: IntTensor<CubeBackend<R>>,
            global_from_compact_gid: IntTensor<CubeBackend<R>>,
        }

        impl<R: CubeRuntime> Operation<FusionCubeRuntime<R>> for BindOp<R> {
            fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<R>>>) {
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

                h.register_float_tensor::<CubeBackend<R>>(&out_img.id, self.out_img.clone());
                h.register_float_tensor::<CubeBackend<R>>(&visible.id, self.visible.clone());
                h.register_float_tensor::<CubeBackend<R>>(&max_radius.id, self.max_radius.clone());
                h.register_float_tensor::<CubeBackend<R>>(
                    &projected_splats.id,
                    self.projected_splats.clone(),
                );
                h.register_int_tensor::<CubeBackend<R>>(
                    &tile_offsets.id,
                    self.tile_offsets.clone(),
                );
                h.register_int_tensor::<CubeBackend<R>>(
                    &compact_gid_from_isect.id,
                    self.compact_gid_from_isect.clone(),
                );
                h.register_int_tensor::<CubeBackend<R>>(
                    &global_from_compact_gid.id,
                    self.global_from_compact_gid.clone(),
                );
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
        let op = BindOp::<R> {
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
