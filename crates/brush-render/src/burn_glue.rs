#![allow(clippy::match_wildcard_for_single_variants)]

use crate::fusion::bind;
use brush_cube::{MainBackend, MainBackendBase};
use burn::backend::{
    Autodiff, BackendTensor, DispatchAutodiffContext, DispatchTensor, DispatchTensorKind,
    GradientCheckpointingStrategy,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{Int, Tensor};
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::Fusion;
use glam::Vec3;

use crate::{
    RenderAuxInner, SplatOps, backend_kind, camera::Camera, gaussian_splats::SplatRenderMode,
    render_aux::RenderOutput,
};
use burn_cubecl::CubeBackend;

/// Inner Wgpu autodiff backend (same as `Autodiff<burn::backend::Wgpu>`).
/// Used as the primitive backend for autodiff `Tensor<D>` operations.
pub type AutodiffMain = Autodiff<MainBackend>;

// `Tensor<D>` is pinned to burn's `Dispatch` backend and brush only runs on
// wgpu, so these bridges assume a `DispatchTensorKind::Cube` (optionally
// wrapped in `Autodiff`) and panic otherwise. The forward render routes
// through the generated `Dispatch` impl; these serve the hand-rolled backward.

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

/// Put `t` on the same autodiff/inner backend variant as `reference`. Frozen
/// tensors like the 3D-filter floor live on the inner backend but get folded
/// against params that may be lifted, and mixing the two trips an assertion.
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
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
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
        let base_min_scale = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(min_scale);
        let base_refine_weight = client
            .clone()
            .resolve_tensor_float::<CubeBackend>(refine_weight);

        let out = <CubeBackend as SplatOps>::render(
            camera,
            img_size,
            base_transforms,
            base_sh_coeffs,
            base_raw_opac,
            base_min_scale,
            has_min_scale,
            base_refine_weight,
            render_mode,
            background,
            pass,
        )
        .await;

        let RenderOutput {
            out_img,
            aux,
            projected_splats,
            compact_gid_from_isect,
            project_uniforms,
            global_from_compact_gid,
            compact_from_global,
        } = out;
        let RenderAuxInner {
            num_visible,
            num_intersections,
            visible,
            max_radius,
            opacities,
            tile_offsets,
            img_size,
        } = aux;

        let bind = |t| bind(&client, t);

        RenderOutput {
            out_img: bind(out_img),
            aux: RenderAuxInner {
                num_visible,
                num_intersections,
                visible: bind(visible),
                max_radius: bind(max_radius),
                opacities: bind(opacities),
                tile_offsets: bind(tile_offsets),
                img_size,
            },
            projected_splats: bind(projected_splats),
            compact_gid_from_isect: bind(compact_gid_from_isect),
            project_uniforms,
            global_from_compact_gid: bind(global_from_compact_gid),
            compact_from_global: bind(compact_from_global),
        }
    }
}
