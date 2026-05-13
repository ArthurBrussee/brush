use burn::tensor::{
    DType, TensorMetadata,
    ops::{FloatTensor, IntTensor},
};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;
use glam::Vec3;

use crate::{
    MainBackendBase, RenderAux, SplatOps, camera::Camera, gaussian_splats::SplatRenderMode,
    render_aux::RenderOutput,
};

/// **Workaround for an upstream `burn-fusion` cross-thread race.**
///
/// `burn-fusion`'s `Client` dispatches custom ops to a small pool of
/// per-stream worker threads (the `DSU-N` threads in stack traces).
/// Each worker owns its own `HandleContainer` and processes ops
/// serially. When a `FusionTensor` is shared across threads (e.g. a
/// `Splats` clone published via a `tokio::sync::watch::Sender<Splats>`
/// and rendered on a different thread than the trainer),
/// `FusionTensor::Drop` from the *reading* thread enqueues a `Drop`
/// op onto the *origin* stream's worker. The cross-stream
/// coordination has a window where the `Drop` can be processed before
/// a pending op on another stream that still needs the handle —
/// hence `Should have handle for tensor TensorId { value: N }` panics
/// from `burn-ir/src/handle.rs::HandleContainer::get_handle`, and
/// the silent variant: a freed page reread as stale memory, surfacing
/// as NaNs in `projected_splats`.
///
/// brush-side workaround: callers (trainer iteration, viewer render,
/// export, etc.) hold this lock for their *entire* iteration — from
/// before the splats clone, through the render / step, until after
/// the iteration's locals (including the snapshot) have dropped. That
/// way no two threads are ever simultaneously touching fusion state,
/// and `FusionTensor::Drop` doesn't fire concurrently with another
/// thread's pending ops on the same tensor.
///
/// `parking_lot::Mutex` (with the `send_guard` feature) so the guard
/// is `Send` and can ride across `.await` on a `Send` future, since
/// the brush trainer stream's body holds the lock across its
/// awaits.
///
/// **Remove when upstream burn fixes cross-thread handle drops.**
pub static FUSION_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

impl SplatOps<Self> for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        bwd_info: bool,
    ) -> RenderOutput<Self> {
        // Callers (trainer step, viewer render, etc.) must hold
        // FUSION_LOCK across their whole iteration — see FUSION_LOCK
        // doc. This impl assumes that and just runs.
        let client = transforms.client.clone();

        // Resolve fusion inputs to MainBackendBase tensors. This
        // drains any pending fusion operations into a concrete buffer.
        let base_transforms = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(transforms);
        let base_sh_coeffs = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(sh_coeffs);
        let base_raw_opac = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(raw_opacities);

        // Run the full pipeline on MainBackendBase (with its own
        // internal readback for num_visible / num_intersections).
        let out = MainBackendBase::render(
            camera,
            img_size,
            base_transforms,
            base_sh_coeffs,
            base_raw_opac,
            render_mode,
            background,
            bwd_info,
        )
        .await;

        // Bind precomputed outputs back into the fusion stream.
        #[derive(Debug)]
        struct BindOp {
            desc: CustomOpIr,
            out_img: FloatTensor<MainBackendBase>,
            visible: FloatTensor<MainBackendBase>,
            max_radius: FloatTensor<MainBackendBase>,
            projected_splats: FloatTensor<MainBackendBase>,
            tile_offsets: IntTensor<MainBackendBase>,
            compact_gid_from_isect: IntTensor<MainBackendBase>,
            global_from_compact_gid: IntTensor<MainBackendBase>,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
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

                h.register_float_tensor::<MainBackendBase>(&out_img.id, self.out_img.clone());
                h.register_float_tensor::<MainBackendBase>(&visible.id, self.visible.clone());
                h.register_float_tensor::<MainBackendBase>(&max_radius.id, self.max_radius.clone());
                h.register_float_tensor::<MainBackendBase>(
                    &projected_splats.id,
                    self.projected_splats.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(
                    &tile_offsets.id,
                    self.tile_offsets.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(
                    &compact_gid_from_isect.id,
                    self.compact_gid_from_isect.clone(),
                );
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
        let visible_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.aux.visible.shape(),
            DType::F32,
        );
        let max_radius_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.aux.max_radius.shape(),
            DType::F32,
        );
        let projected_splats_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.projected_splats.shape(),
            DType::F32,
        );
        let tile_offsets_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.aux.tile_offsets.shape(),
            DType::U32,
        );
        let compact_gid_from_isect_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.compact_gid_from_isect.shape(),
            DType::U32,
        );
        let global_from_compact_gid_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.global_from_compact_gid.shape(),
            DType::U32,
        );

        let stream = OperationStreams::default();
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
            aux: RenderAux {
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
