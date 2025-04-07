use brush_kernel::create_dispatch_buffer;
use brush_render::{MainBackend, MainBackendBase};
use burn::{prelude::Backend, tensor::ops::IntTensor};
use burn_cubecl::cubecl::{self, CubeDim, cube, prelude::*};
use burn_fusion::client::FusionClient;
use glam::UVec2;
use tracing::trace_span;

#[cube(launch)]
#[allow(clippy::useless_conversion)]
fn stats_gather_kernel(
    gs_ids: &Tensor<u32>,
    num_visible: &Tensor<u32>,
    refine_weight: &Tensor<Line<f32>>,
    accum_refine_weight: &mut Tensor<f32>,
    #[comptime] w: u32,
    #[comptime] h: u32,
) {
    let compact_gid = ABSOLUTE_POS_X;
    let num_vis = num_visible[0];

    if compact_gid >= num_vis {
        terminate!();
    }

    let global_gid = gs_ids[compact_gid];

    let mut line: Line<f32> = Line::empty(2u32);
    // Nb: Clippy reports a warning here about a useless conversion but it's wrong.
    line[0] = comptime!(w as f32 / 2.0);
    line[1] = comptime!(h as f32 / 2.0);

    let refine_grads = refine_weight[compact_gid] * line;
    let refine_norm =
        f32::sqrt(refine_grads[0] * refine_grads[0] + refine_grads[1] * refine_grads[1]);
    accum_refine_weight[global_gid] = f32::max(accum_refine_weight[global_gid], refine_norm);
}

pub(crate) struct RefineRecord<B: Backend> {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    pub refine_weight_norm: burn::tensor::Tensor<B, 1>,
}

impl<B: Backend> RefineRecord<B> {
    pub(crate) fn new(num_points: u32, device: &B::Device) -> Self {
        Self {
            refine_weight_norm: burn::tensor::Tensor::<B, 1>::zeros([num_points as usize], device),
        }
    }
}

impl RefineRecord<MainBackend> {
    pub(crate) fn gather_stats(
        &self,
        refine_weight: burn::tensor::Tensor<MainBackend, 1>,
        resolution: UVec2,
        global_from_compact_gid: IntTensor<MainBackend>,
        num_visible: IntTensor<MainBackend>,
    ) {
        let _span = trace_span!("Gather stats", sync_burn = true);

        let [w, h] = [resolution.x, resolution.y];
        let client = &self
            .refine_weight_norm
            .clone()
            .into_primitive()
            .tensor()
            .client;

        let compact_gid = client.resolve_tensor_int::<MainBackendBase>(global_from_compact_gid);
        let num_visible = client.resolve_tensor_int::<MainBackendBase>(num_visible);
        let refine_weight =
            client.resolve_tensor_float::<MainBackendBase>(refine_weight.into_primitive().tensor());

        let refine_accum = client.resolve_tensor_float::<MainBackendBase>(
            self.refine_weight_norm.clone().into_primitive().tensor(),
        );

        const WG_SIZE: u32 = 256;
        // Execute lazily the kernel with the launch information and the given buffers. For
        // simplicity, no vectorization is performed
        stats_gather_kernel::launch(
            &compact_gid.client,
            cubecl::CubeCount::Dynamic(
                create_dispatch_buffer(num_visible.clone(), [WG_SIZE, 1, 1])
                    .handle
                    .binding(),
            ),
            CubeDim::new(WG_SIZE, 1, 1),
            compact_gid.as_tensor_arg::<u32>(1),
            num_visible.as_tensor_arg::<u32>(1),
            refine_weight.as_tensor_arg::<f32>(2),
            refine_accum.as_tensor_arg::<f32>(1),
            w,
            h,
        );
    }
}

impl<B: Backend> RefineRecord<B> {
    pub(crate) fn keep(self, indices: burn::tensor::Tensor<B, 1, burn::prelude::Int>) -> Self {
        Self {
            refine_weight_norm: self.refine_weight_norm.select(0, indices),
        }
    }
}
