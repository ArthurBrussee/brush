use brush_kernel::create_dispatch_buffer;
use brush_render::BBase;
use burn::prelude::*;
use burn::tensor::ops::IntTensor;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::{BoolElement, cubecl};
use burn_fusion::Fusion;
use burn_fusion::client::FusionClient;
use glam::UVec2;
use tracing::trace_span;

use crate::stats_kernel::stats_gather_kernel;

type Fused<BT> = Fusion<BBase<BT>>;

pub(crate) struct RefineRecord<B: Backend> {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    pub refine_weight_norm: Tensor<B, 1>,
}

impl<B: Backend> RefineRecord<B> {
    pub(crate) fn new(num_points: u32, device: &B::Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<B, 1>::zeros([num_points as usize], device),
        }
    }
}

impl<BT: BoolElement> RefineRecord<Fused<BT>> {
    pub(crate) fn gather_stats(
        &self,
        refine_weight: Tensor<Fused<BT>, 1>,
        resolution: UVec2,
        global_from_compact_gid: IntTensor<Fused<BT>>,
        num_visible: IntTensor<Fused<BT>>,
    ) {
        let _span = trace_span!("Gather stats", sync_burn = true);

        let [w, h] = [resolution.x, resolution.y];
        let client = &self
            .refine_weight_norm
            .clone()
            .into_primitive()
            .tensor()
            .client;

        let compact_gid = client.resolve_tensor_int::<BBase<BT>>(global_from_compact_gid);
        let num_visible = client.resolve_tensor_int::<BBase<BT>>(num_visible);
        let refine_weight =
            client.resolve_tensor_float::<BBase<BT>>(refine_weight.into_primitive().tensor());

        let inner_client = &compact_gid.client;

        let refine_accum = client.resolve_tensor_float::<BBase<BT>>(
            self.refine_weight_norm.clone().into_primitive().tensor(),
        );

        const WG_SIZE: u32 = 256;
        // Execute lazily the kernel with the launch information and the given buffers. For
        // simplicity, no vectorization is performed
        stats_gather_kernel::launch(
            inner_client,
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
    pub fn keep(self, indices: Tensor<B, 1, Int>) -> Self {
        Self {
            refine_weight_norm: self.refine_weight_norm.select(0, indices),
        }
    }
}
