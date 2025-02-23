use brush_kernel::create_dispatch_buffer;
use brush_render::{BBase, RenderAux};
use burn::backend::Autodiff;
use burn::prelude::*;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::{cubecl, BoolElement, FloatElement, IntElement};
use burn_fusion::client::FusionClient;
use burn_fusion::Fusion;
use tracing::trace_span;

use crate::stats_kernel::stats_gather_kernel;

type Fused<F, I, BT> = Fusion<BBase<F, I, BT>>;

pub(crate) struct RefineRecord<B: Backend> {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    refine_weight_norm: Tensor<B, 1>,
    visible_counts: Tensor<B, 1, Int>,
    max_radii: Tensor<B, 1>,
}

impl<B: Backend> RefineRecord<B> {
    pub(crate) fn new(num_points: u32, device: &B::Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<B, 1>::zeros([num_points as usize], device),
            visible_counts: Tensor::zeros([num_points as usize], device),
            max_radii: Tensor::zeros([num_points as usize], device),
        }
    }
}

impl<F: FloatElement, I: IntElement, BT: BoolElement> RefineRecord<Fused<F, I, BT>> {
    pub(crate) fn gather_stats(
        &self,
        refine_weight: Tensor<Fused<F, I, BT>, 1>,
        aux: RenderAux<Autodiff<Fused<F, I, BT>>>,
    ) {
        let _span = trace_span!("Gather stats", sync_burn = true);

        let [h, w] = aux.final_index.shape().dims();
        let client = &self
            .refine_weight_norm
            .clone()
            .into_primitive()
            .tensor()
            .client;

        let compact_gid = client
            .resolve_tensor_int::<BBase<F, I, BT>>(aux.global_from_compact_gid.into_primitive());
        let num_visible =
            client.resolve_tensor_int::<BBase<F, I, BT>>(aux.num_visible.into_primitive());
        let radii = client
            .resolve_tensor_float::<BBase<F, I, BT>>(aux.radii.inner().into_primitive().tensor());
        let refine_weight =
            client.resolve_tensor_float::<BBase<F, I, BT>>(refine_weight.into_primitive().tensor());
        let grad_counts = client
            .resolve_tensor_int::<BBase<F, I, BT>>(self.visible_counts.clone().into_primitive());

        let inner_client = &compact_gid.client;

        let refine_accum = client.resolve_tensor_float::<BBase<F, I, BT>>(
            self.refine_weight_norm.clone().into_primitive().tensor(),
        );
        let max_radii = client.resolve_tensor_float::<BBase<F, I, BT>>(
            self.max_radii.clone().into_primitive().tensor(),
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
            radii.as_tensor_arg::<f32>(1),
            refine_weight.as_tensor_arg::<f32>(2),
            refine_accum.as_tensor_arg::<f32>(1),
            grad_counts.as_tensor_arg::<u32>(1),
            max_radii.as_tensor_arg::<f32>(1),
            w as u32,
            h as u32,
        );
    }

    pub fn max_radii(&self) -> Tensor<Fused<F, I, BT>, 1> {
        self.max_radii.clone()
    }

    pub fn refine_weight(&self) -> Tensor<Fused<F, I, BT>, 1> {
        self.refine_weight_norm.clone() / self.visible_counts.clone().clamp_min(1).float()
    }
}
