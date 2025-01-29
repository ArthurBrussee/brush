use brush_kernel::create_dispatch_buffer;
use brush_render::RenderAux;
use burn::backend::wgpu::JitBackend;
use burn::backend::Autodiff;
use burn::prelude::*;
use burn_fusion::client::FusionClient;
use burn_fusion::Fusion;
use burn_jit::cubecl::wgpu::WgpuRuntime;
use burn_jit::cubecl::CubeDim;
use burn_jit::{cubecl, BoolElement, FloatElement, IntElement};
use tracing::trace_span;

use crate::stats_kernel::stats_gather_kernel;

type Inner<F, I, BT> = JitBackend<WgpuRuntime, F, I, BT>;
type Fused<F, I, BT> = Fusion<Inner<F, I, BT>>;
type Diff<F, I, BT> = Autodiff<Fused<F, I, BT>>;

pub(crate) struct RefineRecord<B: Backend> {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    grad_2d_accum: Tensor<B, 1>,
    xy_grad_counts: Tensor<B, 1, Int>,
    max_radii: Tensor<B, 1>,
}

impl<B: Backend> RefineRecord<B> {
    pub(crate) fn new(num_points: usize, device: &B::Device) -> Self {
        Self {
            grad_2d_accum: Tensor::<B, 1>::zeros([num_points], device),
            xy_grad_counts: Tensor::zeros([num_points], device),
            max_radii: Tensor::zeros([num_points], device),
        }
    }
}

impl<F: FloatElement, I: IntElement, BT: BoolElement> RefineRecord<Fused<F, I, BT>> {
    pub(crate) fn gather_stats(
        &self,
        xys_grad: Tensor<Fused<F, I, BT>, 2>,
        aux: RenderAux<Diff<F, I, BT>>,
    ) {
        let _span = trace_span!("Gather stats", sync_burn = true);

        let [h, w] = aux.final_index.shape().dims();
        let client = &self.xy_grad_counts.clone().into_primitive().client;

        let compact_gid = client
            .resolve_tensor_int::<Inner<F, I, BT>>(aux.global_from_compact_gid.into_primitive());
        let num_visible =
            client.resolve_tensor_int::<Inner<F, I, BT>>(aux.num_visible.into_primitive());
        let radii = client
            .resolve_tensor_float::<Inner<F, I, BT>>(aux.radii.inner().into_primitive().tensor());
        let xys_grad =
            client.resolve_tensor_float::<Inner<F, I, BT>>(xys_grad.into_primitive().tensor());

        let inner_client = &compact_gid.client;

        let grad_2d_accum = client.resolve_tensor_float::<Inner<F, I, BT>>(
            self.grad_2d_accum.clone().into_primitive().tensor(),
        );
        let grad_counts = client
            .resolve_tensor_int::<Inner<F, I, BT>>(self.xy_grad_counts.clone().into_primitive());
        let max_radii = client.resolve_tensor_float::<Inner<F, I, BT>>(
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
            xys_grad.as_tensor_arg::<f32>(2),
            grad_2d_accum.as_tensor_arg::<f32>(1),
            grad_counts.as_tensor_arg::<u32>(1),
            max_radii.as_tensor_arg::<f32>(1),
            w as u32,
            h as u32,
        );
    }

    pub(crate) fn average_grad_2d(&self) -> Tensor<Fused<F, I, BT>, 1> {
        self.grad_2d_accum.clone() / self.xy_grad_counts.clone().clamp_min(1).float()
    }

    pub(crate) fn max_radii(&self) -> Tensor<Fused<F, I, BT>, 1> {
        self.max_radii.clone()
    }
}
