use brush_render::MainBackend;
use burn::{
    prelude::{Backend, Int},
    tensor::{Bool, Tensor},
};
use tracing::trace_span;

pub(crate) struct RefineRecord<B: Backend> {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    pub refine_weight_norm: Tensor<B, 1>,
    pub weight_counts: Tensor<B, 1>,
}

impl<B: Backend> RefineRecord<B> {
    pub(crate) fn new(num_points: u32, device: &B::Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<B, 1>::zeros([num_points as usize], device),
            weight_counts: Tensor::<B, 1>::zeros([num_points as usize], device),
        }
    }

    pub(crate) fn above_threshold(&self, threshold: f32) -> Tensor<B, 1, Bool> {
        let counts = self.weight_counts.clone().clamp_min(1.0);
        (self.refine_weight_norm.clone() / counts).greater_elem(threshold)
    }
}

impl RefineRecord<MainBackend> {
    pub(crate) fn gather_stats(
        &mut self,
        refine_weight: Tensor<MainBackend, 1>,
        visible: Tensor<MainBackend, 1>,
    ) {
        let _span = trace_span!("Gather stats").entered();
        self.refine_weight_norm = refine_weight.max_pair(self.refine_weight_norm.clone());
        self.weight_counts = self.weight_counts.clone() + visible;
    }
}

impl<B: Backend> RefineRecord<B> {
    pub(crate) fn keep(self, indices: Tensor<B, 1, Int>) -> Self {
        Self {
            refine_weight_norm: self.refine_weight_norm.select(0, indices.clone()),
            weight_counts: self.weight_counts.select(0, indices),
        }
    }
}
