use burn::{
    prelude::Int,
    tensor::{Bool, Device, Tensor},
};
use tracing::trace_span;

pub(crate) struct RefineRecord {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    pub refine_weight_norm: Tensor<1>,
    pub vis_weight: Tensor<1>,
    pub max_screen_size: Tensor<1>,
}

impl RefineRecord {
    pub(crate) fn new(num_points: u32, device: &Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<1>::zeros([num_points as usize], device),
            vis_weight: Tensor::<1>::zeros([num_points as usize], device),
            max_screen_size: Tensor::<1>::zeros([num_points as usize], device),
        }
    }

    pub(crate) fn above_threshold(&self, threshold: f32) -> Tensor<1, Bool> {
        self.refine_weight_norm
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    pub(crate) fn above_screen_size(&self, threshold: f32) -> Tensor<1, Bool> {
        self.max_screen_size
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    pub(crate) fn gather_stats(
        &mut self,
        refine_weight: Tensor<1>,
        visible: Tensor<1>,
        screen_radius: Tensor<1>,
    ) {
        let _span = trace_span!("Gather stats").entered();
        // `refine_weight` is the densification gradient proxy from the
        // rasterize backward (`grad.refine`). It is a non-negative
        // magnitude by construction but is never clamped or finite-checked
        // upstream (project_backwards passes it raw; bwd_validate ignores
        // it), so a divergent splat can hand us a NaN/±Inf or an
        // astronomically large value. Because the line below latches the
        // running maximum, a single bad iteration would poison this
        // splat's growth weight *forever* — feeding multinomial_sample a
        // non-finite weight (the "Failed to sample from weights" crash,
        // issue #128) or letting one splat monopolize the growth budget.
        // Scrub to a finite, non-negative value before latching.
        let finite = refine_weight.clone().is_finite();
        let refine_weight = refine_weight
            .mask_fill(finite.bool_not(), 0.0)
            .clamp_min(0.0);
        self.refine_weight_norm = refine_weight.max_pair(self.refine_weight_norm.clone());
        self.vis_weight = self.vis_weight.clone() + visible;
        self.max_screen_size = screen_radius.max_pair(self.max_screen_size.clone());
    }

    pub(crate) fn vis_mask(&self) -> Tensor<1, Bool> {
        self.vis_weight.clone().greater_elem(0.0)
    }

    pub(crate) fn keep(self, indices: Tensor<1, Int>) -> Self {
        Self {
            refine_weight_norm: self.refine_weight_norm.select(0, indices.clone()),
            vis_weight: self.vis_weight.clone().select(0, indices.clone()),
            max_screen_size: self.max_screen_size.select(0, indices),
        }
    }
}
