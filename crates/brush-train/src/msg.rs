use burn::tensor::Tensor;

#[derive(Clone)]
pub struct RefineStats {
    pub num_added: u32,
    pub num_pruned: u32,
    pub total_splats: u32,
}

#[derive(Clone)]
pub struct TrainStepStats {
    pub num_visible: u32,
    pub lr_mean: f64,
    pub lr_rotation: f64,
    pub lr_scale: f64,
    pub lr_coeffs: f64,
    pub lr_opac: f64,
    // Non-autodiff inner tensor; consumers read the scalar lazily so disabled
    // logging doesn't force a GPU readback.
    pub loss: Tensor<1>,
}
