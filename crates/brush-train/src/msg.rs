use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
};

#[derive(Clone)]
pub struct RefineStats {
    pub num_added: u32,
    pub num_pruned: u32,
}

#[derive(Clone)]
pub struct TrainStepStats<B: Backend> {
    pub pred_image: Tensor<B, 3>,

    pub num_intersections: Tensor<B, 1, Int>,
    pub num_visible: Tensor<B, 1, Int>,
    pub loss: Tensor<B, 1>,

    pub lr_mean: f64,
    pub lr_rotation: f64,
    pub lr_scale: f64,
    pub lr_coeffs: f64,
    pub lr_opac: f64,
}
