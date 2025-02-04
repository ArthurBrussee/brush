use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};
use rand::distr::{weighted::WeightedIndex, Distribution};

pub(crate) async fn multinomial_sample<B: Backend>(
    weights: Tensor<B, 1>,
    n: u32,
) -> Tensor<B, 1, Int> {
    let device = weights.device();
    let weights = weights
        .into_data_async()
        .await
        .to_vec::<f32>()
        .expect("Failed to read weights");

    let mut rng = rand::rng();

    let dist = WeightedIndex::new(&weights).expect("Invalid weightes for sampling");
    let indices = (0..n).map(|_| dist.sample(&mut rng) as u32).collect();

    let result = TensorData::new(indices, [n as usize]);
    Tensor::from_data(result, &device)
}
