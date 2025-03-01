pub(crate) fn multinomial_sample(weights: &[f32], n: u32) -> Vec<i32> {
    let mut rng = rand::rng();
    rand::seq::index::sample_weighted(&mut rng, weights.len(), |i| weights[i], n as usize)
        .expect("Failed to sample")
        .iter()
        .map(|x| x as i32)
        .collect()
}
