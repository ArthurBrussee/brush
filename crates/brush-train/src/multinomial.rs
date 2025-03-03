pub(crate) fn multinomial_sample(weights: &[f32], n: u32) -> Vec<i32> {
    let mut rng = rand::rng();
    rand::seq::index::sample_weighted(&mut rng, weights.len(), |i| weights[i], n as usize)
        .unwrap_or_else(|_| {
            panic!(
                "Failed to sample from weights. Counts: {} Infinities: {} NaN: {}",
                weights.len(),
                weights.iter().filter(|x| x.is_infinite()).count(),
                weights.iter().filter(|x| x.is_nan()).count()
            )
        })
        .iter()
        .map(|x| x as i32)
        .collect()
}
