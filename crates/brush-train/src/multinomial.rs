pub(crate) fn multinomial_sample(weights: &[f32], n: u32) -> Vec<i32> {
    let mut rng = rand::rng();
    rand::seq::index::sample_weighted(
        &mut rng,
        weights.len(),
        |i| if weights[i].is_nan() { 0.0 } else { weights[i] },
        n as usize,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multinomial_sampling() {
        // Test the complete multinomial sampling workflow (samples indices without replacement)
        let weights = vec![0.1, 0.3, 0.4, 0.2];
        let samples = multinomial_sample(&weights, 3);

        assert_eq!(samples.len(), 3);
        for &sample in &samples {
            assert!(sample >= 0 && sample < weights.len() as i32);
        }
        // Should not have duplicates (sampling without replacement)
        let mut unique_samples = samples.clone();
        unique_samples.sort();
        unique_samples.dedup();
        assert_eq!(unique_samples.len(), samples.len());

        // Test edge case: sampling all indices
        let single_weight = vec![1.0];
        let single_samples = multinomial_sample(&single_weight, 1);
        assert_eq!(single_samples.len(), 1);
        assert_eq!(single_samples[0], 0);
    }

    #[test]
    fn test_nan_weight_handling() {
        // Test that NaN weights are handled (converted to 0.0)
        let weights_with_nan = vec![0.5, f32::NAN, 0.3, 0.2];
        let samples = multinomial_sample(&weights_with_nan, 2);

        assert_eq!(samples.len(), 2);
        // Should never sample index 1 (NaN weight becomes 0.0)
        assert!(!samples.contains(&1));
        // Should only sample from valid indices
        for &sample in &samples {
            assert!(sample == 0 || sample == 2 || sample == 3);
        }
    }

    #[test]
    fn test_all_zero_weights() {
        // Discovered behavior: returns empty vec when all weights are zero
        let zero_weights = vec![0.0, 0.0, 0.0];
        let result = multinomial_sample(&zero_weights, 1);

        // Function returns empty vector when it cannot sample any valid indices
        assert_eq!(result.len(), 0);
    }
}
