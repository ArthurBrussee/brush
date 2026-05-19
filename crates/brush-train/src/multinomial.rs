pub(crate) fn multinomial_sample(weights: &[f32], n: u32) -> Vec<i32> {
    let mut rng = rand::rng();
    // Sanitize: only finite, non-negative weights are valid sampling
    // mass. A non-finite (NaN/±Inf) or negative weight here is a
    // poisoned densification weight latched by `RefineRecord::gather_stats`
    // — historically the source of the "Failed to sample from weights"
    // crash (issue #128 / commit a1f02c65, which only scrubbed NaN). A
    // negative makes `rand`'s `sample_weighted` return `InvalidWeight`
    // (hard panic); a +Inf makes it always-pick that index (silent
    // densification collapse) and occasionally panic on a NaN sort key.
    // Mapping all of these to 0.0 makes a blown-up splat simply
    // ineligible for growth instead of crashing or monopolizing it.
    rand::seq::index::sample_weighted(
        &mut rng,
        weights.len(),
        |i| {
            let w = weights[i];
            if w.is_finite() && w > 0.0 { w } else { 0.0 }
        },
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
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = test)]
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

    #[wasm_bindgen_test(unsupported = test)]
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

    #[wasm_bindgen_test(unsupported = test)]
    fn test_all_zero_weights() {
        // Discovered behavior: returns empty vec when all weights are zero
        let zero_weights = vec![0.0, 0.0, 0.0];
        let result = multinomial_sample(&zero_weights, 1);

        // Function returns empty vector when it cannot sample any valid indices
        assert_eq!(result.len(), 0);
    }

    // Regression: a poisoned densification weight (latched +Inf from
    // gather_stats, or a negative) must NOT crash and must NOT be
    // selected — it should be treated as zero mass. Before the guard was
    // widened, the negative case hit `WeightError::InvalidWeight` →
    // `panic!("Failed to sample from weights …")` (issue #128) and the
    // +Inf case made that index win every draw (densification collapse).

    #[wasm_bindgen_test(unsupported = test)]
    fn inf_weight_is_treated_as_zero_not_a_crash() {
        let weights = vec![1.0, f32::INFINITY, 2.0, 0.5];
        let samples = multinomial_sample(&weights, 3);
        assert_eq!(samples.len(), 3);
        assert!(
            !samples.contains(&1),
            "index 1 had +Inf weight and must be ineligible, got {samples:?}"
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn negative_weight_is_treated_as_zero_not_a_crash() {
        let weights = vec![1.0, -1.0, 2.0, 0.5];
        let samples = multinomial_sample(&weights, 3);
        assert_eq!(samples.len(), 3);
        assert!(
            !samples.contains(&1),
            "index 1 had a negative weight and must be ineligible, got {samples:?}"
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn mixed_nan_inf_negative_all_scrubbed() {
        // All the poison kinds at once: only indices 0 and 4 are valid.
        let weights = vec![3.0, f32::NAN, f32::INFINITY, -5.0, 1.0];
        let samples = multinomial_sample(&weights, 2);
        assert_eq!(samples.len(), 2);
        for s in samples {
            assert!(s == 0 || s == 4, "sampled poisoned index {s}");
        }
    }
}
