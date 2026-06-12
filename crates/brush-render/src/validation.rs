use burn::tensor::Tensor;

/// Scan a tensor for NaN / Inf and out-of-range values. Logs range
/// violations; under `cfg(test)` / `debug-validation` NaN and Inf are
/// promoted to hard panics so CI surfaces them.
pub async fn validate_tensor_val<const D: usize>(
    tensor: Tensor<D>,
    name: &str,
    min_val: Option<f32>,
    max_val: Option<f32>,
) {
    let dims = tensor.dims();
    let data = tensor
        .clone()
        .into_data_async()
        .await
        .expect("Failed to read tensor data");
    let values = data
        .into_vec::<f32>()
        .expect("Failed to convert tensor to f32 vec");

    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut below_min_count = 0;
    let mut above_max_count = 0;
    let mut first_nan_idx: Option<usize> = None;
    let mut first_inf_idx: Option<usize> = None;

    for (i, &value) in values.iter().enumerate() {
        if value.is_nan() {
            nan_count += 1;
            first_nan_idx.get_or_insert(i);
        } else if value.is_infinite() {
            inf_count += 1;
            first_inf_idx.get_or_insert(i);
        } else {
            if let Some(min) = min_val
                && value < min
            {
                below_min_count += 1;
            }
            if let Some(max) = max_val
                && value > max
            {
                above_max_count += 1;
            }
        }
    }

    if nan_count > 0 || inf_count > 0 {
        log::error!(
            "tensor '{name}' shape {dims:?}: {nan_count} NaN (first @ {first_nan_idx:?}), \
             {inf_count} Inf (first @ {first_inf_idx:?}) of {} total",
            values.len(),
        );
    }
    if below_min_count > 0 {
        log::error!(
            "tensor '{name}': {below_min_count} values < {} of {}",
            min_val.unwrap(),
            values.len(),
        );
    }
    if above_max_count > 0 {
        log::error!(
            "tensor '{name}': {above_max_count} values > {} of {}",
            max_val.unwrap(),
            values.len(),
        );
    }

    #[cfg(any(test, feature = "debug-validation"))]
    {
        // Map a flat index to per-dim coordinates, to tell "one whole row"
        // from "scattered elements" at a glance in the failure output.
        let coords = move |idx: usize| -> Vec<usize> {
            let mut rem = idx;
            let mut out = vec![0; D];
            for d in (0..D).rev() {
                out[d] = rem % dims[d];
                rem /= dims[d];
            }
            out
        };
        let values_ref = &values;
        let bad_rows = move |pred: fn(f32) -> bool| -> Vec<usize> {
            let row_len: usize = dims[1..].iter().product();
            let mut rows: Vec<usize> = values_ref
                .iter()
                .enumerate()
                .filter(|(_, v)| pred(**v))
                .map(|(i, _)| i / row_len)
                .collect();
            rows.dedup();
            rows
        };

        // Bit patterns of the bad values discriminate failure modes: all
        // 0x7FC00000 (canonical quiet NaN) points at arithmetic; diverse
        // garbage mantissas point at memory corruption / OOB writes.
        let bad_bits = || -> Vec<String> {
            values
                .iter()
                .filter(|v| !v.is_finite())
                .take(8)
                .map(|v| format!("{:#010x}", v.to_bits()))
                .collect()
        };

        // On a hit, read the SAME tensor a second time. If the re-read is
        // clean, the buffer was fine and the first readback returned phantom
        // bytes (staging/copy race); if it still shows the same garbage, the
        // buffer itself is corrupted.
        let reread_verdict = if nan_count > 0 || inf_count > 0 {
            let values2 = tensor
                .into_data_async()
                .await
                .expect("Failed to re-read tensor data")
                .into_vec::<f32>()
                .expect("Failed to convert re-read to f32 vec");
            let bad2 = values2.iter().filter(|v| !v.is_finite()).count();
            let first_bad = values
                .iter()
                .position(|v| !v.is_finite())
                .unwrap_or_default();
            format!(
                "reread: {} non-finite (was {}), same idx bits {:#010x} -> {:#010x}",
                bad2,
                nan_count + inf_count,
                values[first_bad].to_bits(),
                values2.get(first_bad).copied().unwrap_or(0.0).to_bits(),
            )
        } else {
            String::new()
        };

        assert_eq!(
            nan_count,
            0,
            "tensor '{name}' shape {dims:?} has {nan_count} NaNs (first @ {first_nan_idx:?} = {:?}, rows {:?}, bits {:?}; {})",
            first_nan_idx.map(coords),
            bad_rows(f32::is_nan),
            bad_bits(),
            reread_verdict,
        );
        assert_eq!(
            inf_count,
            0,
            "tensor '{name}' shape {dims:?} has {inf_count} Infs (first @ {first_inf_idx:?} = {:?}, rows {:?}, bits {:?}; {})",
            first_inf_idx.map(coords),
            bad_rows(f32::is_infinite),
            bad_bits(),
            reread_verdict,
        );
    }
}

pub async fn validate_gradient<const D: usize>(gradient: Tensor<D>, name: &str) {
    validate_tensor_val(gradient, &format!("gradient_{name}"), None, None).await;
}
