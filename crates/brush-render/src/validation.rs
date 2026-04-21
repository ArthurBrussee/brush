use burn::{prelude::Backend, tensor::Tensor};

/// Scan a tensor for NaN / Inf and out-of-range values. Logs range
/// violations; under `cfg(test)` / `debug-validation` NaN and Inf are
/// promoted to hard panics so CI surfaces them.
pub async fn validate_tensor_val<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    name: &str,
    min_val: Option<f32>,
    max_val: Option<f32>,
) {
    let data = tensor
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
            "tensor '{name}': {nan_count} NaN (first @ {first_nan_idx:?}), \
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
        assert_eq!(
            nan_count, 0,
            "tensor '{name}' has {nan_count} NaNs (first @ {first_nan_idx:?})"
        );
        assert_eq!(
            inf_count, 0,
            "tensor '{name}' has {inf_count} Infs (first @ {first_inf_idx:?})"
        );
    }
}

pub async fn validate_gradient<B: Backend, const D: usize>(gradient: Tensor<B, D>, name: &str) {
    validate_tensor_val(gradient, &format!("gradient_{name}"), None, None).await;
}
