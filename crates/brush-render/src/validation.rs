use burn::{prelude::Backend, tensor::Tensor};

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

    for &value in &values {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            inf_count += 1;
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
            "Tensor '{}' contains invalid values: {} NaN, {} infinite (out of {} total).\nSample: {:?}",
            name,
            nan_count,
            inf_count,
            values.len(),
            &values[0..values.len().min(16)]
        );
    }

    if below_min_count > 0 {
        log::error!(
            "Tensor '{}' contains {} values below minimum {} (out of {} total)",
            name,
            below_min_count,
            min_val.unwrap(),
            values.len()
        );
    }

    if above_max_count > 0 {
        log::error!(
            "Tensor '{}' contains {} values above maximum {} (out of {} total)",
            name,
            above_max_count,
            max_val.unwrap(),
            values.len()
        );
    }
}

pub async fn validate_gradient<B: Backend, const D: usize>(gradient: Tensor<B, D>, name: &str) {
    validate_tensor_val(gradient, &format!("gradient_{name}"), None, None).await;
}
