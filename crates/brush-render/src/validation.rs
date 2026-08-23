//! Validation readbacks.
//!
//! Every validated tensor is copied from the GPU to the host, which syncs the
//! device. That is fine for correctness tests and ruinous for any measurement:
//! a validated render can cost several times what the render itself does, and
//! it costs more on backends whose reads block (cubecl-metal) than on ones
//! whose reads are async (wgpu). So this is switchable at runtime, and it says
//! so out loud the first time it runs.

use core::sync::atomic::{AtomicU8, Ordering};

const UNSET: u8 = 0;
const ON: u8 = 1;
const OFF: u8 = 2;

static STATE: AtomicU8 = AtomicU8::new(UNSET);

/// Turn validation readbacks on or off for this process.
///
/// Call this with `false` before timing anything. Benchmarks in this workspace
/// do; so should any ad-hoc timing harness.
pub fn set_enabled(on: bool) {
    STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

/// Whether validation readbacks currently run.
///
/// Defaults to on wherever validation is compiled in (`cfg(test)` or the
/// `debug-validation` feature), except under `cargo bench`.
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => {
            let default_on = cfg!(any(test, feature = "debug-validation")) && !is_bench_run();
            set_enabled(default_on);
            default_on
        }
    }
}

fn is_bench_run() -> bool {
    #[cfg(not(target_family = "wasm"))]
    {
        std::env::args().any(|a| a == "--bench")
    }
    #[cfg(target_family = "wasm")]
    {
        false
    }
}

/// Say once, loudly, that timings taken now are not representative.
pub(crate) fn warn_once() {
    #[cfg(not(target_family = "wasm"))]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        // Deliberately not `log::warn!`: this has to be visible in a test or
        // bench binary, which usually has no logger installed.
        #[allow(clippy::print_stderr)]
        ONCE.call_once(|| {
            eprintln!(
                "brush-render: VALIDATION READBACKS ARE ON (cfg(test) / \
                 feature \"debug-validation\"). Every render syncs the GPU to \
                 scan tensors on the host, so any timing you take now is \
                 dominated by that. Call \
                 brush_render::validation::set_enabled(false) to turn it off."
            );
        });
    }
}

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
    let data = tensor
        .into_data_async()
        .await
        .expect("Failed to read tensor data");
    let values = data
        .try_into_vec::<f32>()
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

pub async fn validate_gradient<const D: usize>(gradient: Tensor<D>, name: &str) {
    validate_tensor_val(gradient, &format!("gradient_{name}"), None, None).await;
}
