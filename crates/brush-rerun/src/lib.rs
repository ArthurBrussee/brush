// Exclude entirely on wasm.
#[cfg(not(target_family = "wasm"))]
pub mod burn_to_rerun;
