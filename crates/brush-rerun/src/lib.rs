// Noop on wasm but useful to have the type available.
pub mod visualize_tools;

// Exclude entirely on wasm.
#[cfg(not(target_family = "wasm"))]
pub mod burn_to_rerun;
