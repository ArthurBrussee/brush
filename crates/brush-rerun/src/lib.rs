// Noop on wasm but usefuly to have the types.
pub mod visualize_tools;

// Exclude entirely on wasm.
#[cfg(not(target_family = "wasm"))]
pub mod burn_to_rerun;
