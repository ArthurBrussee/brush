#![recursion_limit = "256"]

// Platform-specific modules.
#[cfg(target_os = "android")]
mod android;
#[cfg(target_family = "wasm")]
pub mod wasm;

#[cfg(any(target_family = "wasm", target_os = "android"))]
mod ui;
