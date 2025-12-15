#![recursion_limit = "256"]

// Platform-specific modules.
#[cfg(target_os = "android")]
mod android;
#[cfg(target_family = "wasm")]
pub mod wasm;

// FFI for native training API.
#[cfg(feature = "training")]
#[cfg(not(target_family = "wasm"))]
pub mod ffi;
