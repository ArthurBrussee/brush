#![recursion_limit = "256"]

pub mod eval;
pub mod ssim;
pub mod train;

pub mod image;
pub mod scene;

pub mod burn_glue;
mod kernels;
mod shaders;

mod adam_scaled;
mod multinomial;
mod stats;
mod stats_kernel;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;
