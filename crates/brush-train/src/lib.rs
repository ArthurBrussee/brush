#![recursion_limit = "256"]

pub mod config;
pub mod eval;
pub mod msg;
pub mod train;

mod adam_scaled;
mod multinomial;
mod quat_vec;
mod ssim;
mod stats;

#[cfg(all(test, not(target_family = "wasm")))]
mod integration_tests;
