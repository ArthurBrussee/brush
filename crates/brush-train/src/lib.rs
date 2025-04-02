#![recursion_limit = "256"]
pub mod config;
pub mod train;

mod adam_scaled;
mod multinomial;
mod quat_vec;
mod stats;
mod stats_kernel;
