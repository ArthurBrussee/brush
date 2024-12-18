pub mod data_source;
mod orbit_controls;
mod panels;
pub mod process_loop;

#[cfg(not(target_family = "wasm"))]
mod rerun_tools;

mod app;

#[cfg(not(target_family = "wasm"))]
pub mod cli;

pub use app::*;
use burn::backend::Autodiff;
use burn_wgpu::Wgpu;
pub type MainBackend = Autodiff<Wgpu>;
