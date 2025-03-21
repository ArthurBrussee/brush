#![recursion_limit = "256"]

pub mod camera_controls;
mod panels;

mod app;
pub mod running_process;

pub use app::*;
use burn::backend::Autodiff;
use burn_wgpu::Wgpu;
pub type MainBackend = Autodiff<Wgpu>;
