#![recursion_limit = "256"]

#[allow(unused)]
use brush_app::{App, AppCreateCb};

use burn::{prelude::Backend, tensor::Tensor};
use burn_wgpu::Wgpu;
#[allow(unused)]
use tokio::sync::oneshot::error::RecvError;
use tokio_with_wasm::alias as tokio_wasm;

#[cfg(not(target_family = "wasm"))]
type MainResult = Result<(), clap::Error>;

#[cfg(target_family = "wasm")]
type MainResult = Result<(), ()>;

fn main() {
    tokio_wasm::task::spawn(async {
        let device = brush_render::burn_init_setup().await;
        let vals: Tensor<Wgpu, 1> = Tensor::ones([10], &device);
        vals.sum()
    });
}
