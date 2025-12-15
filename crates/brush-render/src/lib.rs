#![recursion_limit = "256"]

use burn::prelude::Backend;
use burn::tensor::ops::FloatTensor;
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use burn_wgpu::graphics::{AutoGraphicsApi, GraphicsApi};
use burn_wgpu::{RuntimeOptions, WgpuDevice, WgpuRuntime};
use camera::Camera;
use clap::ValueEnum;
use glam::Vec3;
use render_aux::RenderAux;
use wgpu::{Adapter, Device, Queue};

mod burn_glue;
mod dim_check;
pub mod render_aux;
pub mod shaders;

pub mod sh;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
mod get_tile_offset;
pub mod render;
pub mod validation;

pub type MainBackendBase = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type MainBackend = Fusion<MainBackendBase>;

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub num_visible: u32,
    pub num_intersections: u32,
}

// The maximum number of intersections that can be rendered.
//
// Bounded by max nr. of dispatches for the intersection kernel.
const INTERSECTS_UPPER_BOUND: u32 = 512 * 65535;
// The maximum number of gaussians that can be rendered.
const GAUSSIANS_UPPER_BOUND: u32 = 256 * 65535;

pub trait SplatForward<B: Backend> {
    /// Render splats to a buffer.
    ///
    /// This projects the gaussians, sorts them, and rasterizes them to a buffer, in a
    /// differentiable way.
    /// The arguments are all passed as raw tensors. See [`Splats`] for a convenient Module that wraps this fun
    /// The [`xy_grad_dummy`] variable is only used to carry screenspace xy gradients.
    /// This function can optionally render a "u32" buffer, which is a packed RGBA (8 bits per channel)
    /// buffer. This is useful when the results need to be displayed immediately.
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<B>,
        log_scales: FloatTensor<B>,
        quats: FloatTensor<B>,
        sh_coeffs: FloatTensor<B>,
        raw_opacities: FloatTensor<B>,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<B>, RenderAux<B>);
}

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(&wgpu::InstanceDescriptor::default()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    burn_wgpu::init_device(setup, burn_options())
}

pub async fn burn_init_setup() -> WgpuDevice {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
        .await;
    WgpuDevice::DefaultDevice
}

#[derive(Default, ValueEnum, Clone, Copy, Eq, PartialEq, Debug)]
pub enum AlphaMode {
    #[default]
    Masked,
    Transparent,
}
