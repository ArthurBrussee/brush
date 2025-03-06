#![recursion_limit = "256"]

use burn::prelude::{Backend, Tensor};
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{ElementConversion, Int, TensorPrimitive};
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use burn_wgpu::graphics::AutoGraphicsApi;
use burn_wgpu::{RuntimeOptions, WgpuDevice, WgpuRuntime};
use camera::Camera;
use shaders::helpers::TILE_WIDTH;
use wgpu::{Adapter, Device, Queue};

mod burn_glue;
mod dim_check;
mod kernels;
pub mod shaders;

pub mod sh;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
pub mod render;

#[derive(Debug, Clone)]
pub struct RenderAuxPrimitive<B: Backend> {
    /// The packed projected splat information, see `ProjectedSplat` in helpers.wgsl
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub num_intersections: IntTensor<B>,
    pub num_visible: IntTensor<B>,
    pub final_index: IntTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub radii: FloatTensor<B>,
}

impl<B: Backend> RenderAuxPrimitive<B> {
    pub fn into_wrapped(self) -> RenderAux<B> {
        RenderAux {
            num_intersections: Tensor::from_primitive(self.num_intersections),
            num_visible: Tensor::from_primitive(self.num_visible),
            final_index: Tensor::from_primitive(self.final_index),
            tile_offsets: Tensor::from_primitive(self.tile_offsets),
            compact_gid_from_isect: Tensor::from_primitive(self.compact_gid_from_isect),
            global_from_compact_gid: Tensor::from_primitive(self.global_from_compact_gid),
            radii: Tensor::from_primitive(TensorPrimitive::Float(self.radii)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    /// The packed projected splat information, see `ProjectedSplat` in helpers.wgsl
    pub num_intersections: Tensor<B, 1, Int>,
    pub num_visible: Tensor<B, 1, Int>,
    pub final_index: Tensor<B, 2, Int>,
    pub tile_offsets: Tensor<B, 1, Int>,
    pub compact_gid_from_isect: Tensor<B, 1, Int>,
    pub global_from_compact_gid: Tensor<B, 1, Int>,
    pub radii: Tensor<B, 1>,
}

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub num_visible: u32,
    pub num_intersections: u32,
}

const INTERSECTS_UPPER_BOUND: u32 = shaders::map_gaussian_to_intersects::WORKGROUP_SIZE[0] * 65535;
const GAUSSIANS_UPPER_BOUND: u32 = 256 * 65535;

impl<B: Backend> RenderAux<B> {
    #[allow(clippy::single_range_in_vec_init)]
    pub fn calc_tile_depth(&self) -> Tensor<B, 2, Int> {
        let bins = self.tile_offsets.clone();
        let n_bins = bins.dims()[0];
        let max = bins.clone().slice([1..n_bins]);
        let min = bins.slice([0..n_bins - 1]);
        let [h, w] = self.final_index.shape().dims();
        let [ty, tx] = [
            h.div_ceil(TILE_WIDTH as usize),
            w.div_ceil(TILE_WIDTH as usize),
        ];
        (max - min).reshape([ty, tx])
    }

    pub fn debug_assert_valid(self) {
        let num_intersections = self.num_intersections.into_scalar().elem::<i32>();
        let num_points = self.radii.dims()[0] as u32;
        let num_visible = self.num_visible.into_scalar().elem::<i32>();

        // let [h, w] = self.final_index.dims();
        // let [ty, tx] = [
        //     (h as u32).div_ceil(TILE_WIDTH),
        //     (w as u32).div_ceil(TILE_WIDTH),
        // ];

        // 'Visible' gaussians seemingly can still generate 0 intersections.
        // assert!(
        //     num_visible <= num_intersections,
        //     "somehow there are more gaussian visible than intersections."
        // );

        assert!(
            num_intersections >= 0 && num_intersections < INTERSECTS_UPPER_BOUND as i32,
            "Too many intersections, Brush currently can't handle this. {num_intersections} > {INTERSECTS_UPPER_BOUND}"
        );

        assert!(
            num_visible >= 0 && num_visible <= num_points as i32,
            "Something went wrong when calculating the number of visible gaussians. {num_visible} > {num_points}"
        );
        assert!(
            num_visible >= 0 && num_visible < GAUSSIANS_UPPER_BOUND as i32,
            "Brush doesn't support this many gaussians currently. {num_visible} > {GAUSSIANS_UPPER_BOUND}"
        );

        let final_index = self
            .final_index
            .into_data()
            .to_vec::<i32>()
            .expect("Failed to fetch final index");
        for &final_index in &final_index {
            assert!(
                final_index >= 0 && final_index <= num_intersections,
                "Final index exceeds bounds. Final index {final_index}, num_intersections: {num_intersections}"
            );
        }

        let tile_offsets = self
            .tile_offsets
            .into_data()
            .to_vec::<i32>()
            .expect("Failed to fetch tile offsets");
        for &offsets in &tile_offsets {
            assert!(
                offsets >= 0 && offsets <= num_intersections,
                "Tile offsets exceed bounds. Value: {offsets}, num_intersections: {num_intersections}"
            );
        }

        for i in 0..tile_offsets.len() - 1 {
            let start = tile_offsets[i];
            let end = tile_offsets[i + 1];
            assert!(
                start >= 0 && end >= 0,
                "Invalid elements in tile offsets. Start {start} ending at {end}"
            );
            assert!(
                end >= start,
                "Invalid elements in tile offsets. Start {start} ending at {end}"
            );
            assert!(
                end - start <= num_visible,
                "One tile has more hits than total visible splats. Start {start} ending at {end}"
            );
        }

        let compact_gid_from_isect = &self
            .compact_gid_from_isect
            .into_data()
            .to_vec::<i32>()
            .expect("Failed to fetch compact_gid_from_isect")[0..num_intersections as usize];

        for &compact_gid in compact_gid_from_isect {
            assert!(
                compact_gid >= 0 && compact_gid < num_visible,
                "Invalid gaussian ID in intersection buffer. {compact_gid} out of {num_visible}"
            );
        }

        // assert that every ID in global_from_compact_gid is valid.
        let global_from_compact_gid = &self
            .global_from_compact_gid
            .into_data()
            .to_vec::<i32>()
            .expect("Failed to fetch global_from_compact_gid")[0..num_visible as usize];

        for &global_gid in global_from_compact_gid {
            assert!(
                global_gid >= 0 && global_gid < num_points as i32,
                "Invalid gaussian ID in global_from_compact_gid buffer. {global_gid} out of {num_points}"
            );
        }
    }
}

/// The base WGPU backend these extension are written against. The bool type varies
/// between the native vulkan and WebGPU backend.
pub type BBase<BT> = CubeBackend<WgpuRuntime, f32, i32, BT>;
pub type BFused<BT> = Fusion<BBase<BT>>;

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
        raw_opacity: FloatTensor<B>,
        render_u32_buffer: bool,
    ) -> (FloatTensor<B>, RenderAuxPrimitive<B>);
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
    };
    burn_wgpu::init_device(setup, burn_options())
}

pub async fn burn_init_setup() -> WgpuDevice {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
        .await;
    WgpuDevice::DefaultDevice
}
