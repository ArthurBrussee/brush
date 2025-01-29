#![recursion_limit = "256"]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]

use burn::backend::Autodiff;
use burn::prelude::Tensor;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{ElementConversion, Int, TensorPrimitive};
use burn_fusion::Fusion;
use burn_jit::JitBackend;
use burn_wgpu::graphics::AutoGraphicsApi;
use burn_wgpu::{RuntimeOptions, SpirvCompiler, WgpuDevice, WgpuRuntime};
use camera::Camera;
use shaders::helpers::TILE_WIDTH;
use wgpu::{Adapter, Device, Queue};

mod burn_glue;
mod dim_check;
mod kernels;
mod safetensor_utils;
mod shaders;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
pub mod render;

type InnerWgpu<C, F, I, B> = JitBackend<WgpuRuntime<C>, F, I, B>;

// Currently active backends.
pub type Compiler = SpirvCompiler;
pub type BJit = JitBackend<WgpuRuntime<SpirvCompiler>, f32, i32, u8>;
pub type BInner = Fusion<BJit>;
pub type BDiff = Autodiff<BInner>;

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
    fn into_wrapped(self) -> RenderAux<B> {
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
pub struct RenderAux<B: burn::prelude::Backend> {
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

#[derive(Debug, Clone)]
pub struct SplatGrads<B: Backend> {
    v_means: FloatTensor<B>,
    v_quats: FloatTensor<B>,
    v_scales: FloatTensor<B>,
    v_coeffs: FloatTensor<B>,
    v_raw_opac: FloatTensor<B>,
    v_xy: FloatTensor<B>,
}

#[derive(Debug, Clone)]
pub struct GaussianBackwardState<B: Backend> {
    means: FloatTensor<B>,
    quats: FloatTensor<B>,
    log_scales: FloatTensor<B>,
    raw_opac: FloatTensor<B>,

    out_img: FloatTensor<B>,

    projected_splats: FloatTensor<B>,
    uniforms_buffer: IntTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    global_from_compact_gid: IntTensor<B>,
    tile_offsets: IntTensor<B>,
    final_index: IntTensor<B>,

    sh_degree: u32,
}

// Custom operations in Burn work by extending the backend with an extra func.
pub trait Backend: burn::tensor::backend::Backend {
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
        means: FloatTensor<Self>,
        xy_grad_dummy: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacity: FloatTensor<Self>,
        render_u32_buffer: bool,
    ) -> (FloatTensor<Self>, RenderAuxPrimitive<Self>);

    /// Backward pass for `render_splats`.
    ///
    /// Do not use directly, `render_splats` will use this to calculate gradients.
    #[allow(unused_variables)]
    fn render_splats_bwd(
        state: GaussianBackwardState<Self>,
        v_output: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        panic!("Do not call this manually.");
    }
}

pub trait AutodiffBackend:
    Backend + burn::tensor::backend::AutodiffBackend<InnerBackend: Backend>
{
}

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

fn set_hard_floats(adapter: &Adapter) {
    let hard_floats = adapter
        .features()
        .contains(wgpu::Features::SHADER_FLOAT32_ATOMIC);

    render::set_hard_floats_available(hard_floats);
    log::info!("Running with native atomic floats: {hard_floats}");
}

pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    set_hard_floats(&adapter);

    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(&wgpu::InstanceDescriptor::default()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
    };
    burn_wgpu::init_device(setup, burn_options())
}

pub async fn burn_init_setup() -> WgpuDevice {
    let setup =
        burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
            .await;
    set_hard_floats(&setup.adapter);
    WgpuDevice::DefaultDevice
}
