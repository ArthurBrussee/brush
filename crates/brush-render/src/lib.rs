#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]
use burn::prelude::Tensor;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{ElementConversion, Int, TensorPrimitive, Transaction};
use burn_jit::JitBackend;
use burn_wgpu::WgpuRuntime;
use camera::Camera;
use shaders::helpers::TILE_WIDTH;
use tokio::sync::watch::{Receiver, Sender};

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

#[derive(Default, Debug, Clone)]
struct BwdAuxData {
    num_visible: u32,
    num_intersects: u32,
}

#[derive(Default, Debug, Clone)]
struct BwdAux {
    data: Option<BwdAuxData>,
}

impl BwdAux {
    fn new(num_visible: u32, num_intersects: u32) -> Self {
        Self {
            data: Some(BwdAuxData {
                num_visible,
                num_intersects,
            }),
        }
    }

    fn data(&self) -> &BwdAuxData {
        self.data
            .as_ref()
            .expect("Bwd state must be resolved before calling num_visible")
    }
}

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
    sender: Option<Sender<BwdAux>>,
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
            sender: self.sender,
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
    sender: Option<Sender<BwdAux>>,
}

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub num_visible: u32,
    pub num_intersections: u32,
}

const INTERSECTS_UPPER_BOUND: u32 = shaders::map_gaussian_to_intersects::WORKGROUP_SIZE[0] * 65535;
const GAUSSIANS_UPPER_BOUND: u32 = 256 * 65535;

impl<B: Backend> RenderAux<B> {
    pub async fn resolve_bwd_data(&self) {
        if let Some(send) = self.sender.clone() {
            if !send.is_closed() {
                let data = Transaction::default()
                    .register(self.num_visible.clone())
                    .register(self.num_intersections.clone())
                    .execute_async()
                    .await;

                let num_visible: i32 = data[0].to_vec().expect("Failed to resolve num_visible")[0];
                let num_intersections: i32 = data[1]
                    .to_vec()
                    .expect("Failed to resolve num_intersections")[0];
                let _ = send.send(BwdAux::new(num_visible as u32, num_intersections as u32));
            }
        }
    }

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
    rx: Receiver<BwdAux>,
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

type BBase = JitBackend<WgpuRuntime, f32, i32, u32>;
