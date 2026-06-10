//! GPU-resident binary-search bracket state.
//!
//! All per-crossing tensors (`left_pos`, `right_pos`, `left_sdf`) live on
//! the GPU from construction until [`RefineState::finish`]. Each of the
//! [`N_STEPS`] fixed steps is a small chain of kernel launches with no
//! CPU round-trips: [`bisect::compute_midpoints_kernel`] writes the
//! bracket midpoints, the per-view integrate kernels fold a `min_alpha`
//! aggregator over them, and [`bisect::bracket_update_kernel`] shrinks
//! one bracket endpoint per crossing. `finish` reads back the converged
//! midpoints once. Every crossing runs all steps: convergence tracking
//! measured as pure overhead at current extraction speeds.

use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_render::kernels;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::tensor::TensorData;
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use glam::Vec3;

use crate::marching_tet::Crossing;

pub const N_STEPS: usize = 8;

type B = MainBackendBase;

pub struct RefineState {
    left_pos_t: FloatTensor<B>,
    right_pos_t: FloatTensor<B>,
    left_sdf_t: FloatTensor<B>,
    right_sdf_t: FloatTensor<B>,
    /// Scratch for the per-step midpoint output.
    mid_pos_t: FloatTensor<B>,
    n_crossings: usize,
}

impl RefineState {
    pub fn new(
        crossings: &[Crossing],
        vertex_positions: &[Vec3],
        vertex_sdf: &[f32],
        device: &WgpuDevice,
    ) -> Self {
        let n = crossings.len();

        let mut lp: Vec<f32> = Vec::with_capacity(n * 3);
        let mut rp: Vec<f32> = Vec::with_capacity(n * 3);
        let mut ls: Vec<f32> = Vec::with_capacity(n);
        let mut rs: Vec<f32> = Vec::with_capacity(n);
        for c in crossings {
            let a = vertex_positions[c.a as usize];
            let b = vertex_positions[c.b as usize];
            lp.extend_from_slice(&[a.x, a.y, a.z]);
            rp.extend_from_slice(&[b.x, b.y, b.z]);
            ls.push(vertex_sdf[c.a as usize]);
            rs.push(vertex_sdf[c.b as usize]);
        }

        let left_pos_t = B::float_from_data(TensorData::new(lp, [n, 3]), device);
        let right_pos_t = B::float_from_data(TensorData::new(rp, [n, 3]), device);
        let left_sdf_t = B::float_from_data(TensorData::new(ls, [n]), device);
        let right_sdf_t = B::float_from_data(TensorData::new(rs, [n]), device);
        let mid_pos_t = B::float_from_data(TensorData::new(vec![0.0f32; n * 3], [n, 3]), device);

        Self {
            left_pos_t,
            right_pos_t,
            left_sdf_t,
            right_sdf_t,
            mid_pos_t,
            n_crossings: n,
        }
    }

    pub fn n_crossings(&self) -> usize {
        self.n_crossings
    }

    /// Launches the midpoint kernel. Returns a tensor handle suitable as
    /// the `points` input to the integrate kernels.
    pub fn compute_midpoints_t(&self) -> FloatTensor<B> {
        let client = self.left_pos_t.client.clone();
        kernels::bisect::compute_midpoints_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(self.n_crossings as u32, kernels::bisect::WG_SIZE),
            CubeDim::new_1d(kernels::bisect::WG_SIZE),
            self.left_pos_t.clone().into_tensor_arg(),
            self.right_pos_t.clone().into_tensor_arg(),
            self.mid_pos_t.clone().into_tensor_arg(),
            self.n_crossings as u32,
        );
        self.mid_pos_t.clone()
    }

    /// Updates the bracket using a freshly-integrated `min_alpha_t`.
    pub fn update_bracket_t(&mut self, min_alpha_t: FloatTensor<B>, iso: f32) {
        let client = self.left_pos_t.client.clone();
        kernels::bisect::bracket_update_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(self.n_crossings as u32, kernels::bisect::WG_SIZE),
            CubeDim::new_1d(kernels::bisect::WG_SIZE),
            self.left_pos_t.clone().into_tensor_arg(),
            self.right_pos_t.clone().into_tensor_arg(),
            self.left_sdf_t.clone().into_tensor_arg(),
            self.right_sdf_t.clone().into_tensor_arg(),
            self.mid_pos_t.clone().into_tensor_arg(),
            min_alpha_t.into_tensor_arg(),
            self.n_crossings as u32,
            iso,
        );
    }

    /// Reads back the bracket state and returns each crossing's bracket
    /// lerped by its endpoint sdf magnitudes (one regula-falsi step past
    /// the bisection). Recovers small cracks that plain midpoints leave
    /// where neighbouring crossings land inconsistently.
    pub async fn finish(self) -> Vec<Vec3> {
        let lp = read_back_f32(self.left_pos_t).await;
        let rp = read_back_f32(self.right_pos_t).await;
        let ls = read_back_f32(self.left_sdf_t).await;
        let rs = read_back_f32(self.right_sdf_t).await;
        (0..self.n_crossings)
            .map(|i| {
                let l = Vec3::new(lp[3 * i], lp[3 * i + 1], lp[3 * i + 2]);
                let r = Vec3::new(rp[3 * i], rp[3 * i + 1], rp[3 * i + 2]);
                let (a, b) = (ls[i].abs(), rs[i].abs());
                let w = if a + b > 1e-12 { a / (a + b) } else { 0.5 };
                l.lerp(r, w)
            })
            .collect()
    }
}

/// Drain a `f32` tensor from the GPU into host memory.
pub(crate) async fn read_back_f32(t: FloatTensor<B>) -> Vec<f32> {
    use burn::backend::ops::{TransactionOps, TransactionPrimitive};
    let tp = TransactionPrimitive::<B>::new(vec![t], vec![], vec![], vec![]);
    let data = <B as TransactionOps<B>>::tr_execute(tp)
        .await
        .expect("read float tensor");
    data.read_floats[0]
        .clone()
        .into_vec::<f32>()
        .expect("float vec")
}
