//! GPU-resident binary-search bracket state.
//!
//! All per-crossing tensors (`left_pos`, `right_pos`, `left_sdf`,
//! `right_sdf`, the double-buffered active id lists) live on the GPU
//! from construction until [`RefineState::finish`]. Each BS step is a
//! small chain of kernel launches:
//!
//! 1. [`bisect::compute_midpoints_kernel`] — gathers active left/right
//!    endpoints and writes midpoints into a contiguous `[n_active, 3]`
//!    tensor.
//! 2. The 292 [`integrate_alpha_kernel`] launches (one per view) fold
//!    into a `min_alpha` running aggregator over the midpoint tensor.
//! 3. [`bisect::bracket_update_kernel`] reads the aggregator back,
//!    decides convergence per crossing, updates left/right in place,
//!    and atomically appends survivors to the next-step active list.
//!
//! The only per-step CPU↔GPU traffic is a 4-byte readback of the new
//! active count to size the next dispatch. Everything else stays
//! on-device until `finish` reads back the averaged left+right
//! positions.

use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor, create_tensor_from_slice};
use brush_render::kernels;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, TensorData};
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use glam::Vec3;

use crate::marching_tet::Crossing;

pub const N_STEPS: usize = 8;

/// |sdf| < this and the crossing is locked at its current midpoint.
/// Empirically tuned: 5e-2 catches ~98% of crossings by step 4 on
/// bonsai-scale runs with no PSNR loss.
pub const CONVERGE_EPS: f32 = 5e-2;

type B = MainBackendBase;

pub struct RefineState {
    device: WgpuDevice,

    left_pos_t: FloatTensor<B>,
    right_pos_t: FloatTensor<B>,
    left_sdf_t: FloatTensor<B>,
    right_sdf_t: FloatTensor<B>,
    /// Scratch for the per-step midpoint output; only `[0..n_active]`
    /// is meaningful after [`compute_midpoints_t`].
    mid_pos_t: FloatTensor<B>,

    /// Double-buffered active-id lists. `a_is_input` says which side
    /// holds the current step's input; the other receives the next
    /// step's compacted survivors.
    active_a_t: brush_cube::CubeTensor<WgpuRuntime>,
    active_b_t: brush_cube::CubeTensor<WgpuRuntime>,
    a_is_input: bool,

    n_crossings: usize,
    n_active: usize,
}

impl RefineState {
    pub fn new(
        crossings: &[Crossing],
        vertex_positions: &[Vec3],
        vertex_sdf: &[f32],
        device: WgpuDevice,
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

        let left_pos_t = B::float_from_data(TensorData::new(lp, [n, 3]), &device);
        let right_pos_t = B::float_from_data(TensorData::new(rp, [n, 3]), &device);
        let left_sdf_t = B::float_from_data(TensorData::new(ls, [n]), &device);
        let right_sdf_t = B::float_from_data(TensorData::new(rs, [n]), &device);
        let mid_pos_t = B::float_from_data(TensorData::new(vec![0.0f32; n * 3], [n, 3]), &device);

        let init_active: Vec<u32> = (0..n as u32).collect();
        let active_a_t = create_tensor_from_slice(&init_active, &device, DType::U32);
        let active_b_t = create_tensor::<1>([n], &device, DType::U32);

        Self {
            device,
            left_pos_t,
            right_pos_t,
            left_sdf_t,
            right_sdf_t,
            mid_pos_t,
            active_a_t,
            active_b_t,
            a_is_input: true,
            n_crossings: n,
            n_active: n,
        }
    }

    pub fn n_active(&self) -> usize {
        self.n_active
    }

    pub fn n_crossings(&self) -> usize {
        self.n_crossings
    }

    /// Launches the gather-and-average kernel. Returns a tensor handle
    /// suitable as the `points` input to `integrate_alpha`. Only the
    /// first `n_active` rows are meaningful.
    pub fn compute_midpoints_t(&self) -> FloatTensor<B> {
        if self.n_active == 0 {
            return self.mid_pos_t.clone();
        }
        let client = self.active_a_t.client.clone();
        let active_in = if self.a_is_input {
            &self.active_a_t
        } else {
            &self.active_b_t
        };
        kernels::bisect::compute_midpoints_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(self.n_active as u32, kernels::bisect::WG_SIZE),
            CubeDim::new_1d(kernels::bisect::WG_SIZE),
            self.left_pos_t.clone().into_tensor_arg(),
            self.right_pos_t.clone().into_tensor_arg(),
            active_in.clone().into_tensor_arg(),
            self.mid_pos_t.clone().into_tensor_arg(),
            self.n_active as u32,
        );
        self.mid_pos_t.clone()
    }

    /// Updates the bracket using a freshly-integrated `min_alpha_t`
    /// (length `n_active`, written by the integrate kernels). Reads back
    /// the new active count and swaps the active buffers.
    pub async fn update_bracket_t(&mut self, min_alpha_t: FloatTensor<B>, iso: f32) {
        if self.n_active == 0 {
            return;
        }
        let client = self.active_a_t.client.clone();
        // Fresh atomic counter for this step.
        let active_count_t = create_tensor_from_slice(&[0u32], &self.device, DType::U32);

        let (active_in, active_out) = if self.a_is_input {
            (&self.active_a_t, &self.active_b_t)
        } else {
            (&self.active_b_t, &self.active_a_t)
        };

        kernels::bisect::bracket_update_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(self.n_active as u32, kernels::bisect::WG_SIZE),
            CubeDim::new_1d(kernels::bisect::WG_SIZE),
            self.left_pos_t.clone().into_tensor_arg(),
            self.right_pos_t.clone().into_tensor_arg(),
            self.left_sdf_t.clone().into_tensor_arg(),
            self.right_sdf_t.clone().into_tensor_arg(),
            active_in.clone().into_tensor_arg(),
            self.mid_pos_t.clone().into_tensor_arg(),
            min_alpha_t.into_tensor_arg(),
            active_out.clone().into_tensor_arg(),
            active_count_t.clone().into_tensor_arg(),
            self.n_active as u32,
            iso,
            CONVERGE_EPS,
        );

        // 4-byte readback of the new active count.
        let handles = vec![active_count_t.handle.clone()];
        let read = client.read_async(handles).await.expect("read active_count");
        let bytes = &read[0];
        let new_n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        self.n_active = new_n;
        self.a_is_input = !self.a_is_input;
    }

    /// Reads back the bracket state and returns the per-crossing
    /// average of `left_pos` and `right_pos` (converged crossings have
    /// `left == right == midpoint`).
    pub async fn finish(self) -> Vec<Vec3> {
        let lp = read_back_f32(self.left_pos_t).await;
        let rp = read_back_f32(self.right_pos_t).await;
        let n = self.n_crossings;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let lx = lp[3 * i];
            let ly = lp[3 * i + 1];
            let lz = lp[3 * i + 2];
            let rx = rp[3 * i];
            let ry = rp[3 * i + 1];
            let rz = rp[3 * i + 2];
            out.push(Vec3::new((lx + rx) * 0.5, (ly + ry) * 0.5, (lz + rz) * 0.5));
        }
        out
    }
}

async fn read_back_f32(t: FloatTensor<B>) -> Vec<f32> {
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
