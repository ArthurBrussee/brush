//! Fused SSIM forward + backward in [`cubecl`].
//!
//! Inspired by <https://github.com/rahul-goel/fused-ssim>. Two separable
//! 11×11 Gaussian blurs plus the full SSIM formula in one shared-memory
//! pass — replaces 5 `conv2d` calls + ~20 elementwise ops + their
//! autodiff backward chain (~30 ms at 3K res) with a fused
//! forward+backward pair (~5 ms).
//!
//! Layout: input is `[C, H, W]` contiguous (one batch). Each workgroup is
//! 16×16 threads handling one 16×16 output tile of one channel. The
//! workgroup grid is `(tiles_x, tiles_y, C)`.
//!
//! Forward stores 3 partial-derivative tensors for backward
//! (`dm_dmu1`, `dm_dsigma1_sq`, `dm_dsigma12`) — paying ~3 image-sized
//! writes to skip recomputing the blurs in backward. Backward consumes
//! those + `dL/d(map)` and produces `dL/d(img1)` only; `img2` is treated
//! as a constant (matches the reference, which is fine because brush
//! always calls `ssim(pred, gt)`).

use brush_render::MainBackendBase;
use burn::{
    backend::{
        Autodiff,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        wgpu::WgpuRuntime,
    },
    prelude::Backend,
    tensor::{DType, Shape, Tensor, TensorMetadata, TensorPrimitive, ops::FloatTensor},
};
use burn_cubecl::{
    CubeRuntime, fusion::FusionCubeRuntime, kernel::into_contiguous, tensor::CubeTensor,
};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

mod kernels {
    // Bring `cubecl` into scope under that exact name so `#[cube]`
    // resolves — brush-train doesn't depend on cubecl directly, but
    // burn-cubecl re-exports it.
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::cube;
    use burn_cubecl::cubecl::frontend::CompilationArg;
    use burn_cubecl::cubecl::frontend::CubeIndexMutExpand;
    use burn_cubecl::cubecl::prelude::*;

    // 11-tap Gaussian, sigma = 1.5. Bit-equivalent at f32 to the Brush
    // Ssim convolutional reference (and the fused-ssim CUDA reference).
    const GAUSS: [f32; 11] = [
        0.001_028_380_1,
        0.007_598_758,
        0.036_000_773,
        0.109_360_69,
        0.213_005_53,
        0.266_011_72,
        0.213_005_53,
        0.109_360_69,
        0.036_000_773,
        0.007_598_758,
        0.001_028_380_1,
    ];

    pub const BLOCK_X: u32 = 16;
    pub const BLOCK_Y: u32 = 16;
    const HALO: u32 = 5;
    const SHARED_X: u32 = BLOCK_X + 2 * HALO; // 26
    const SHARED_Y: u32 = BLOCK_Y + 2 * HALO; // 26
    const CONV_Y: u32 = SHARED_Y;

    const C1: f32 = 0.01 * 0.01;
    const C2: f32 = 0.03 * 0.03;

    /// Read `img[c, y, x]` with zero-padding for out-of-bounds.
    /// Bounds-checked positions encoded as u32 with sentinel `u32::MAX` for OOB.
    #[cube]
    fn read_pix<F: Float>(img: &Tensor<F>, c: u32, y: u32, x: u32, oob: bool, h: u32, w: u32) -> F {
        if oob {
            F::cast_from(0.0_f32)
        } else {
            let idx = (c * h * w + y * w + x) as usize;
            img[idx]
        }
    }

    /// Compute `(clamped y, clamped x, oob_flag)` for a tile-relative position.
    /// `local_*` are in `0..SHARED_*` (26). Subtracting HALO=5 from
    /// `tile_y0+local_y` can underflow at the image's top-left edge — we
    /// detect that and saturate the address to 0 so the OOB sentinel is
    /// safe to index even for the load that won't actually use the value.
    #[cube]
    fn coords(
        tile_y0: u32,
        tile_x0: u32,
        local_y: u32,
        local_x: u32,
        h: u32,
        w: u32,
    ) -> (u32, u32, bool) {
        let total_y = tile_y0 + local_y;
        let total_x = tile_x0 + local_x;
        let oob_under = total_y < HALO || total_x < HALO;
        let zero = u32::cast_from(0u32);
        let gy = select(oob_under, zero, total_y - HALO);
        let gx = select(oob_under, zero, total_x - HALO);
        let oob = oob_under || gy >= h || gx >= w;
        (gy, gx, oob)
    }

    #[cube]
    fn gw<F: Float>(#[comptime] i: u32) -> F {
        F::new(comptime![GAUSS[i as usize]])
    }

    /// Forward kernel: writes SSIM map per pixel and (when training) the three
    /// partial derivative tensors used by the backward pass.
    // The cubek macro emits `x = x + y` patterns inside its expansion of the
    // launch glue; clippy can't see through to suppress them itself.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch_unchecked)]
    pub fn fused_ssim_forward_kernel<F: Float>(
        img1: &Tensor<F>,
        img2: &Tensor<F>,
        ssim_map: &mut Tensor<F>,
        dm_dmu1: &mut Tensor<F>,
        dm_dsigma1_sq: &mut Tensor<F>,
        dm_dsigma12: &mut Tensor<F>,
        h: u32,
        w: u32,
        #[comptime] train: bool,
    ) {
        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * BLOCK_Y;
        let tile_x0 = CUBE_POS_X * BLOCK_X;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;

        let mut s_tile = SharedMemory::<F>::new((SHARED_Y * SHARED_X * 2) as usize);
        let mut x_conv = SharedMemory::<F>::new((CONV_Y * BLOCK_X * 5) as usize);

        // ---- 1) Load tile + halo ----
        let thread_rank = UNIT_POS_Y * BLOCK_X + UNIT_POS_X;
        let threads = BLOCK_X * BLOCK_Y;
        let tile_size = SHARED_Y * SHARED_X;
        #[unroll]
        for s in 0u32..3u32 {
            let tid = s * threads + thread_rank;
            if tid < tile_size {
                let local_y = tid / SHARED_X;
                let local_x = tid % SHARED_X;
                let (gy, gx, oob) = coords(tile_y0, tile_x0, local_y, local_x, h, w);
                let xv = read_pix::<F>(img1, c, gy, gx, oob, h, w);
                let yv = read_pix::<F>(img2, c, gy, gx, oob, h, w);
                s_tile[((local_y * SHARED_X + local_x) * 2u32) as usize] = xv;
                s_tile[((local_y * SHARED_X + local_x) * 2u32 + 1u32) as usize] = yv;
            }
        }
        sync_cube();

        // ---- 2) Horizontal 11x1 ----
        {
            let lx = UNIT_POS_X + HALO;
            #[unroll]
            for pass in 0u32..2u32 {
                let ly = UNIT_POS_Y + pass * BLOCK_Y;
                if ly < CONV_Y {
                    let mut sum_x = F::cast_from(0.0_f32);
                    let mut sum_x2 = F::cast_from(0.0_f32);
                    let mut sum_y = F::cast_from(0.0_f32);
                    let mut sum_y2 = F::cast_from(0.0_f32);
                    let mut sum_xy = F::cast_from(0.0_f32);

                    #[unroll]
                    for d in 1u32..6u32 {
                        let w_l = gw::<F>(comptime![5u32 - d]);
                        let il = (ly * SHARED_X + (lx - d)) as usize;
                        let ir = (ly * SHARED_X + (lx + d)) as usize;
                        let xl = s_tile[il * 2];
                        let yl = s_tile[il * 2 + 1];
                        let xr = s_tile[ir * 2];
                        let yr = s_tile[ir * 2 + 1];
                        sum_x += (xl + xr) * w_l;
                        sum_x2 += (xl * xl + xr * xr) * w_l;
                        sum_y += (yl + yr) * w_l;
                        sum_y2 += (yl * yl + yr * yr) * w_l;
                        sum_xy += (xl * yl + xr * yr) * w_l;
                    }
                    let ic = (ly * SHARED_X + lx) as usize;
                    let xc = s_tile[ic * 2];
                    let yc = s_tile[ic * 2 + 1];
                    let wc = gw::<F>(5u32);
                    sum_x += xc * wc;
                    sum_x2 += xc * xc * wc;
                    sum_y += yc * wc;
                    sum_y2 += yc * yc * wc;
                    sum_xy += xc * yc * wc;

                    let base = ((ly * BLOCK_X + UNIT_POS_X) * 5) as usize;
                    x_conv[base] = sum_x;
                    x_conv[base + 1] = sum_x2;
                    x_conv[base + 2] = sum_y;
                    x_conv[base + 3] = sum_y2;
                    x_conv[base + 4] = sum_xy;
                }
            }
        }
        sync_cube();

        // ---- 3) Vertical 1x11 + final SSIM ----
        let ly = UNIT_POS_Y + HALO;
        let lx = UNIT_POS_X;
        let mut out0 = F::cast_from(0.0_f32);
        let mut out1 = F::cast_from(0.0_f32);
        let mut out2 = F::cast_from(0.0_f32);
        let mut out3 = F::cast_from(0.0_f32);
        let mut out4 = F::cast_from(0.0_f32);

        #[unroll]
        for d in 1u32..6u32 {
            let w_d = gw::<F>(comptime![5u32 - d]);
            let bt = (((ly - d) * BLOCK_X + lx) * 5) as usize;
            let bb = (((ly + d) * BLOCK_X + lx) * 5) as usize;
            out0 += (x_conv[bt] + x_conv[bb]) * w_d;
            out1 += (x_conv[bt + 1] + x_conv[bb + 1]) * w_d;
            out2 += (x_conv[bt + 2] + x_conv[bb + 2]) * w_d;
            out3 += (x_conv[bt + 3] + x_conv[bb + 3]) * w_d;
            out4 += (x_conv[bt + 4] + x_conv[bb + 4]) * w_d;
        }
        let bc = ((ly * BLOCK_X + lx) * 5) as usize;
        let wc = gw::<F>(5u32);
        out0 += x_conv[bc] * wc;
        out1 += x_conv[bc + 1] * wc;
        out2 += x_conv[bc + 2] * wc;
        out3 += x_conv[bc + 3] * wc;
        out4 += x_conv[bc + 4] * wc;

        if pix_x < w && pix_y < h {
            let mu1 = out0;
            let mu2 = out2;
            let mu1_sq = mu1 * mu1;
            let mu2_sq = mu2 * mu2;
            let sigma1_sq = F::max(F::cast_from(0.0_f32), out1 - mu1_sq);
            let sigma2_sq = F::max(F::cast_from(0.0_f32), out3 - mu2_sq);
            let sigma12 = out4 - mu1 * mu2;

            let c1_f = F::new(C1);
            let c2_f = F::new(C2);
            let a = mu1_sq + mu2_sq + c1_f;
            let b = sigma1_sq + sigma2_sq + c2_f;
            let c_top = F::cast_from(2.0_f32) * mu1 * mu2 + c1_f;
            let d_top = F::cast_from(2.0_f32) * sigma12 + c2_f;
            let raw = (c_top * d_top) / (a * b);
            let val = F::min(F::cast_from(1.0_f32), F::max(F::cast_from(-1.0_f32), raw));

            let global_idx = (c * h * w + pix_y * w + pix_x) as usize;
            ssim_map[global_idx] = val;

            if train {
                let clamped = raw < F::cast_from(-1.0_f32) || raw > F::cast_from(1.0_f32);
                let zero = F::cast_from(0.0_f32);
                let two = F::cast_from(2.0_f32);
                let inv_ab = F::cast_from(1.0_f32) / (a * b);
                let cd_div_aa_bb = (c_top * d_top) * inv_ab;
                let dmu1_raw = (mu2 * two * d_top) * inv_ab
                    - (mu2 * two * c_top) * inv_ab
                    - cd_div_aa_bb * (two * mu1 / a)
                    + cd_div_aa_bb * (two * mu1 / b);
                let dmu1_v = if clamped { zero } else { dmu1_raw };
                let dsigma1_v = if clamped {
                    zero
                } else {
                    F::cast_from(-1.0_f32) * c_top * d_top * inv_ab / b
                };
                let dsigma12_v = if clamped { zero } else { two * c_top * inv_ab };

                dm_dmu1[global_idx] = dmu1_v;
                dm_dsigma1_sq[global_idx] = dsigma1_v;
                dm_dsigma12[global_idx] = dsigma12_v;
            }
        }
    }

    /// Backward kernel: takes upstream gradient `dl_dmap` and the three saved
    /// partial derivatives, plus `img1` and `img2`, and writes `dl_dimg1`.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch_unchecked)]
    pub fn fused_ssim_backward_kernel<F: Float>(
        img1: &Tensor<F>,
        img2: &Tensor<F>,
        dl_dmap: &Tensor<F>,
        dm_dmu1: &Tensor<F>,
        dm_dsigma1_sq: &Tensor<F>,
        dm_dsigma12: &Tensor<F>,
        dl_dimg1: &mut Tensor<F>,
        h: u32,
        w: u32,
    ) {
        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * BLOCK_Y;
        let tile_x0 = CUBE_POS_X * BLOCK_X;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;

        let mut s_data = SharedMemory::<F>::new((SHARED_Y * SHARED_X * 3) as usize);
        let mut s_scratch = SharedMemory::<F>::new((CONV_Y * BLOCK_X * 3) as usize);

        // ---- 1) Load + fuse multiplication by chain ----
        let thread_rank = UNIT_POS_Y * BLOCK_X + UNIT_POS_X;
        let threads = BLOCK_X * BLOCK_Y;
        let tile_size = SHARED_Y * SHARED_X;
        #[unroll]
        for s in 0u32..3u32 {
            let tid = s * threads + thread_rank;
            if tid < tile_size {
                let local_y = tid / SHARED_X;
                let local_x = tid % SHARED_X;
                let (gy, gx, oob) = coords(tile_y0, tile_x0, local_y, local_x, h, w);
                let chain = read_pix::<F>(dl_dmap, c, gy, gx, oob, h, w);
                let v0 = read_pix::<F>(dm_dmu1, c, gy, gx, oob, h, w) * chain;
                let v1 = read_pix::<F>(dm_dsigma1_sq, c, gy, gx, oob, h, w) * chain;
                let v2 = read_pix::<F>(dm_dsigma12, c, gy, gx, oob, h, w) * chain;
                let base = ((local_y * SHARED_X + local_x) * 3) as usize;
                s_data[base] = v0;
                s_data[base + 1] = v1;
                s_data[base + 2] = v2;
            }
        }
        sync_cube();

        // ---- 2) Horizontal pass ----
        {
            let lx = UNIT_POS_X + HALO;
            #[unroll]
            for pass in 0u32..2u32 {
                let ly = UNIT_POS_Y + pass * BLOCK_Y;
                if ly < CONV_Y {
                    let mut a0 = F::cast_from(0.0_f32);
                    let mut a1 = F::cast_from(0.0_f32);
                    let mut a2 = F::cast_from(0.0_f32);
                    #[unroll]
                    for d in 1u32..6u32 {
                        let w_d = gw::<F>(comptime![5u32 - d]);
                        let il = ((ly * SHARED_X + (lx - d)) * 3) as usize;
                        let ir = ((ly * SHARED_X + (lx + d)) * 3) as usize;
                        a0 += (s_data[il] + s_data[ir]) * w_d;
                        a1 += (s_data[il + 1] + s_data[ir + 1]) * w_d;
                        a2 += (s_data[il + 2] + s_data[ir + 2]) * w_d;
                    }
                    let ic = ((ly * SHARED_X + lx) * 3) as usize;
                    let wc = gw::<F>(5u32);
                    a0 = a0 + s_data[ic] * wc;
                    a1 = a1 + s_data[ic + 1] * wc;
                    a2 = a2 + s_data[ic + 2] * wc;

                    let base = ((ly * BLOCK_X + UNIT_POS_X) * 3) as usize;
                    s_scratch[base] = a0;
                    s_scratch[base + 1] = a1;
                    s_scratch[base + 2] = a2;
                }
            }
        }
        sync_cube();

        // ---- 3) Vertical pass + finalize ----
        if pix_x < w && pix_y < h {
            let ly = UNIT_POS_Y + HALO;
            let lx = UNIT_POS_X;
            let mut s0 = F::cast_from(0.0_f32);
            let mut s1 = F::cast_from(0.0_f32);
            let mut s2 = F::cast_from(0.0_f32);

            #[unroll]
            for d in 1u32..6u32 {
                let w_d = gw::<F>(comptime![5u32 - d]);
                let bt = (((ly - d) * BLOCK_X + lx) * 3) as usize;
                let bb = (((ly + d) * BLOCK_X + lx) * 3) as usize;
                s0 += (s_scratch[bt] + s_scratch[bb]) * w_d;
                s1 += (s_scratch[bt + 1] + s_scratch[bb + 1]) * w_d;
                s2 += (s_scratch[bt + 2] + s_scratch[bb + 2]) * w_d;
            }
            let bc = ((ly * BLOCK_X + lx) * 3) as usize;
            let wc = gw::<F>(5u32);
            s0 += s_scratch[bc] * wc;
            s1 += s_scratch[bc + 1] * wc;
            s2 += s_scratch[bc + 2] * wc;

            let idx = (c * h * w + pix_y * w + pix_x) as usize;
            let p1 = img1[idx];
            let p2 = img2[idx];
            dl_dimg1[idx] = s0 + (F::cast_from(2.0_f32) * p1) * s1 + p2 * s2;
        }
    }
}

/// Output of the fused SSIM forward — the user-facing map plus three saved
/// tensors needed by the backward.
#[derive(Debug, Clone)]
pub struct FusedSsimForward<B: Backend> {
    pub map: FloatTensor<B>,
    pub dm_dmu1: FloatTensor<B>,
    pub dm_dsigma1_sq: FloatTensor<B>,
    pub dm_dsigma12: FloatTensor<B>,
}

/// Backend-level operations needed by the autodiff wrapper.
pub trait FusedSsimOps<B: Backend> {
    /// Forward. Inputs are `[C, H, W]`. Outputs all match shape. `train`
    /// controls whether the saved-for-backward tensors are filled.
    fn fused_ssim_forward(
        img1: FloatTensor<B>,
        img2: FloatTensor<B>,
        train: bool,
    ) -> FusedSsimForward<B>;

    /// Backward. Returns `dL/d(img1)`. `img2` has no gradient.
    #[allow(clippy::too_many_arguments)]
    fn fused_ssim_backward(
        img1: FloatTensor<B>,
        img2: FloatTensor<B>,
        dl_dmap: FloatTensor<B>,
        dm_dmu1: FloatTensor<B>,
        dm_dsigma1_sq: FloatTensor<B>,
        dm_dsigma12: FloatTensor<B>,
    ) -> FloatTensor<B>;
}

fn alloc_zeros<R: CubeRuntime>(template: &CubeTensor<R>) -> CubeTensor<R> {
    let dims = template.shape().as_slice().to_vec();
    burn_cubecl::ops::numeric::zeros_client::<R>(
        template.client.clone(),
        template.device.clone(),
        Shape::from(dims),
        template.dtype,
    )
}

fn launch_forward<R: CubeRuntime>(
    img1: CubeTensor<R>,
    img2: CubeTensor<R>,
    train: bool,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    use burn_cubecl::cubecl;
    use cubecl::prelude::{CubeCount, CubeDim};

    let img1 = into_contiguous(img1);
    let img2 = into_contiguous(img2);

    let dims = img1.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "fused_ssim expects [C, H, W] inputs");
    let (c_dim, h, w) = (dims[0], dims[1], dims[2]);

    let map = alloc_zeros(&img1);
    let dm_dmu1 = alloc_zeros(&img1);
    let dm_dsigma1_sq = alloc_zeros(&img1);
    let dm_dsigma12 = alloc_zeros(&img1);

    let cube_dim = CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y);
    let cube_count = CubeCount::Static(
        (w as u32).div_ceil(kernels::BLOCK_X),
        (h as u32).div_ceil(kernels::BLOCK_Y),
        c_dim as u32,
    );

    let client = img1.client.clone();
    // SAFETY: Kernel bounds-checks every read with `coords` + `read_pix` and
    // gates writes on `pix_x < w && pix_y < h`. All shared-memory indices
    // are computed from `UNIT_POS_*` (< BLOCK_*) and `pass`/`d` constants
    // bounded above by the buffer dimensions.
    unsafe {
        kernels::fused_ssim_forward_kernel::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            img1.into_tensor_arg(),
            img2.into_tensor_arg(),
            map.clone().into_tensor_arg(),
            dm_dmu1.clone().into_tensor_arg(),
            dm_dsigma1_sq.clone().into_tensor_arg(),
            dm_dsigma12.clone().into_tensor_arg(),
            h as u32,
            w as u32,
            train,
        );
    }

    (map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
}

#[allow(clippy::too_many_arguments)]
fn launch_backward<R: CubeRuntime>(
    img1: CubeTensor<R>,
    img2: CubeTensor<R>,
    dl_dmap: CubeTensor<R>,
    dm_dmu1: CubeTensor<R>,
    dm_dsigma1_sq: CubeTensor<R>,
    dm_dsigma12: CubeTensor<R>,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl;
    use cubecl::prelude::{CubeCount, CubeDim};

    let img1 = into_contiguous(img1);
    let img2 = into_contiguous(img2);
    let dl_dmap = into_contiguous(dl_dmap);
    let dm_dmu1 = into_contiguous(dm_dmu1);
    let dm_dsigma1_sq = into_contiguous(dm_dsigma1_sq);
    let dm_dsigma12 = into_contiguous(dm_dsigma12);

    let dims = img1.shape().as_slice().to_vec();
    let (c_dim, h, w) = (dims[0], dims[1], dims[2]);
    let dl_dimg1 = alloc_zeros(&img1);

    let cube_dim = CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y);
    let cube_count = CubeCount::Static(
        (w as u32).div_ceil(kernels::BLOCK_X),
        (h as u32).div_ceil(kernels::BLOCK_Y),
        c_dim as u32,
    );

    let client = img1.client.clone();
    // SAFETY: same boundary guarantees as the forward — `coords`/`read_pix`
    // for halo loads and `pix_x < w && pix_y < h` for the per-pixel write.
    unsafe {
        kernels::fused_ssim_backward_kernel::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            img1.into_tensor_arg(),
            img2.into_tensor_arg(),
            dl_dmap.into_tensor_arg(),
            dm_dmu1.into_tensor_arg(),
            dm_dsigma1_sq.into_tensor_arg(),
            dm_dsigma12.into_tensor_arg(),
            dl_dimg1.clone().into_tensor_arg(),
            h as u32,
            w as u32,
        );
    }

    dl_dimg1
}

// ---- MainBackendBase impl ----

impl FusedSsimOps<Self> for MainBackendBase {
    fn fused_ssim_forward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        train: bool,
    ) -> FusedSsimForward<Self> {
        let (map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12) = launch_forward(img1, img2, train);
        FusedSsimForward {
            map,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        }
    }

    fn fused_ssim_backward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        dm_dmu1: FloatTensor<Self>,
        dm_dsigma1_sq: FloatTensor<Self>,
        dm_dsigma12: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        launch_backward(img1, img2, dl_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
    }
}

// ---- Fusion<MainBackendBase> impl: dispatch via CustomOpIr ----

impl FusedSsimOps<Self> for Fusion<MainBackendBase> {
    fn fused_ssim_forward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        train: bool,
    ) -> FusedSsimForward<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            train: bool,
        }
        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();
                let [img1, img2] = inputs;
                let [map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12] = outputs;
                let out = <MainBackendBase as FusedSsimOps<MainBackendBase>>::fused_ssim_forward(
                    h.get_float_tensor::<MainBackendBase>(img1),
                    h.get_float_tensor::<MainBackendBase>(img2),
                    self.train,
                );
                h.register_float_tensor::<MainBackendBase>(&map.id, out.map);
                h.register_float_tensor::<MainBackendBase>(&dm_dmu1.id, out.dm_dmu1);
                h.register_float_tensor::<MainBackendBase>(&dm_dsigma1_sq.id, out.dm_dsigma1_sq);
                h.register_float_tensor::<MainBackendBase>(&dm_dsigma12.id, out.dm_dsigma12);
            }
        }

        let shape = img1.shape();
        let client = img1.client.clone();

        let mk_out = || TensorIr::uninit(client.create_empty_handle(), shape.clone(), DType::F32);
        let map_out = mk_out();
        let dm_dmu1_out = mk_out();
        let dm_dsigma1_sq_out = mk_out();
        let dm_dsigma12_out = mk_out();

        let inputs = [img1, img2];
        let stream = OperationStreams::with_inputs(&inputs);
        let desc = CustomOpIr::new(
            "fused_ssim_forward",
            &inputs.map(|t| t.into_ir()),
            &[map_out, dm_dmu1_out, dm_dsigma1_sq_out, dm_dsigma12_out],
        );
        let op = CustomOp {
            desc: desc.clone(),
            train,
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();
        let [map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12] = outputs;

        FusedSsimForward {
            map,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        }
    }

    fn fused_ssim_backward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        dm_dmu1: FloatTensor<Self>,
        dm_dsigma1_sq: FloatTensor<Self>,
        dm_dsigma12: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
        }
        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();
                let [img1, img2, dl_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12] = inputs;
                let [dl_dimg1] = outputs;
                let out = <MainBackendBase as FusedSsimOps<MainBackendBase>>::fused_ssim_backward(
                    h.get_float_tensor::<MainBackendBase>(img1),
                    h.get_float_tensor::<MainBackendBase>(img2),
                    h.get_float_tensor::<MainBackendBase>(dl_dmap),
                    h.get_float_tensor::<MainBackendBase>(dm_dmu1),
                    h.get_float_tensor::<MainBackendBase>(dm_dsigma1_sq),
                    h.get_float_tensor::<MainBackendBase>(dm_dsigma12),
                );
                h.register_float_tensor::<MainBackendBase>(&dl_dimg1.id, out);
            }
        }

        let shape = img1.shape();
        let client = img1.client.clone();

        let dl_dimg1_out = TensorIr::uninit(client.create_empty_handle(), shape, DType::F32);

        let inputs = [img1, img2, dl_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12];
        let stream = OperationStreams::with_inputs(&inputs);
        let desc = CustomOpIr::new(
            "fused_ssim_backward",
            &inputs.map(|t| t.into_ir()),
            &[dl_dimg1_out],
        );
        let op = CustomOp { desc: desc.clone() };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();
        let [dl_dimg1] = outputs;
        dl_dimg1
    }
}

// ---- Autodiff wrapper ----

#[derive(Debug)]
struct FusedSsimBackward;

#[derive(Debug, Clone)]
struct FusedSsimState<B: Backend> {
    img1: FloatTensor<B>,
    img2: FloatTensor<B>,
    dm_dmu1: FloatTensor<B>,
    dm_dsigma1_sq: FloatTensor<B>,
    dm_dsigma12: FloatTensor<B>,
}

impl<B: Backend + FusedSsimOps<B>> Backward<B, 1> for FusedSsimBackward {
    type State = FusedSsimState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let dl_dmap = grads.consume::<B>(&ops.node);

        let [img1_parent] = ops.parents;

        let dl_dimg1 = B::fused_ssim_backward(
            state.img1,
            state.img2,
            dl_dmap,
            state.dm_dmu1,
            state.dm_dsigma1_sq,
            state.dm_dsigma12,
        );

        if let Some(node) = img1_parent {
            grads.register::<B>(node.id, dl_dimg1);
        }
    }
}

/// Fused SSIM forward+backward in [`cubecl`]. Drop-in replacement for a
/// conv2d-based SSIM loss. `img1` is differentiable; `img2` is treated
/// as a constant (no gradient computed) — matches the reference and works
/// fine for brush since we always call `ssim(pred, gt)`.
///
/// Inputs are `[H, W, C]` to match the existing `Ssim::ssim` signature.
pub fn fused_ssim<B, C>(
    img1: Tensor<Autodiff<B, C>, 3>,
    img2: Tensor<Autodiff<B, C>, 3>,
) -> Tensor<Autodiff<B, C>, 3>
where
    B: Backend + FusedSsimOps<B>,
    C: CheckpointStrategy,
{
    // Permute [H, W, C] -> [C, H, W] (metadata-only; into_contiguous in the
    // launcher does the actual data movement).
    let img1_chw = img1.permute([2, 0, 1]);
    let img2_chw = img2.permute([2, 0, 1]);

    let img1_inner = img1_chw.into_primitive().tensor();
    let img2_inner = img2_chw.into_primitive().tensor();

    let prep = FusedSsimBackward
        .prepare::<C>([img1_inner.node.clone()])
        .compute_bound()
        .stateful();

    let train = matches!(prep, OpsKind::Tracked(_));
    let img1_p = img1_inner.primitive;
    let img2_p = img2_inner.primitive;
    let out = B::fused_ssim_forward(img1_p.clone(), img2_p.clone(), train);

    let map_p = match prep {
        OpsKind::Tracked(prep) => {
            let state = FusedSsimState {
                img1: img1_p,
                img2: img2_p,
                dm_dmu1: out.dm_dmu1,
                dm_dsigma1_sq: out.dm_dsigma1_sq,
                dm_dsigma12: out.dm_dsigma12,
            };
            prep.finish(state, out.map)
        }
        OpsKind::UnTracked(prep) => prep.finish(out.map),
    };

    let map: Tensor<Autodiff<B, C>, 3> = Tensor::from_primitive(TensorPrimitive::Float(map_p));
    // Permute back to [H, W, C].
    map.permute([1, 2, 0])
}

/// Forward-only fused SSIM for non-differentiable backends — used by eval.
///
/// Same kernel as [`fused_ssim`] but skips storing the saved-for-backward
/// partial-derivative tensors.
pub fn fused_ssim_eval<B>(img1: Tensor<B, 3>, img2: Tensor<B, 3>) -> Tensor<B, 3>
where
    B: Backend + FusedSsimOps<B>,
{
    let img1_chw = img1.permute([2, 0, 1]);
    let img2_chw = img2.permute([2, 0, 1]);

    let img1_p = img1_chw.into_primitive().tensor();
    let img2_p = img2_chw.into_primitive().tensor();
    let out = B::fused_ssim_forward(img1_p, img2_p, false);

    let map: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(out.map));
    map.permute([1, 2, 0])
}
