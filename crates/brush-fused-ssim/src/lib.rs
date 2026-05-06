//! Fused L1 + SSIM image loss in cubecl.
//!
//! Forward computes `l1_w * |img1 - img2| + ssim_w * ssim(img1, img2)` per
//! pixel. Backward recomputes the SSIM partials inline from img1/img2,
//! multiplies by the upstream gradient, then applies the second gaussian
//! blur to produce `dL/d(img1)`. No per-pixel partial-derivative tensors
//! survive across the autograd tape.

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
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::cube;
    use burn_cubecl::cubecl::frontend::CompilationArg;
    use burn_cubecl::cubecl::frontend::CubeIndexMutExpand;
    use burn_cubecl::cubecl::prelude::*;
    use half::f16;

    // 11-tap Gaussian, sigma = 1.5.
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
    // Backward needs partials at every tile + halo pixel; each partial is
    // itself a blur of img1/img2, so the bwd widens the loaded image tile
    // by an extra halo to feed that inner blur.
    const EXT_X: u32 = BLOCK_X + 4 * HALO; // 36
    const EXT_Y: u32 = BLOCK_Y + 4 * HALO; // 36

    const C1: f32 = 0.01 * 0.01;
    const C2: f32 = 0.03 * 0.03;

    /// Read `img[c, y, x]` returning zero for out-of-bounds.
    #[cube]
    fn read_pix<F: Float>(img: &Tensor<F>, c: u32, y: u32, x: u32, oob: bool, h: u32, w: u32) -> F {
        if oob {
            F::cast_from(0.0_f32)
        } else {
            img[(c * h * w + y * w + x) as usize]
        }
    }

    /// Map a tile-local position offset by `halo` to global image coords.
    /// Returns clamped (gy, gx) and an OOB flag the caller passes to `read_pix`.
    #[cube]
    fn coords(
        tile_y0: u32,
        tile_x0: u32,
        local_y: u32,
        local_x: u32,
        #[comptime] halo: u32,
        h: u32,
        w: u32,
    ) -> (u32, u32, bool) {
        let total_y = tile_y0 + local_y;
        let total_x = tile_x0 + local_x;
        let oob_under = total_y < halo || total_x < halo;
        let zero = u32::cast_from(0u32);
        let gy = select(oob_under, zero, total_y - halo);
        let gx = select(oob_under, zero, total_x - halo);
        (gy, gx, oob_under || gy >= h || gx >= w)
    }

    #[cube]
    fn gw<F: Float>(#[comptime] i: u32) -> F {
        F::new(comptime![GAUSS[i as usize]])
    }

    /// Forward: produce the L1 + SSIM loss map. Two separable 11-tap blurs
    /// over img1/img2/img1²/img2²/img1·img2, then SSIM + L1 finalize.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch_unchecked)]
    pub fn fused_ssim_forward_kernel<F: Float>(
        img1: &Tensor<F>,
        img2: &Tensor<F>,
        loss_map: &mut Tensor<F>,
        h: u32,
        w: u32,
        l1_weight: f32,
        ssim_weight: f32,
    ) {
        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * BLOCK_Y;
        let tile_x0 = CUBE_POS_X * BLOCK_X;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;

        let mut s_tile = SharedMemory::<F>::new((SHARED_Y * SHARED_X * 2) as usize);
        let mut x_conv = SharedMemory::<F>::new((SHARED_Y * BLOCK_X * 5) as usize);

        let thread_rank = UNIT_POS_Y * BLOCK_X + UNIT_POS_X;
        let threads = BLOCK_X * BLOCK_Y;
        let tile_size = SHARED_Y * SHARED_X;
        #[unroll]
        for s in 0u32..3u32 {
            let tid = s * threads + thread_rank;
            if tid < tile_size {
                let local_y = tid / SHARED_X;
                let local_x = tid % SHARED_X;
                let (gy, gx, oob) = coords(tile_y0, tile_x0, local_y, local_x, HALO, h, w);
                let base = ((local_y * SHARED_X + local_x) * 2u32) as usize;
                s_tile[base] = read_pix::<F>(img1, c, gy, gx, oob, h, w);
                s_tile[base + 1] = read_pix::<F>(img2, c, gy, gx, oob, h, w);
            }
        }
        sync_cube();

        // Horizontal 11-tap blur over s_tile -> 5 sums per pixel in x_conv.
        let lx = UNIT_POS_X + HALO;
        #[unroll]
        for pass in 0u32..2u32 {
            let ly = UNIT_POS_Y + pass * BLOCK_Y;
            if ly < SHARED_Y {
                let mut sum_x = F::cast_from(0.0_f32);
                let mut sum_x2 = F::cast_from(0.0_f32);
                let mut sum_y = F::cast_from(0.0_f32);
                let mut sum_y2 = F::cast_from(0.0_f32);
                let mut sum_xy = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let il = (ly * SHARED_X + (lx - d)) as usize;
                    let ir = (ly * SHARED_X + (lx + d)) as usize;
                    let xl = s_tile[il * 2];
                    let yl = s_tile[il * 2 + 1];
                    let xr = s_tile[ir * 2];
                    let yr = s_tile[ir * 2 + 1];
                    sum_x += (xl + xr) * w_d;
                    sum_x2 += (xl * xl + xr * xr) * w_d;
                    sum_y += (yl + yr) * w_d;
                    sum_y2 += (yl * yl + yr * yr) * w_d;
                    sum_xy += (xl * yl + xr * yr) * w_d;
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
        sync_cube();

        // Vertical 11-tap blur, then derive SSIM and emit L1 + SSIM loss.
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
            let zero = F::cast_from(0.0_f32);
            let two = F::cast_from(2.0_f32);
            let mu1 = out0;
            let mu2 = out2;
            let mu1_sq = mu1 * mu1;
            let mu2_sq = mu2 * mu2;
            let sigma1_sq = F::max(zero, out1 - mu1_sq);
            let sigma2_sq = F::max(zero, out3 - mu2_sq);
            let sigma12 = out4 - mu1 * mu2;
            let a = mu1_sq + mu2_sq + F::new(C1);
            let b = sigma1_sq + sigma2_sq + F::new(C2);
            let c_top = two * mu1 * mu2 + F::new(C1);
            let d_top = two * sigma12 + F::new(C2);
            let raw = (c_top * d_top) / (a * b);
            let val = clamp(raw, F::cast_from(-1.0_f32), F::cast_from(1.0_f32));

            // Center-pixel L1 reads raw img values (not blurred mu1, mu2)
            // straight out of s_tile.
            let center = ((UNIT_POS_Y + HALO) * SHARED_X + (UNIT_POS_X + HALO)) as usize;
            let l1 = F::abs(s_tile[center * 2] - s_tile[center * 2 + 1]);

            let idx = (c * h * w + pix_y * w + pix_x) as usize;
            loss_map[idx] = F::cast_from(l1_weight) * l1 + F::cast_from(ssim_weight) * val;
        }
    }

    /// Backward: recompute SSIM partials inline, no saved-for-backward tensors.
    /// Phases:
    ///   1. Load img1/img2 with halo of 2*HALO into `s_imgs`.
    ///   2. Horizontal blur into `s_horiz`.
    ///   3. Vertical blur to recover (mu1, mu2, sigmas), derive SSIM partials,
    ///      multiply by upstream chain, store in `s_chain_partials`.
    ///   4. Second blur (horizontal then vertical) over `s_chain_partials`,
    ///      add the L1 sign term, write `dl_dimg1`.
    /// `s_imgs` and `s_horiz` use f16 storage to keep workgroup memory
    /// under ~28 KiB. All math runs in f32 registers.
    #[allow(clippy::assign_op_pattern)]
    #[cube(launch_unchecked)]
    pub fn fused_ssim_backward_kernel<F: Float>(
        img1: &Tensor<F>,
        img2: &Tensor<F>,
        dl_dmap: &Tensor<F>,
        dl_dimg1: &mut Tensor<F>,
        h: u32,
        w: u32,
        l1_weight: f32,
        ssim_weight: f32,
    ) {
        let c = CUBE_POS_Z;
        let tile_y0 = CUBE_POS_Y * BLOCK_Y;
        let tile_x0 = CUBE_POS_X * BLOCK_X;
        let pix_y = tile_y0 + UNIT_POS_Y;
        let pix_x = tile_x0 + UNIT_POS_X;

        let mut s_imgs = SharedMemory::<f16>::new((EXT_Y * EXT_X * 2) as usize);
        let mut s_horiz = SharedMemory::<f16>::new((EXT_Y * SHARED_X * 5) as usize);
        let mut s_chain_partials = SharedMemory::<F>::new((SHARED_Y * SHARED_X * 3) as usize);
        let mut s_bwd_horiz = SharedMemory::<F>::new((SHARED_Y * BLOCK_X * 3) as usize);

        let thread_rank = UNIT_POS_Y * BLOCK_X + UNIT_POS_X;
        let threads = BLOCK_X * BLOCK_Y;

        // Load img1/img2 with halo of 2*HALO.
        let ext_size = EXT_Y * EXT_X;
        #[unroll]
        for s in 0u32..6u32 {
            let tid = s * threads + thread_rank;
            if tid < ext_size {
                let local_y = tid / EXT_X;
                let local_x = tid % EXT_X;
                let (gy, gx, oob) = coords(tile_y0, tile_x0, local_y, local_x, 2u32 * HALO, h, w);
                let base = ((local_y * EXT_X + local_x) * 2u32) as usize;
                s_imgs[base] = f16::cast_from(read_pix::<F>(img1, c, gy, gx, oob, h, w));
                s_imgs[base + 1] = f16::cast_from(read_pix::<F>(img2, c, gy, gx, oob, h, w));
            }
        }
        sync_cube();

        // Horizontal blur over the extended tile. s_horiz[row, col] is
        // centered at s_imgs col (col + HALO), so col in 0..SHARED_X covers
        // the horizontal extent the next vertical conv needs.
        let horiz_size = EXT_Y * SHARED_X;
        #[unroll]
        for s in 0u32..4u32 {
            let tid = s * threads + thread_rank;
            if tid < horiz_size {
                let row_y = tid / SHARED_X;
                let col_x = tid % SHARED_X;
                let center = col_x + HALO;
                let mut sum_x = F::cast_from(0.0_f32);
                let mut sum_x2 = F::cast_from(0.0_f32);
                let mut sum_y = F::cast_from(0.0_f32);
                let mut sum_y2 = F::cast_from(0.0_f32);
                let mut sum_xy = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let il = ((row_y * EXT_X + (center - d)) * 2u32) as usize;
                    let ir = ((row_y * EXT_X + (center + d)) * 2u32) as usize;
                    let xl = F::cast_from(s_imgs[il]);
                    let yl = F::cast_from(s_imgs[il + 1]);
                    let xr = F::cast_from(s_imgs[ir]);
                    let yr = F::cast_from(s_imgs[ir + 1]);
                    sum_x += (xl + xr) * w_d;
                    sum_x2 += (xl * xl + xr * xr) * w_d;
                    sum_y += (yl + yr) * w_d;
                    sum_y2 += (yl * yl + yr * yr) * w_d;
                    sum_xy += (xl * yl + xr * yr) * w_d;
                }
                let ic = ((row_y * EXT_X + center) * 2u32) as usize;
                let xc = F::cast_from(s_imgs[ic]);
                let yc = F::cast_from(s_imgs[ic + 1]);
                let wc = gw::<F>(5u32);
                sum_x += xc * wc;
                sum_x2 += xc * xc * wc;
                sum_y += yc * wc;
                sum_y2 += yc * yc * wc;
                sum_xy += xc * yc * wc;
                let base = ((row_y * SHARED_X + col_x) * 5u32) as usize;
                s_horiz[base] = f16::cast_from(sum_x);
                s_horiz[base + 1] = f16::cast_from(sum_x2);
                s_horiz[base + 2] = f16::cast_from(sum_y);
                s_horiz[base + 3] = f16::cast_from(sum_y2);
                s_horiz[base + 4] = f16::cast_from(sum_xy);
            }
        }
        sync_cube();

        // Vertical blur, derive SSIM partials, multiply by chain.
        let partial_size = SHARED_Y * SHARED_X;
        #[unroll]
        for s in 0u32..3u32 {
            let tid = s * threads + thread_rank;
            if tid < partial_size {
                let part_y = tid / SHARED_X;
                let part_x = tid % SHARED_X;
                let center = part_y + HALO;

                let mut out0 = F::cast_from(0.0_f32);
                let mut out1 = F::cast_from(0.0_f32);
                let mut out2 = F::cast_from(0.0_f32);
                let mut out3 = F::cast_from(0.0_f32);
                let mut out4 = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let bt = (((center - d) * SHARED_X + part_x) * 5u32) as usize;
                    let bb = (((center + d) * SHARED_X + part_x) * 5u32) as usize;
                    out0 += (F::cast_from(s_horiz[bt]) + F::cast_from(s_horiz[bb])) * w_d;
                    out1 += (F::cast_from(s_horiz[bt + 1]) + F::cast_from(s_horiz[bb + 1])) * w_d;
                    out2 += (F::cast_from(s_horiz[bt + 2]) + F::cast_from(s_horiz[bb + 2])) * w_d;
                    out3 += (F::cast_from(s_horiz[bt + 3]) + F::cast_from(s_horiz[bb + 3])) * w_d;
                    out4 += (F::cast_from(s_horiz[bt + 4]) + F::cast_from(s_horiz[bb + 4])) * w_d;
                }
                let bc = ((center * SHARED_X + part_x) * 5u32) as usize;
                let wc = gw::<F>(5u32);
                out0 += F::cast_from(s_horiz[bc]) * wc;
                out1 += F::cast_from(s_horiz[bc + 1]) * wc;
                out2 += F::cast_from(s_horiz[bc + 2]) * wc;
                out3 += F::cast_from(s_horiz[bc + 3]) * wc;
                out4 += F::cast_from(s_horiz[bc + 4]) * wc;

                let zero = F::cast_from(0.0_f32);
                let two = F::cast_from(2.0_f32);
                let mu1 = out0;
                let mu2 = out2;
                let mu1_sq = mu1 * mu1;
                let mu2_sq = mu2 * mu2;
                let sigma1_sq = F::max(zero, out1 - mu1_sq);
                let sigma2_sq = F::max(zero, out3 - mu2_sq);
                let sigma12 = out4 - mu1 * mu2;
                let a = mu1_sq + mu2_sq + F::new(C1);
                let b = sigma1_sq + sigma2_sq + F::new(C2);
                let c_top = two * mu1 * mu2 + F::new(C1);
                let d_top = two * sigma12 + F::new(C2);
                let inv_ab = F::cast_from(1.0_f32) / (a * b);
                let cd = c_top * d_top * inv_ab;
                let raw = cd;
                let clamped = raw < F::cast_from(-1.0_f32) || raw > F::cast_from(1.0_f32);

                // d/dmu1 SSIM. Factored from the explicit form to expose the
                // (d_top - c_top) and (1/a - 1/b) cancellations.
                let dmu1 = if clamped {
                    zero
                } else {
                    two * mu2 * inv_ab * (d_top - c_top)
                        - two * mu1 * cd * (F::cast_from(1.0_f32) / a - F::cast_from(1.0_f32) / b)
                };
                let dsigma1 = if clamped { zero } else { -cd / b };
                let dsigma12 = if clamped { zero } else { two * c_top * inv_ab };

                let (gy, gx, oob) = coords(tile_y0, tile_x0, part_y, part_x, HALO, h, w);
                let chain = read_pix::<F>(dl_dmap, c, gy, gx, oob, h, w);

                let base = ((part_y * SHARED_X + part_x) * 3u32) as usize;
                s_chain_partials[base] = dmu1 * chain;
                s_chain_partials[base + 1] = dsigma1 * chain;
                s_chain_partials[base + 2] = dsigma12 * chain;
            }
        }
        sync_cube();

        // Second horizontal blur over chain * partials.
        let lx_b = UNIT_POS_X + HALO;
        #[unroll]
        for pass in 0u32..2u32 {
            let ly_b = UNIT_POS_Y + pass * BLOCK_Y;
            if ly_b < SHARED_Y {
                let mut a0 = F::cast_from(0.0_f32);
                let mut a1 = F::cast_from(0.0_f32);
                let mut a2 = F::cast_from(0.0_f32);
                #[unroll]
                for d in 1u32..6u32 {
                    let w_d = gw::<F>(comptime![5u32 - d]);
                    let il = ((ly_b * SHARED_X + (lx_b - d)) * 3u32) as usize;
                    let ir = ((ly_b * SHARED_X + (lx_b + d)) * 3u32) as usize;
                    a0 += (s_chain_partials[il] + s_chain_partials[ir]) * w_d;
                    a1 += (s_chain_partials[il + 1] + s_chain_partials[ir + 1]) * w_d;
                    a2 += (s_chain_partials[il + 2] + s_chain_partials[ir + 2]) * w_d;
                }
                let ic = ((ly_b * SHARED_X + lx_b) * 3u32) as usize;
                let wc = gw::<F>(5u32);
                a0 += s_chain_partials[ic] * wc;
                a1 += s_chain_partials[ic + 1] * wc;
                a2 += s_chain_partials[ic + 2] * wc;
                let base = ((ly_b * BLOCK_X + UNIT_POS_X) * 3u32) as usize;
                s_bwd_horiz[base] = a0;
                s_bwd_horiz[base + 1] = a1;
                s_bwd_horiz[base + 2] = a2;
            }
        }
        sync_cube();

        // Second vertical blur + L1 sign + write.
        if pix_x < w && pix_y < h {
            let ly = UNIT_POS_Y + HALO;
            let lx = UNIT_POS_X;
            let mut s0 = F::cast_from(0.0_f32);
            let mut s1 = F::cast_from(0.0_f32);
            let mut s2 = F::cast_from(0.0_f32);
            #[unroll]
            for d in 1u32..6u32 {
                let w_d = gw::<F>(comptime![5u32 - d]);
                let bt = (((ly - d) * BLOCK_X + lx) * 3u32) as usize;
                let bb = (((ly + d) * BLOCK_X + lx) * 3u32) as usize;
                s0 += (s_bwd_horiz[bt] + s_bwd_horiz[bb]) * w_d;
                s1 += (s_bwd_horiz[bt + 1] + s_bwd_horiz[bb + 1]) * w_d;
                s2 += (s_bwd_horiz[bt + 2] + s_bwd_horiz[bb + 2]) * w_d;
            }
            let bc = ((ly * BLOCK_X + lx) * 3u32) as usize;
            let wc = gw::<F>(5u32);
            s0 += s_bwd_horiz[bc] * wc;
            s1 += s_bwd_horiz[bc + 1] * wc;
            s2 += s_bwd_horiz[bc + 2] * wc;

            let idx = (c * h * w + pix_y * w + pix_x) as usize;
            let p1 = img1[idx];
            let p2 = img2[idx];
            let ssim_grad = s0 + (F::cast_from(2.0_f32) * p1) * s1 + p2 * s2;
            let diff = p1 - p2;
            let zero = F::cast_from(0.0_f32);
            let l1_sign = if diff > zero {
                F::cast_from(1.0_f32)
            } else if diff < zero {
                F::cast_from(-1.0_f32)
            } else {
                zero
            };
            dl_dimg1[idx] = F::cast_from(ssim_weight) * ssim_grad
                + F::cast_from(l1_weight) * l1_sign * dl_dmap[idx];
        }
    }
}

/// Backend hooks for the fused SSIM kernels.
pub trait FusedSsimOps<B: Backend> {
    fn fused_ssim_forward(
        img1: FloatTensor<B>,
        img2: FloatTensor<B>,
        l1_weight: f32,
        ssim_weight: f32,
    ) -> FloatTensor<B>;

    fn fused_ssim_backward(
        img1: FloatTensor<B>,
        img2: FloatTensor<B>,
        dl_dmap: FloatTensor<B>,
        l1_weight: f32,
        ssim_weight: f32,
    ) -> FloatTensor<B>;
}

fn alloc_zeros<R: CubeRuntime>(template: &CubeTensor<R>) -> CubeTensor<R> {
    burn_cubecl::ops::numeric::zeros_client::<R>(
        template.client.clone(),
        template.device.clone(),
        Shape::from(template.shape().as_slice().to_vec()),
        template.dtype,
    )
}

fn cube_count(c: u32, h: u32, w: u32) -> burn_cubecl::cubecl::prelude::CubeCount {
    use burn_cubecl::cubecl::prelude::CubeCount;
    CubeCount::Static(
        w.div_ceil(kernels::BLOCK_X),
        h.div_ceil(kernels::BLOCK_Y),
        c,
    )
}

fn launch_forward<R: CubeRuntime>(
    img1: CubeTensor<R>,
    img2: CubeTensor<R>,
    l1_weight: f32,
    ssim_weight: f32,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::CubeDim;

    let img1 = into_contiguous(img1);
    let img2 = into_contiguous(img2);
    let dims = img1.shape().as_slice().to_vec();
    assert_eq!(dims.len(), 3, "fused_ssim expects [C, H, W] inputs");
    let (c, h, w) = (dims[0] as u32, dims[1] as u32, dims[2] as u32);
    let map = alloc_zeros(&img1);
    let client = img1.client.clone();
    // SAFETY: every read goes through `coords` + `read_pix` (zero-padded for
    // OOB) and writes are gated on `pix_x < w && pix_y < h`.
    unsafe {
        kernels::fused_ssim_forward_kernel::launch_unchecked::<f32, R>(
            &client,
            cube_count(c, h, w),
            CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
            img1.into_tensor_arg(),
            img2.into_tensor_arg(),
            map.clone().into_tensor_arg(),
            h,
            w,
            l1_weight,
            ssim_weight,
        );
    }
    map
}

fn launch_backward<R: CubeRuntime>(
    img1: CubeTensor<R>,
    img2: CubeTensor<R>,
    dl_dmap: CubeTensor<R>,
    l1_weight: f32,
    ssim_weight: f32,
) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::CubeDim;

    let img1 = into_contiguous(img1);
    let img2 = into_contiguous(img2);
    let dl_dmap = into_contiguous(dl_dmap);
    let dims = img1.shape().as_slice().to_vec();
    assert_eq!(
        dims.len(),
        3,
        "fused_ssim_backward expects [C, H, W] inputs"
    );
    let (c, h, w) = (dims[0] as u32, dims[1] as u32, dims[2] as u32);
    let dl_dimg1 = alloc_zeros(&img1);
    let client = img1.client.clone();
    // SAFETY: same boundary guarantees as the forward.
    unsafe {
        kernels::fused_ssim_backward_kernel::launch_unchecked::<f32, R>(
            &client,
            cube_count(c, h, w),
            CubeDim::new_2d(kernels::BLOCK_X, kernels::BLOCK_Y),
            img1.into_tensor_arg(),
            img2.into_tensor_arg(),
            dl_dmap.into_tensor_arg(),
            dl_dimg1.clone().into_tensor_arg(),
            h,
            w,
            l1_weight,
            ssim_weight,
        );
    }
    dl_dimg1
}

impl FusedSsimOps<Self> for MainBackendBase {
    fn fused_ssim_forward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        l1_weight: f32,
        ssim_weight: f32,
    ) -> FloatTensor<Self> {
        launch_forward(img1, img2, l1_weight, ssim_weight)
    }

    fn fused_ssim_backward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        l1_weight: f32,
        ssim_weight: f32,
    ) -> FloatTensor<Self> {
        launch_backward(img1, img2, dl_dmap, l1_weight, ssim_weight)
    }
}

impl FusedSsimOps<Self> for Fusion<MainBackendBase> {
    fn fused_ssim_forward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        l1_weight: f32,
        ssim_weight: f32,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            l1_weight: f32,
            ssim_weight: f32,
        }
        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();
                let [img1, img2] = inputs;
                let [map] = outputs;
                let out = <MainBackendBase as FusedSsimOps<MainBackendBase>>::fused_ssim_forward(
                    h.get_float_tensor::<MainBackendBase>(img1),
                    h.get_float_tensor::<MainBackendBase>(img2),
                    self.l1_weight,
                    self.ssim_weight,
                );
                h.register_float_tensor::<MainBackendBase>(&map.id, out);
            }
        }

        let shape = img1.shape();
        let client = img1.client.clone();
        let map_out = TensorIr::uninit(client.create_empty_handle(), shape, DType::F32);
        let inputs = [img1, img2];
        let stream = OperationStreams::with_inputs(&inputs);
        let desc = CustomOpIr::new(
            "fused_ssim_forward",
            &inputs.map(|t| t.into_ir()),
            &[map_out],
        );
        let op = CustomOp {
            desc: desc.clone(),
            l1_weight,
            ssim_weight,
        };
        let [map] = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();
        map
    }

    fn fused_ssim_backward(
        img1: FloatTensor<Self>,
        img2: FloatTensor<Self>,
        dl_dmap: FloatTensor<Self>,
        l1_weight: f32,
        ssim_weight: f32,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            l1_weight: f32,
            ssim_weight: f32,
        }
        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();
                let [img1, img2, dl_dmap] = inputs;
                let [dl_dimg1] = outputs;
                let out = <MainBackendBase as FusedSsimOps<MainBackendBase>>::fused_ssim_backward(
                    h.get_float_tensor::<MainBackendBase>(img1),
                    h.get_float_tensor::<MainBackendBase>(img2),
                    h.get_float_tensor::<MainBackendBase>(dl_dmap),
                    self.l1_weight,
                    self.ssim_weight,
                );
                h.register_float_tensor::<MainBackendBase>(&dl_dimg1.id, out);
            }
        }

        let shape = img1.shape();
        let client = img1.client.clone();
        let dl_dimg1_out = TensorIr::uninit(client.create_empty_handle(), shape, DType::F32);
        let inputs = [img1, img2, dl_dmap];
        let stream = OperationStreams::with_inputs(&inputs);
        let desc = CustomOpIr::new(
            "fused_ssim_backward",
            &inputs.map(|t| t.into_ir()),
            &[dl_dimg1_out],
        );
        let op = CustomOp {
            desc: desc.clone(),
            l1_weight,
            ssim_weight,
        };
        let [dl_dimg1] = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();
        dl_dimg1
    }
}

#[derive(Debug)]
struct FusedSsimBackward;

#[derive(Debug, Clone)]
struct FusedSsimState<B: Backend> {
    img1: FloatTensor<B>,
    img2: FloatTensor<B>,
    l1_weight: f32,
    ssim_weight: f32,
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
            state.l1_weight,
            state.ssim_weight,
        );
        if let Some(node) = img1_parent {
            grads.register::<B>(node.id, dl_dimg1);
        }
    }
}

/// Combined L1 + SSIM image loss in a single fused kernel.
///
/// Returns a `[H, W, C]` per-pixel map of `l1_weight * |img1 - img2| +
/// ssim_weight * ssim(img1, img2)`. Reduce with `.mean()` for the scalar
/// loss; backward routes the upstream gradient through both terms,
/// recomputing SSIM partials inline rather than reading saved tensors.
pub fn fused_image_loss<B, C>(
    img1: Tensor<Autodiff<B, C>, 3>,
    img2: Tensor<Autodiff<B, C>, 3>,
    l1_weight: f32,
    ssim_weight: f32,
) -> Tensor<Autodiff<B, C>, 3>
where
    B: Backend + FusedSsimOps<B>,
    C: CheckpointStrategy,
{
    // Permute [H, W, C] -> [C, H, W] (metadata-only; into_contiguous in the
    // launcher does the actual data movement).
    let img1_inner = img1.permute([2, 0, 1]).into_primitive().tensor();
    let img2_inner = img2.permute([2, 0, 1]).into_primitive().tensor();

    let prep = FusedSsimBackward
        .prepare::<C>([img1_inner.node.clone()])
        .compute_bound()
        .stateful();

    let img1_p = img1_inner.primitive;
    let img2_p = img2_inner.primitive;
    let map = B::fused_ssim_forward(img1_p.clone(), img2_p.clone(), l1_weight, ssim_weight);

    let map_p = match prep {
        OpsKind::Tracked(prep) => prep.finish(
            FusedSsimState {
                img1: img1_p,
                img2: img2_p,
                l1_weight,
                ssim_weight,
            },
            map,
        ),
        OpsKind::UnTracked(prep) => prep.finish(map),
    };

    Tensor::<Autodiff<B, C>, 3>::from_primitive(TensorPrimitive::Float(map_p)).permute([1, 2, 0])
}

pub fn fused_ssim<B, C>(
    img1: Tensor<Autodiff<B, C>, 3>,
    img2: Tensor<Autodiff<B, C>, 3>,
) -> Tensor<Autodiff<B, C>, 3>
where
    B: Backend + FusedSsimOps<B>,
    C: CheckpointStrategy,
{
    fused_image_loss(img1, img2, 0.0, 1.0)
}

/// Forward-only fused SSIM for non-differentiable backends, used by eval.
pub fn fused_ssim_eval<B>(img1: Tensor<B, 3>, img2: Tensor<B, 3>) -> Tensor<B, 3>
where
    B: Backend + FusedSsimOps<B>,
{
    let img1_p = img1.permute([2, 0, 1]).into_primitive().tensor();
    let img2_p = img2.permute([2, 0, 1]).into_primitive().tensor();
    let map = B::fused_ssim_forward(img1_p, img2_p, 0.0, 1.0);
    Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(map)).permute([1, 2, 0])
}
