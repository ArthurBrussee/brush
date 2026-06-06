//! Viewer-only colormap + RGBA8 packing for the PGSR geometry channels.
//! Takes the burn-computed depth / normal / alpha maps and writes a packed
//! `u32` image in the same layout the splat-backbuffer display shader reads.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// Cheap "jet" ramp (blue→cyan→green→yellow→red) over `t in [0, 1]`.
#[cube]
fn jet(t: f32) -> (f32, f32, f32) {
    let r = clamp(1.5f32 - f32::abs(4.0f32 * t - 3.0f32), 0.0f32, 1.0f32);
    let g = clamp(1.5f32 - f32::abs(4.0f32 * t - 2.0f32), 0.0f32, 1.0f32);
    let b = clamp(1.5f32 - f32::abs(4.0f32 * t - 1.0f32), 0.0f32, 1.0f32);
    (r, g, b)
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn colormap_pack_kernel(
    depth: &Tensor<f32>,
    normal: &Tensor<f32>,
    alpha: &Tensor<f32>,
    out: &mut Tensor<u32>,
    dmin: f32,
    dinv_range: f32,
    num_pixels: u32,
    #[comptime] normal_mode: bool,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= num_pixels {
        terminate!();
    }

    let mut r = 0.0f32;
    let mut g = 0.0f32;
    let mut b = 0.0f32;

    // Empty pixels (no surface) stay black.
    if alpha[idx as usize] > 0.5f32 {
        if comptime![normal_mode] {
            let nb = (idx * 3u32) as usize;
            r = clamp(normal[nb] * 0.5f32 + 0.5f32, 0.0f32, 1.0f32);
            g = clamp(normal[nb + 1] * 0.5f32 + 0.5f32, 0.0f32, 1.0f32);
            b = clamp(normal[nb + 2] * 0.5f32 + 0.5f32, 0.0f32, 1.0f32);
        } else {
            let t = clamp((depth[idx as usize] - dmin) * dinv_range, 0.0f32, 1.0f32);
            let (jr, jg, jb) = jet(t);
            r = jr;
            g = jg;
            b = jb;
        }
    }

    let ri = clamp(r * 255.0f32, 0.0f32, 255.0f32) as u32;
    let gi = clamp(g * 255.0f32, 0.0f32, 255.0f32) as u32;
    let bi = clamp(b * 255.0f32, 0.0f32, 255.0f32) as u32;
    out[idx as usize] = ri | (gi << 8u32) | (bi << 16u32) | (255u32 << 24u32);
}
