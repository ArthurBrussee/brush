use super::helpers::*;
use burn::cubecl;
use burn_cubecl::cubecl::prelude::*;

/// Rasterize kernel - renders gaussians to output image with alpha blending
/// Uses SharedMemory for efficient batching of gaussians per tile
///
/// BWD_INFO: When true, outputs float4 and tracks visibility for backward pass
#[cube(launch_unchecked)]
pub fn rasterize(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    out_img: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    uniforms: &Tensor<f32>,
    #[comptime] bwd_info: bool,
) {
    let global_id = ABSOLUTE_POS;

    // Extract parameters from uniforms
    let img_size_x = u32::cast_from(uniforms[0]);
    let img_size_y = u32::cast_from(uniforms[1]);
    let tile_bounds_x = u32::cast_from(uniforms[2]);
    let background_r = uniforms[3];
    let background_g = uniforms[4];
    let background_b = uniforms[5];

    let pix = map_1d_to_2d(global_id, tile_bounds_x);
    let pix_x = pix.0;
    let pix_y = pix.1;

    let inside = pix_x < img_size_x && pix_y < img_size_y;

    let pix_id = if inside {
        pix_x + pix_y * img_size_x
    } else {
        u32::new(0)
    };
    let pixel_coord_x = f32::cast_from(pix_x) + 0.5;
    let pixel_coord_y = f32::cast_from(pix_y) + 0.5;

    let tile_x = pix_x / TILE_WIDTH;
    let tile_y = pix_y / TILE_WIDTH;
    let tile_id = tile_x + tile_y * tile_bounds_x;

    let range_start = tile_offsets[tile_id * 2];
    let range_end = tile_offsets[tile_id * 2 + 1];

    let mut shared_batch = SharedMemory::<f32>::new(TILE_SIZE * 9);
    let mut shared_gids = SharedMemory::<u32>::new(TILE_SIZE);

    let mut t = 1.0;
    let mut pix_r = 0.0;
    let mut pix_g = 0.0;
    let mut pix_b = 0.0;
    let mut done = !inside;

    let mut batch_start = range_start;
    while batch_start < range_end {
        let remaining = Min::min(TILE_SIZE, range_end - batch_start);

        let load_isect_id = batch_start + UNIT_POS;
        let compact_gid = compact_gid_from_isect[load_isect_id];

        sync_cube();
        if UNIT_POS < remaining {
            let base_idx = compact_gid * 9;

            // For BWD_INFO, also load global gaussian IDs
            if bwd_info {
                let global_gid = global_from_compact_gid[compact_gid];
                shared_gids[UNIT_POS] = global_gid;
            }

            let shared_base = UNIT_POS * 9;
            shared_batch[shared_base] = projected[base_idx];
            shared_batch[shared_base + 1] = projected[base_idx + 1];
            shared_batch[shared_base + 2] = projected[base_idx + 2];
            shared_batch[shared_base + 3] = projected[base_idx + 3];
            shared_batch[shared_base + 4] = projected[base_idx + 4];
            shared_batch[shared_base + 5] = projected[base_idx + 5];
            shared_batch[shared_base + 6] = projected[base_idx + 6];
            shared_batch[shared_base + 7] = projected[base_idx + 7];
            shared_batch[shared_base + 8] = projected[base_idx + 8];
        }
        sync_cube();

        let mut t_idx = 0u32;
        while !done && t_idx < remaining {
            let shared_base = t_idx * 9;
            let xy_x = shared_batch[shared_base];
            let xy_y = shared_batch[shared_base + 1];
            let conic_x = shared_batch[shared_base + 2];
            let conic_y = shared_batch[shared_base + 3];
            let conic_z = shared_batch[shared_base + 4];
            let color_r = shared_batch[shared_base + 5];
            let color_g = shared_batch[shared_base + 6];
            let color_b = shared_batch[shared_base + 7];
            let color_a = shared_batch[shared_base + 8];

            let sigma = calc_sigma(
                pixel_coord_x,
                pixel_coord_y,
                conic_x,
                conic_y,
                conic_z,
                xy_x,
                xy_y,
            );

            let alpha = f32::min(0.999, color_a * f32::exp(-sigma));

            if sigma >= 0.0 && alpha >= 1.0 / 255.0 {
                let next_t = t * (1.0 - alpha);

                if next_t <= 1e-4 {
                    done = true;
                    break;
                }

                // For BWD_INFO, mark gaussian as visible
                if bwd_info {
                    let global_gid = shared_gids[t_idx];
                    visible[global_gid] = 1.0;
                }

                let vis = alpha * t;
                pix_r += f32::max(0.0, color_r) * vis;
                pix_g += f32::max(0.0, color_g) * vis;
                pix_b += f32::max(0.0, color_b) * vis;
                t = next_t;
            }

            t_idx += 1;
        }

        batch_start += TILE_SIZE;
    }

    if inside {
        let final_r = pix_r + t * background_r;
        let final_g = pix_g + t * background_g;
        let final_b = pix_b + t * background_b;
        let final_a = 1.0 - t;

        // CubeCL always outputs float4 (no packing like WGSL)
        // This is simpler and avoids bitcast issues
        let base_out = pix_id * 4;
        out_img[base_out] = final_r;
        out_img[base_out + 1] = final_g;
        out_img[base_out + 2] = final_b;
        out_img[base_out + 3] = final_a;
    }
}
