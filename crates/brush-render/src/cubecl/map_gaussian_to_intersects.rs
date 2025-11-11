use burn::cubecl;
use super::helpers::*;
use burn_cubecl::cubecl::prelude::*;

/// Prepass: Count how many tiles each gaussian intersects
#[cube(launch_unchecked)]
pub fn map_gaussian_to_intersects_prepass(
    projected: &Tensor<f32>,
    splat_intersect_counts: &mut Tensor<u32>,
    num_visible: &Tensor<u32>,
    tile_bounds_x: u32,
    tile_bounds_y: u32,
) {
    let compact_gid = ABSOLUTE_POS;

    // Read num_visible from GPU tensor (it's a single u32 value at index 0)
    let num_vis = num_visible[0];

    if compact_gid >= num_vis {
        terminate!();
    }

    // Read ProjectedSplat data (9 floats per splat)
    // projected tensor is flattened, each splat takes 9 consecutive floats
    let base_idx = compact_gid * 9;
    let xy_x = projected[base_idx];
    let xy_y = projected[base_idx + 1];
    let conic_x = projected[base_idx + 2];
    let conic_y = projected[base_idx + 3];
    let conic_z = projected[base_idx + 4];
    let color_a = projected[base_idx + 8]; // opacity

    let mean2d_x = xy_x;
    let mean2d_y = xy_y;

    // Calculate power threshold
    let power_threshold = Log::log(color_a * 255.0);

    // Calculate 2D covariance from conic (inverse)
    let cov2d = mat2_inverse(conic_x, conic_y, conic_y, conic_z);

    // Calculate bounding box extent
    let extent = compute_bbox_extent(cov2d, power_threshold);
    let extent_x = extent.0;
    let extent_y = extent.1;

    // Get tile bounding box
    let tile_bbox = get_tile_bbox(
        mean2d_x,
        mean2d_y,
        extent_x,
        extent_y,
        tile_bounds_x,
        tile_bounds_y,
    );
    let tile_bbox_min_x = tile_bbox.0;
    let tile_bbox_min_y = tile_bbox.1;
    let tile_bbox_max_x = tile_bbox.2;
    let tile_bbox_max_y = tile_bbox.3;

    let mut num_tiles_hit = 0u32;

    // Iterate over tiles in bounding box
    let tile_bbox_width = tile_bbox_max_x - tile_bbox_min_x;
    let num_tiles_bbox = (tile_bbox_max_y - tile_bbox_min_y) * tile_bbox_width;

    let mut tile_idx = 0u32;
    while tile_idx < num_tiles_bbox {
        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min_x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min_y;

        let rect = tile_rect(tx, ty);
        let contributes = will_primitive_contribute(
            rect,
            mean2d_x,
            mean2d_y,
            conic_x,
            conic_y,
            conic_z,
            power_threshold,
        );

        if contributes {
            num_tiles_hit += 1;
        }

        tile_idx += 1;
    }

    // Write count to splat_intersect_counts[compact_gid + 1]
    // The +1 offset is for prefix sum algorithm
    splat_intersect_counts[compact_gid + 1] = num_tiles_hit;
}

/// Main pass: Write intersection data (tile_id and compact_gid pairs)
#[cube(launch_unchecked)]
pub fn map_gaussian_to_intersects_main(
    projected: &Tensor<f32>,
    splat_cum_hit_counts: &Tensor<u32>,
    tile_id_from_isect: &mut Tensor<u32>,
    compact_gid_from_isect: &mut Tensor<u32>,
    num_intersections: &mut Tensor<u32>,
    num_visible: &Tensor<u32>,
    tile_bounds_x: u32,
    tile_bounds_y: u32,
) {
    let compact_gid = ABSOLUTE_POS;

    // Read num_visible from GPU tensor (it's a single u32 value at index 0)
    let num_vis = num_visible[0];

    // Special case: thread 0 writes total number of intersections
    if compact_gid == 0 {
        num_intersections[0] = splat_cum_hit_counts[num_vis];
    }

    if compact_gid >= num_vis {
        terminate!();
    }

    // Read ProjectedSplat data (9 floats per splat)
    // projected tensor is flattened, each splat takes 9 consecutive floats
    let base_idx = compact_gid * 9;
    let xy_x = projected[base_idx];
    let xy_y = projected[base_idx + 1];
    let conic_x = projected[base_idx + 2];
    let conic_y = projected[base_idx + 3];
    let conic_z = projected[base_idx + 4];
    let color_a = projected[base_idx + 8]; // opacity

    let mean2d_x = xy_x;
    let mean2d_y = xy_y;

    // Calculate power threshold
    let power_threshold = Log::log(color_a * 255.0);

    // Calculate 2D covariance from conic (inverse)
    let cov2d = mat2_inverse(conic_x, conic_y, conic_y, conic_z);

    // Calculate bounding box extent
    let extent = compute_bbox_extent(cov2d, power_threshold);
    let extent_x = extent.0;
    let extent_y = extent.1;

    // Get tile bounding box
    let tile_bbox = get_tile_bbox(
        mean2d_x,
        mean2d_y,
        extent_x,
        extent_y,
        tile_bounds_x,
        tile_bounds_y,
    );
    let tile_bbox_min_x = tile_bbox.0;
    let tile_bbox_min_y = tile_bbox.1;
    let tile_bbox_max_x = tile_bbox.2;
    let tile_bbox_max_y = tile_bbox.3;

    // Get base intersection ID from cumulative counts
    let base_isect_id = splat_cum_hit_counts[compact_gid];
    let mut num_tiles_hit = 0u32;

    // Iterate over tiles in bounding box
    let tile_bbox_width = tile_bbox_max_x - tile_bbox_min_x;
    let num_tiles_bbox = (tile_bbox_max_y - tile_bbox_min_y) * tile_bbox_width;

    let mut tile_idx = 0u32;
    while tile_idx < num_tiles_bbox {
        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min_x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min_y;

        let rect = tile_rect(tx, ty);
        let contributes = will_primitive_contribute(
            rect,
            mean2d_x,
            mean2d_y,
            conic_x,
            conic_y,
            conic_z,
            power_threshold,
        );

        if contributes {
            let tile_id = tx + ty * tile_bounds_x;
            let isect_id = base_isect_id + num_tiles_hit;

            // Write intersection data
            // Note: isect_id might be out of bounds in degenerate cases
            // These kernels should be launched with bounds checking
            tile_id_from_isect[isect_id] = tile_id;
            compact_gid_from_isect[isect_id] = compact_gid;

            num_tiles_hit += 1;
        }

        tile_idx += 1;
    }
}
