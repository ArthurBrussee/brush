use crate::{
    BBase, INTERSECTS_UPPER_BOUND, RenderAux,
    camera::Camera,
    dim_check::DimCheck,
    kernels::{MapGaussiansToIntersect, ProjectSplats, ProjectVisible, Rasterize},
    sh::sh_degree_from_coeffs,
};

use super::shaders;

use brush_kernel::create_dispatch_buffer;
use brush_kernel::create_tensor;
use brush_kernel::create_uniform_buffer;
use brush_kernel::{CubeCount, calc_cube_count};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::{DType, Tensor};
use burn::tensor::{
    Int,
    ops::{FloatTensorOps, IntTensorOps},
};

use burn_cubecl::BoolElement;
use burn_wgpu::CubeTensor;
use burn_wgpu::WgpuRuntime;
use glam::uvec2;
use std::mem::{offset_of, size_of};

pub(crate) fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

// On wasm, we cannot do a sync readback at all.
// Instead, can just estimate a max number of intersects. All the kernels only handle the actual
// number of intersects, and spin up empty threads for the rest atm. In the future, could use indirect
// dispatch to avoid this.
// Estimating the max number of intersects can be a bad hack though... The worst case sceneario is so massive
// that it's easy to run out of memory... How do we actually properly deal with this :/
pub(crate) fn max_intersections(img_size: glam::UVec2, num_splats: u32) -> u32 {
    // Divide screen into tiles.
    let tile_bounds = calc_tile_bounds(img_size);
    let num_tiles = tile_bounds[0] * tile_bounds[1];
    let max = num_splats.saturating_mul(num_tiles);
    // clamp to max nr. of dispatches.
    max.min(INTERSECTS_UPPER_BOUND)
}

pub(crate) fn render_forward<BT: BoolElement>(
    camera: &Camera,
    img_size: glam::UVec2,
    means: CubeTensor<WgpuRuntime>,
    log_scales: CubeTensor<WgpuRuntime>,
    quats: CubeTensor<WgpuRuntime>,
    sh_coeffs: CubeTensor<WgpuRuntime>,
    opacities: CubeTensor<WgpuRuntime>,
    bwd_info: bool,
) -> (CubeTensor<WgpuRuntime>, RenderAux<BBase<BT>>) {
    assert!(
        img_size[0] > 0 && img_size[1] > 0,
        "Can't render images with 0 size."
    );

    let device = &means.device.clone();
    let client = means.client.clone();

    // Check whether any work needs to be flushed.
    tracing::trace_span!("pre setup", sync_burn = true).in_scope(|| {});

    let _span = tracing::trace_span!("render_forward", sync_burn = true).entered();

    // Check whether input dimensions are valid.
    DimCheck::new()
        .check_dims(&means, &["D".into(), 3.into()])
        .check_dims(&log_scales, &["D".into(), 3.into()])
        .check_dims(&quats, &["D".into(), 4.into()])
        .check_dims(&sh_coeffs, &["D".into(), "C".into(), 3.into()])
        .check_dims(&opacities, &["D".into()]);

    // Divide screen into tiles.
    let tile_bounds = calc_tile_bounds(img_size);

    // A note on some confusing naming that'll be used throughout this function:
    // Gaussians are stored in various states of buffers, eg. at the start they're all in one big buffer,
    // then we sparsely store some results, then sort gaussian based on depths, etc.
    // Overall this means there's lots of indices flying all over the place, and it's hard to keep track
    // what is indexing what. So, for some sanity, try to match a few "gaussian ids" (gid) variable names.
    // - Global Gaussin ID - global_gid
    // - Compacted Gaussian ID - compact_gid
    // - Per tile intersection depth sorted ID - tiled_gid
    // - Sorted by tile per tile intersection depth sorted ID - sorted_tiled_gid
    // Then, various buffers map between these, which are named x_from_y_gid, eg.
    //  global_from_compact_gid.

    // Tile rendering setup.
    let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape.dims[1] as u32);
    let total_splats = means.shape.dims[0];

    let uniforms = shaders::helpers::RenderUniforms {
        viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        focal: camera.focal(img_size).into(),
        pixel_center: camera.center(img_size).into(),
        img_size: img_size.into(),
        tile_bounds: tile_bounds.into(),
        sh_degree,
        total_splats: total_splats as u32,
        // Nb: Bit of a hack as these aren't _really_ uniforms but are written to by the shaders.
        num_visible: 0,
        num_intersections: 0,
    };

    let uniforms_buffer = create_uniform_buffer(uniforms, device, &client);

    let client = &means.client.clone();

    let (global_from_compact_gid, num_visible) = {
        let global_from_presort_gid = BBase::<BT>::int_zeros([total_splats].into(), device);
        let depths = create_tensor([total_splats], device, client, DType::F32);

        tracing::trace_span!("ProjectSplats", sync_burn = true).in_scope(||
            // SAFETY: Kernel checked to have no OOB.
            unsafe {
            client.execute_unchecked(
                ProjectSplats::task(),
                calc_cube_count([total_splats as u32], ProjectSplats::WORKGROUP_SIZE),
                vec![
                    uniforms_buffer.clone().handle.binding(),
                    means.clone().handle.binding(),
                    quats.clone().handle.binding(),
                    log_scales.clone().handle.binding(),
                    opacities.clone().handle.binding(),
                    global_from_presort_gid.clone().handle.binding(),
                    depths.clone().handle.binding(),
                ],
            );
        });

        // Get just the number of visible splats from the uniforms buffer.
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        let num_visible = BBase::<BT>::int_slice(
            uniforms_buffer.clone(),
            &[num_vis_field_offset..num_vis_field_offset + 1],
        );

        let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort", sync_burn = true)
            .in_scope(|| {
                // Interpret the depth as a u32. This is fine for a radix sort, as long as the depth > 0.0,
                // which we know to be the case given how we cull splats.
                radix_argsort(depths, global_from_presort_gid, &num_visible, 32)
            });

        (global_from_compact_gid, num_visible)
    };

    // Create a buffer of 'projected' splats, that is,
    // project XY, projected conic, and converted color.
    let projected_size = size_of::<shaders::helpers::ProjectedSplat>() / size_of::<f32>();
    let projected_splats =
        create_tensor::<2, _>([total_splats, projected_size], device, client, DType::F32);

    let max_intersects = max_intersections(img_size, total_splats as u32);
    // 1 extra length to make this an exclusive sum.
    let tiles_hit_per_splat = BBase::<BT>::int_zeros([total_splats + 1].into(), device);
    let isect_info =
        create_tensor::<2, WgpuRuntime>([max_intersects as usize, 2], device, client, DType::I32);

    // Create a buffer to determine how many threads to dispatch for all visible splats.
    let num_vis_wg = create_dispatch_buffer(num_visible.clone(), [shaders::helpers::MAIN_WG, 1, 1]);

    tracing::trace_span!("ProjectVisible", sync_burn = true).in_scope(||
        // SAFETY: Kernel has to contain no OOB indexing.
        unsafe {
        client.execute_unchecked(
            ProjectVisible::task(),
            CubeCount::Dynamic(num_vis_wg.clone().handle.binding()),
            vec![
                uniforms_buffer.clone().handle.binding(),
                means.handle.binding(),
                log_scales.handle.binding(),
                quats.handle.binding(),
                sh_coeffs.handle.binding(),
                opacities.handle.binding(),
                global_from_compact_gid.handle.clone().binding(),
                projected_splats.handle.clone().binding(),
                tiles_hit_per_splat.handle.clone().binding(),
                isect_info.handle.clone().binding(),
            ],
        );
    });

    // Create a tensor containing just the number of intersections.
    let num_intersections_offset =
        offset_of!(shaders::helpers::RenderUniforms, num_intersections) / 4;
    let num_intersections = BBase::<BT>::int_slice(
        uniforms_buffer.clone(),
        &[num_intersections_offset..num_intersections_offset + 1],
    );

    // Each intersection maps to a gaussian.
    let (tile_offsets, compact_gid_from_isect) = {
        let num_tiles = tile_bounds.x * tile_bounds.y;

        let tile_id_from_isect =
            create_tensor::<1, _>([max_intersects as usize], device, client, DType::I32);
        let compact_gid_from_isect =
            create_tensor::<1, _>([max_intersects as usize], device, client, DType::I32);

        // Number of intersections per tile. Range ID's are later derived from this
        // by a prefix sum.
        let tile_counts = BBase::<BT>::int_zeros(
            [(tile_bounds.y * tile_bounds.x) as usize + 1].into(),
            device,
        );

        let cum_tiles_hit = tracing::trace_span!("PrefixSum", sync_burn = true).in_scope(|| {
            // TODO: Only need to do this up to num_visible gaussians really.
            prefix_sum(tiles_hit_per_splat)
        });

        tracing::trace_span!("MapGaussiansToIntersect", sync_burn = true).in_scope(|| {
            // The workgroup size is [256, 2] to be an effective 512 threads.
            let num_intersects: Tensor<BBase<BT>, 1, Int> =
                Tensor::from_primitive(num_intersections.clone());
            let dispatch = (num_intersects + 1) / 2;
            let intersect_wg_buf = create_dispatch_buffer(
                dispatch.into_primitive(),
                MapGaussiansToIntersect::WORKGROUP_SIZE,
            );

            // SAFETY: Kernel has to contain no OOB indexing.
            unsafe {
                client.execute_unchecked(
                    MapGaussiansToIntersect::task(),
                    CubeCount::Dynamic(intersect_wg_buf.handle.binding()),
                    vec![
                        num_intersections.clone().handle.binding(),
                        isect_info.handle.clone().binding(),
                        cum_tiles_hit.handle.binding(),
                        tile_counts.handle.clone().binding(),
                        tile_id_from_isect.handle.clone().binding(),
                        compact_gid_from_isect.handle.clone().binding(),
                    ],
                );
            }
        });

        // We're sorting by tile ID, but we know beforehand what the maximum value
        // can be. We don't need to sort all the leading 0 bits!
        let bits = u32::BITS - num_tiles.leading_zeros();

        let (_, compact_gid_from_isect) = tracing::trace_span!("Tile sort", sync_burn = true)
            .in_scope(|| {
                radix_argsort(
                    tile_id_from_isect,
                    compact_gid_from_isect,
                    &num_intersections,
                    bits,
                )
            });

        let _span = tracing::trace_span!("PrefixSumTileCounts", sync_burn = true).entered();

        // Figure out start/end values for each tile.
        let tile_offsets = prefix_sum(tile_counts);
        (tile_offsets, compact_gid_from_isect)
    };

    let _span = tracing::trace_span!("Rasterize", sync_burn = true).entered();

    let out_dim = if bwd_info {
        4
    } else {
        // Channels are packed into 4 bytes, aka one float.
        1
    };

    let out_img = create_tensor(
        [img_size.y as usize, img_size.x as usize, out_dim],
        device,
        client,
        if bwd_info { DType::F32 } else { DType::U32 },
    );

    let mut bindings = vec![
        uniforms_buffer.clone().handle.binding(),
        compact_gid_from_isect.handle.clone().binding(),
        tile_offsets.handle.clone().binding(),
        projected_splats.handle.clone().binding(),
        out_img.handle.clone().binding(),
    ];

    let (visible, final_index) = if bwd_info {
        let visible = BBase::<BT>::float_zeros([total_splats].into(), device);

        // Buffer containing the final visible splat per tile.
        let final_index = create_tensor::<2, _>(
            [img_size.y as usize, img_size.x as usize],
            device,
            client,
            DType::I32,
        );

        bindings.push(global_from_compact_gid.handle.clone().binding());
        bindings.push(final_index.handle.clone().binding());
        bindings.push(visible.handle.clone().binding());

        (visible, final_index)
    } else {
        let visible = create_tensor::<1, _>([1], device, client, DType::F32);

        // Buffer containing the final visible splat per tile.
        let final_index = create_tensor::<2, _>([1, 1], device, client, DType::I32);
        (visible, final_index)
    };

    // Compile the kernel, including/excluding info for backwards pass.
    // see the BWD_INFO define in the rasterize shader.
    let raster_task = Rasterize::task(bwd_info);

    // SAFETY: Kernel has to contain no OOB indexing.
    unsafe {
        client.execute_unchecked(
            raster_task,
            calc_cube_count([img_size.x, img_size.y], Rasterize::WORKGROUP_SIZE),
            bindings,
        );
    }

    (
        out_img,
        RenderAux {
            uniforms_buffer,
            num_visible,
            num_intersections,
            tile_offsets,
            projected_splats,
            compact_gid_from_isect,
            global_from_compact_gid,
            visible,
            final_index,
        },
    )
}
