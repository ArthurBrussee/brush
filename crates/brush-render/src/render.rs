use std::mem::offset_of;

use super::shaders;
use crate::{
    INTERSECTS_UPPER_BOUND, MainBackendBase, camera::Camera, dim_check::DimCheck,
    render_aux::RenderAux, sh::sh_degree_from_coeffs,
};

// WGSL kernels (used when cubecl-kernels feature is not enabled)
#[cfg(not(feature = "cubecl-kernels"))]
use crate::kernels::{MapGaussiansToIntersect, ProjectSplats, ProjectVisible, Rasterize};

// CubeCL versions of kernels when feature is enabled
#[cfg(feature = "cubecl-kernels")]
use crate::cubecl::{map_gaussian_to_intersects, project_forward, rasterize};

// These are always available from burn_cubecl
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::prelude::{ABSOLUTE_POS, Tensor};
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::cubecl::{self, terminate};
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use burn_wgpu::{CubeDim, CubeTensor};

#[cfg(feature = "cubecl-kernels")]
use burn_cubecl::cubecl::frontend::ScalarArg;

use brush_kernel::create_dispatch_buffer;
use brush_kernel::create_tensor;
use brush_kernel::create_uniform_buffer;
use brush_kernel::{CubeCount, calc_cube_count};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::{DType, IntDType};

#[cfg(feature = "cubecl-kernels")]
use burn::tensor::TensorData;
use burn::tensor::{
    FloatDType,
    ops::{FloatTensorOps, IntTensorOps},
};

use glam::{Vec3, uvec2};

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
pub fn max_intersections(img_size: glam::UVec2, num_splats: u32) -> u32 {
    // Divide screen into tiles.
    let tile_bounds = calc_tile_bounds(img_size);
    // Assume on average each splat is maximally covering half x half the screen,
    // and adjust for the variance such that we're fairly certain we have enough intersections.
    let num_tiles = tile_bounds[0] * tile_bounds[1];
    let max_possible = num_tiles.saturating_mul(num_splats);
    // clamp to max nr. of dispatches.
    max_possible.min(INTERSECTS_UPPER_BOUND)
}

pub(crate) fn render_forward(
    camera: &Camera,
    img_size: glam::UVec2,
    means: CubeTensor<WgpuRuntime>,
    log_scales: CubeTensor<WgpuRuntime>,
    quats: CubeTensor<WgpuRuntime>,
    sh_coeffs: CubeTensor<WgpuRuntime>,
    raw_opacities: CubeTensor<WgpuRuntime>,
    background: Vec3,
    bwd_info: bool,
) -> (CubeTensor<WgpuRuntime>, RenderAux<MainBackendBase>) {
    assert!(
        img_size[0] > 0 && img_size[1] > 0,
        "Can't render images with 0 size."
    );

    // Tensor params might not be contiguous, convert them to contiguous tensors.
    let means = into_contiguous(means);
    let log_scales = into_contiguous(log_scales);
    let quats = into_contiguous(quats);
    let sh_coeffs = into_contiguous(sh_coeffs);
    let raw_opacities = into_contiguous(raw_opacities);

    let device = &means.device.clone();
    let client = means.client.clone();

    let _span = tracing::trace_span!("render_forward").entered();

    // Check whether input dimensions are valid.
    DimCheck::new()
        .check_dims("means", &means, &["D".into(), 3.into()])
        .check_dims("log_scales", &log_scales, &["D".into(), 3.into()])
        .check_dims("quats", &quats, &["D".into(), 4.into()])
        .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
        .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

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
    let max_intersects = max_intersections(img_size, total_splats as u32);

    let uniforms = shaders::helpers::RenderUniforms {
        viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        focal: camera.focal(img_size).into(),
        pixel_center: camera.center(img_size).into(),
        img_size: img_size.into(),
        tile_bounds: tile_bounds.into(),
        sh_degree,
        total_splats: total_splats as u32,
        max_intersects,
        background: [background.x, background.y, background.z, 1.0],
        // Nb: Bit of a hack as these aren't _really_ uniforms but are written to by the shaders.
        num_visible: 0,
    };

    // Nb: This contains both static metadata and some dynamic data so can't pass this as metadata to execute. In the future
    // should separate the two.
    let uniforms_buffer = create_uniform_buffer(uniforms, device, &client);

    let client = &means.client.clone();

    let (global_from_compact_gid, num_visible) = {
        let global_from_presort_gid =
            MainBackendBase::int_zeros([total_splats].into(), device, IntDType::U32);
        let depths = create_tensor([total_splats], device, DType::F32);

        #[cfg(not(feature = "cubecl-kernels"))]
        let num_visible = {
            tracing::trace_span!("ProjectSplats").in_scope(||
                // SAFETY: Kernel checked to have no OOB, bounded loops.
                unsafe {
                client.execute_unchecked(
                    ProjectSplats::task(),
                    calc_cube_count([total_splats as u32], ProjectSplats::WORKGROUP_SIZE),
                    Bindings::new().with_buffers(
                    vec![
                        uniforms_buffer.handle.clone().binding(),
                        means.handle.clone().binding(),
                        quats.handle.clone().binding(),
                        log_scales.handle.clone().binding(),
                        raw_opacities.handle.clone().binding(),
                        global_from_presort_gid.handle.clone().binding(),
                        depths.handle.clone().binding(),
                    ]),
                );
            });

            // Get just the number of visible splats from the uniforms buffer.
            let num_vis_field_offset =
                offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
            MainBackendBase::int_slice(
                uniforms_buffer.clone(),
                &[(num_vis_field_offset..num_vis_field_offset + 1).into()],
            )
        };

        #[cfg(feature = "cubecl-kernels")]
        let num_visible = {
            use burn::tensor::TensorData;

            // Create atomic counter for num_visible
            let num_visible = MainBackendBase::int_zeros([1].into(), device, IntDType::U32);

            // Extract viewmat as 12-element tensor [r00, r10, r20, t_x, r01, r11, r21, t_y, r02, r12, r22, t_z]
            let viewmat_cols = glam::Mat4::from(camera.world_to_local()).to_cols_array();
            let viewmat_data = vec![
                viewmat_cols[0],
                viewmat_cols[1],
                viewmat_cols[2],
                viewmat_cols[3], // col 0
                viewmat_cols[4],
                viewmat_cols[5],
                viewmat_cols[6],
                viewmat_cols[7], // col 1
                viewmat_cols[8],
                viewmat_cols[9],
                viewmat_cols[10],
                viewmat_cols[11], // col 2
            ];
            let viewmat_tensor =
                MainBackendBase::float_from_data(TensorData::new(viewmat_data, [12]), device);

            // Create uniforms tensor with [focal_x, focal_y, pixel_center_x, pixel_center_y, img_size_x, img_size_y]
            let focal = camera.focal(img_size);
            let pixel_center = camera.center(img_size);
            let uniforms_data = vec![
                focal.x,
                focal.y,
                pixel_center.x,
                pixel_center.y,
                img_size[0] as f32,
                img_size[1] as f32,
            ];
            let uniforms_tensor =
                MainBackendBase::float_from_data(TensorData::new(uniforms_data, [6]), device);

            tracing::trace_span!("ProjectForward").in_scope(|| {
                // SAFETY: Kernel checked to have no OOB, bounded loops.
                unsafe {
                    project_forward::project_forward::launch_unchecked::<WgpuRuntime>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((total_splats as u32).div_ceil(256), 1, 1),
                        means.as_tensor_arg::<f32>(3),
                        quats.as_tensor_arg::<f32>(4),
                        log_scales.as_tensor_arg::<f32>(3),
                        raw_opacities.as_tensor_arg::<f32>(1),
                        global_from_presort_gid.as_tensor_arg::<u32>(1),
                        depths.as_tensor_arg::<f32>(1),
                        num_visible.as_tensor_arg::<u32>(1),
                        ScalarArg::new(total_splats as u32),
                        viewmat_tensor.as_tensor_arg::<f32>(1),
                        uniforms_tensor.as_tensor_arg::<f32>(1),
                    );
                }
            });

            num_visible
        };

        let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort").in_scope(|| {
            // Interpret the depth as a u32. This is fine for a radix sort, as long as the depth > 0.0,
            // which we know to be the case given how we cull splats.
            radix_argsort(depths, global_from_presort_gid, &num_visible, 32)
        });

        (global_from_compact_gid, num_visible)
    };

    // Create a buffer of 'projected' splats, that is,
    // project XY, projected conic, and converted color.
    let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / size_of::<f32>();
    let projected_splats = create_tensor([total_splats, proj_size], device, DType::F32);

    #[cfg(not(feature = "cubecl-kernels"))]
    tracing::trace_span!("ProjectVisible").in_scope(|| {
        // Create a buffer to determine how many threads to dispatch for all visible splats.
        let num_vis_wg = create_dispatch_buffer(
            num_visible.clone(),
            shaders::project_visible::WORKGROUP_SIZE,
        );
        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client.execute_unchecked(
                ProjectVisible::task(),
                CubeCount::Dynamic(num_vis_wg.handle.binding()),
                Bindings::new().with_buffers(vec![
                    uniforms_buffer.clone().handle.binding(),
                    means.handle.binding(),
                    log_scales.handle.binding(),
                    quats.handle.binding(),
                    sh_coeffs.handle.binding(),
                    raw_opacities.handle.binding(),
                    global_from_compact_gid.handle.clone().binding(),
                    projected_splats.handle.clone().binding(),
                ]),
            );
        }
    });

    #[cfg(feature = "cubecl-kernels")]
    tracing::trace_span!("ProjectVisibleCubeCL").in_scope(|| {
        // Prepare viewmat buffer (16 floats - all 4 columns of 4x4 matrix)
        // Note: uniforms.viewmat is column-major from to_cols_array_2d()
        // Columns 0-2 contain the rotation matrix, column 3 contains translation + padding
        let viewmat_data: Vec<f32> = uniforms
            .viewmat
            .iter()
            .flat_map(|col| col.iter().copied())
            .collect();
        let viewmat_tensor =
            MainBackendBase::float_from_data(TensorData::new(viewmat_data, [16]), device);

        // Prepare camera position (3 floats)
        let camera_pos_tensor = MainBackendBase::float_from_data(
            TensorData::new(uniforms.camera_position[..3].to_vec(), [3]),
            device,
        );

        // Prepare uniforms buffer (8 floats: focal_x, focal_y, center_x, center_y, img_w, img_h, num_visible, sh_degree)
        // For num_visible, we'll pass 0 as placeholder since we use dynamic dispatch
        let uniforms_data = vec![
            uniforms.focal[0],
            uniforms.focal[1],
            uniforms.pixel_center[0],
            uniforms.pixel_center[1],
            uniforms.img_size[0] as f32,
            uniforms.img_size[1] as f32,
            0.0, // num_visible placeholder - handled by dynamic dispatch
            sh_degree as f32,
        ];
        let uniforms_tensor =
            MainBackendBase::float_from_data(TensorData::new(uniforms_data, [8]), device);

        // Use dynamic dispatch based on num_visible (same as WGSL version)
        let num_vis_wg = create_dispatch_buffer(num_visible.clone(), [256, 1, 1]);
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Dynamic(num_vis_wg.handle.binding());

        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            project_visible::project_visible::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                means.as_tensor_arg::<f32>(1),
                log_scales.as_tensor_arg::<f32>(1),
                quats.as_tensor_arg::<f32>(1),
                sh_coeffs.as_tensor_arg::<f32>(1),
                raw_opacities.as_tensor_arg::<f32>(1),
                global_from_compact_gid.as_tensor_arg::<u32>(1),
                projected_splats.as_tensor_arg::<f32>(1),
                viewmat_tensor.as_tensor_arg::<f32>(1),
                camera_pos_tensor.as_tensor_arg::<f32>(1),
                uniforms_tensor.as_tensor_arg::<f32>(1),
            );
        }
    });

    // Each intersection maps to a gaussian.
    let (tile_offsets, compact_gid_from_isect, num_intersections) = {
        let num_tiles = tile_bounds.x * tile_bounds.y;

        let splat_intersect_counts =
            MainBackendBase::int_zeros([total_splats + 1].into(), device, IntDType::U32);

        #[cfg(not(feature = "cubecl-kernels"))]
        let num_vis_map_wg = create_dispatch_buffer(
            num_visible,
            shaders::map_gaussian_to_intersects::WORKGROUP_SIZE,
        );

        // First do a prepass to compute the tile counts, then fill in intersection counts.
        #[cfg(feature = "cubecl-kernels")]
        tracing::trace_span!("MapGaussiansToIntersectPrepassCubeCL").in_scope(|| {
            let num_vis_wg = create_dispatch_buffer(num_visible.clone(), [256, 1, 1]);
            let cube_dim = CubeDim::new_1d(256);
            let cube_count = CubeCount::Dynamic(num_vis_wg.handle.binding());

            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                map_gaussian_to_intersects::map_gaussian_to_intersects_prepass::launch_unchecked::<
                    WgpuRuntime,
                >(
                    &client,
                    cube_count,
                    cube_dim,
                    projected_splats.as_tensor_arg::<f32>(1),
                    splat_intersect_counts.as_tensor_arg::<u32>(1),
                    num_visible.as_tensor_arg::<u32>(1),
                    ScalarArg::new(tile_bounds.x),
                    ScalarArg::new(tile_bounds.y),
                );
            }
        });

        #[cfg(not(feature = "cubecl-kernels"))]
        tracing::trace_span!("MapGaussiansToIntersectPrepass").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client.execute_unchecked(
                    MapGaussiansToIntersect::task(true),
                    CubeCount::Dynamic(num_vis_map_wg.handle.clone().binding()),
                    Bindings::new().with_buffers(vec![
                        uniforms_buffer.handle.clone().binding(),
                        projected_splats.handle.clone().binding(),
                        splat_intersect_counts.handle.clone().binding(),
                    ]),
                );
            }
        });

        // TODO: Only need to do this up to num_visible gaussians really.
        let cum_tiles_hit = tracing::trace_span!("PrefixSumGaussHits")
            .in_scope(|| prefix_sum(splat_intersect_counts));

        let tile_id_from_isect = create_tensor([max_intersects as usize], device, DType::U32);
        let compact_gid_from_isect = create_tensor([max_intersects as usize], device, DType::U32);

        // Zero this out, as the kernel _might_ not run at all if no gaussians are visible.
        let num_intersections = MainBackendBase::int_zeros([1].into(), device, IntDType::U32);

        #[cfg(feature = "cubecl-kernels")]
        tracing::trace_span!("MapGaussiansToIntersectMainCubeCL").in_scope(|| {
            let num_vis_wg = create_dispatch_buffer(num_visible.clone(), [256, 1, 1]);
            let cube_dim = CubeDim::new_1d(256);
            let cube_count = CubeCount::Dynamic(num_vis_wg.handle.binding());

            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                map_gaussian_to_intersects::map_gaussian_to_intersects_main::launch_unchecked::<
                    WgpuRuntime,
                >(
                    &client,
                    cube_count,
                    cube_dim,
                    projected_splats.as_tensor_arg::<f32>(1),
                    cum_tiles_hit.as_tensor_arg::<u32>(1),
                    tile_id_from_isect.as_tensor_arg::<u32>(1),
                    compact_gid_from_isect.as_tensor_arg::<u32>(1),
                    num_intersections.as_tensor_arg::<u32>(1),
                    num_visible.as_tensor_arg::<u32>(1),
                    ScalarArg::new(tile_bounds.x),
                    ScalarArg::new(tile_bounds.y),
                );
            }
        });

        #[cfg(not(feature = "cubecl-kernels"))]
        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client.execute_unchecked(
                    MapGaussiansToIntersect::task(false),
                    CubeCount::Dynamic(num_vis_map_wg.handle.clone().binding()),
                    Bindings::new().with_buffers(vec![
                        uniforms_buffer.handle.clone().binding(),
                        projected_splats.handle.clone().binding(),
                        cum_tiles_hit.handle.binding(),
                        tile_id_from_isect.handle.clone().binding(),
                        compact_gid_from_isect.handle.clone().binding(),
                        num_intersections.handle.clone().binding(),
                    ]),
                );
            }
        });

        // We're sorting by tile ID, but we know beforehand what the maximum value
        // can be. We don't need to sort all the leading 0 bits!
        let bits = u32::BITS - num_tiles.leading_zeros();

        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| {
                radix_argsort(
                    tile_id_from_isect,
                    compact_gid_from_isect,
                    &num_intersections,
                    bits,
                )
            });

        #[cube(launch_unchecked)]
        pub fn get_tile_offsets(
            tile_id_from_isect: &Tensor<u32>,
            tile_offsets: &mut Tensor<u32>,
            num_inter: &Tensor<u32>,
        ) {
            let inter = num_inter[0];
            let isect_id = ABSOLUTE_POS;
            if isect_id >= inter {
                terminate!();
            }
            let prev_tid = tile_id_from_isect[isect_id - 1];
            let tid = tile_id_from_isect[isect_id];

            if isect_id == inter - 1 {
                // Write the end of the previous tile.
                tile_offsets[tid * 2 + 1] = ABSOLUTE_POS + 1;
            }
            if tid != prev_tid {
                // Write the end of the previous tile.
                tile_offsets[prev_tid * 2 + 1] = ABSOLUTE_POS;
                // Write start of this tile.
                tile_offsets[tid * 2] = ABSOLUTE_POS;
            }
        }

        let cube_dim = CubeDim::new_1d(512);
        let num_vis_map_wg = create_dispatch_buffer(num_intersections.clone(), [256, 1, 1]);
        let cube_count = CubeCount::Dynamic(num_vis_map_wg.handle.binding());

        // Tiles without splats will be written as having a range of [0, 0].
        let tile_offsets = MainBackendBase::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            device,
            IntDType::U32,
        );

        // SAFETY: Safe kernel.
        unsafe {
            get_tile_offsets::launch_unchecked::<WgpuRuntime>(
                client,
                cube_count,
                cube_dim,
                tile_id_from_isect.as_tensor_arg::<u32>(1),
                tile_offsets.as_tensor_arg::<u32>(1),
                num_intersections.as_tensor_arg::<u32>(1),
            );
        }

        (tile_offsets, compact_gid_from_isect, num_intersections)
    };

    let _span = tracing::trace_span!("Rasterize").entered();

    #[cfg(feature = "cubecl-kernels")]
    let out_dim = 4; // CubeCL always outputs 4 floats (RGBA)

    #[cfg(not(feature = "cubecl-kernels"))]
    let out_dim = if bwd_info {
        4
    } else {
        // Channels are packed into 4 bytes, aka one float.
        1
    };

    let out_img = create_tensor(
        [img_size.y as usize, img_size.x as usize, out_dim],
        device,
        DType::F32,
    );

    #[cfg_attr(feature = "cubecl-kernels", allow(unused_mut, unused_variables))]
    let mut bindings = Bindings::new().with_buffers(vec![
        uniforms_buffer.handle.clone().binding(),
        compact_gid_from_isect.handle.clone().binding(),
        tile_offsets.handle.clone().binding(),
        projected_splats.handle.clone().binding(),
        out_img.handle.clone().binding(),
    ]);

    let visible = if bwd_info {
        let visible = MainBackendBase::float_zeros([total_splats].into(), device, FloatDType::F32);

        // Add the buffer to the bindings (only used for WGSL path)
        #[cfg(not(feature = "cubecl-kernels"))]
        {
            bindings = bindings.with_buffers(vec![
                global_from_compact_gid.handle.clone().binding(),
                visible.handle.clone().binding(),
            ]);
        }

        visible
    } else {
        create_tensor([1], device, DType::F32)
    };

    // Compile the kernel, including/excluding info for backwards pass.
    // see the BWD_INFO define in the rasterize shader.
    #[cfg(not(feature = "cubecl-kernels"))]
    {
        let raster_task = Rasterize::task(bwd_info, cfg!(target_family = "wasm"));

        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client.execute_unchecked(
                raster_task,
                CubeCount::Static(tile_bounds.x * tile_bounds.y, 1, 1),
                bindings,
            );
        }
    }

    #[cfg(feature = "cubecl-kernels")]
    tracing::trace_span!("RasterizeCubeCL").in_scope(|| {
        let cube_dim = CubeDim::new_1d(256);
        let num_tiles = tile_bounds.x * tile_bounds.y;
        let cube_count = CubeCount::Static(num_tiles, 1, 1);

        // Prepare uniforms tensor for kernel
        let uniforms_data = vec![
            img_size.x as f32,
            img_size.y as f32,
            tile_bounds.x as f32,
            background.x,
            background.y,
            background.z,
        ];
        let uniforms_tensor =
            MainBackendBase::float_from_data(TensorData::new(uniforms_data, [6]), device);

        // Create dummy tensors for unused parameters when bwd_info=false
        let dummy_gid = if bwd_info {
            global_from_compact_gid.clone()
        } else {
            create_tensor([1], device, DType::U32)
        };
        let dummy_visible = if bwd_info {
            visible.clone()
        } else {
            create_tensor([1], device, DType::F32)
        };

        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            rasterize::rasterize::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                compact_gid_from_isect.as_tensor_arg::<u32>(1),
                tile_offsets.as_tensor_arg::<u32>(1),
                projected_splats.as_tensor_arg::<f32>(1),
                out_img.as_tensor_arg::<f32>(1),
                dummy_gid.as_tensor_arg::<u32>(1),
                dummy_visible.as_tensor_arg::<f32>(1),
                uniforms_tensor.as_tensor_arg::<f32>(1),
                bwd_info,
            );
        }
    });

    // Sanity check the buffers.
    assert!(
        uniforms_buffer.is_contiguous(),
        "Uniforms must be contiguous"
    );
    assert!(
        tile_offsets.is_contiguous(),
        "Tile offsets must be contiguous"
    );
    assert!(
        global_from_compact_gid.is_contiguous(),
        "Global from compact gid must be contiguous"
    );
    assert!(visible.is_contiguous(), "Visible must be contiguous");
    assert!(
        projected_splats.is_contiguous(),
        "Projected splats must be contiguous"
    );
    assert!(
        num_intersections.is_contiguous(),
        "Num intersections must be contiguous"
    );

    (
        out_img,
        RenderAux {
            uniforms_buffer,
            tile_offsets,
            num_intersections,
            projected_splats,
            compact_gid_from_isect,
            global_from_compact_gid,
            visible,
            img_size,
        },
    )
}
