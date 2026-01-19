use crate::{
    MainBackendBase, RenderAux, SplatOps,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::SplatRenderMode,
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    render_aux::ProjectOutput,
    sh::sh_degree_from_coeffs,
    shaders::{self, MapGaussiansToIntersect, ProjectSplats, ProjectVisible, Rasterize},
};
use brush_kernel::bytemuck;
use brush_kernel::create_dispatch_buffer_1d;
use brush_kernel::create_meta_binding;
use brush_kernel::create_tensor;
use brush_kernel::{CubeCount, calc_cube_count_1d};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::{DType, IntDType, Shape, ops::FloatTensor};
use burn::tensor::{
    FloatDType,
    ops::{FloatTensorOps, IntTensor, IntTensorOps},
};
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::{CubeDim, CubeTensor, WgpuRuntime};
use glam::{Vec3, uvec2};

pub(crate) fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

impl SplatOps<Self> for MainBackendBase {
    fn project(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
    ) -> ProjectOutput<Self> {
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

        let _span = tracing::trace_span!("project_prepare").entered();

        // Check whether input dimensions are valid.
        DimCheck::new()
            .check_dims("means", &means, &["D".into(), 3.into()])
            .check_dims("log_scales", &log_scales, &["D".into(), 3.into()])
            .check_dims("quats", &quats, &["D".into(), 4.into()])
            .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
            .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

        // Divide screen into tiles.
        let tile_bounds = calc_tile_bounds(img_size);

        // Tile rendering setup.
        let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape.dims[1] as u32);
        let total_splats = means.shape.dims[0];

        let project_uniforms = shaders::helpers::ProjectUniforms {
            viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            focal: camera.focal(img_size).into(),
            pixel_center: camera.center(img_size).into(),
            img_size: img_size.into(),
            tile_bounds: tile_bounds.into(),
            sh_degree,
            total_splats: total_splats as u32,
            pad_a: 0,
            pad_b: 0,
        };

        // Separate buffer for num_visible (written atomically by ProjectSplats)
        let num_visible_buffer = Self::int_zeros([1].into(), device, IntDType::U32);

        let client = &means.client.clone();
        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        // Step 1: ProjectSplats - culling pass
        let global_from_compact_gid = {
            let global_from_presort_gid =
                Self::int_zeros([total_splats].into(), device, IntDType::U32);
            let depths = create_tensor([total_splats], device, DType::F32);

            tracing::trace_span!("ProjectSplats").in_scope(||
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
            client.launch_unchecked(
                ProjectSplats::task(mip_splat),
                calc_cube_count_1d(total_splats as u32, ProjectSplats::WORKGROUP_SIZE[0]),
                Bindings::new()
                    .with_buffers(vec![
                        means.handle.clone().binding(),
                        quats.handle.clone().binding(),
                        log_scales.handle.clone().binding(),
                        raw_opacities.handle.clone().binding(),
                        global_from_presort_gid.handle.clone().binding(),
                        depths.handle.clone().binding(),
                        num_visible_buffer.handle.clone().binding(),
                    ])
                    .with_metadata(create_meta_binding(project_uniforms)),
            ).expect("Failed to render splats");
        });

            let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort").in_scope(|| {
                radix_argsort(
                    depths,
                    global_from_presort_gid,
                    32,
                    Some(num_visible_buffer.clone()),
                )
            });

            global_from_compact_gid
        };

        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / size_of::<f32>();
        let projected_splats = create_tensor([total_splats, proj_size], device, DType::F32);
        let splat_intersect_counts = Self::int_zeros([total_splats].into(), device, IntDType::U32);

        tracing::trace_span!("ProjectVisibleWithCounting").in_scope(|| {
            let num_vis_wg = create_dispatch_buffer_1d(
                num_visible_buffer.clone(),
                ProjectVisible::WORKGROUP_SIZE[0],
            );
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        ProjectVisible::task(mip_splat),
                        CubeCount::Dynamic(num_vis_wg.handle.binding()),
                        Bindings::new()
                            .with_buffers(vec![
                                num_visible_buffer.handle.clone().binding(),
                                means.handle.binding(),
                                log_scales.handle.binding(),
                                quats.handle.binding(),
                                sh_coeffs.handle.binding(),
                                raw_opacities.handle.binding(),
                                global_from_compact_gid.handle.clone().binding(),
                                projected_splats.handle.clone().binding(),
                                splat_intersect_counts.handle.clone().binding(),
                            ])
                            .with_metadata(create_meta_binding(project_uniforms)),
                    )
                    .expect("Failed to render splats");
            }
        });

        let cum_tiles_hit = tracing::trace_span!("PrefixSumGaussHits")
            .in_scope(|| prefix_sum(splat_intersect_counts));

        ProjectOutput {
            projected_splats,
            project_uniforms,
            num_visible: num_visible_buffer,
            global_from_compact_gid,
            cum_tiles_hit,
            img_size,
        }
    }

    fn rasterize(
        project_output: &ProjectOutput<Self>,
        num_intersections: u32,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>, IntTensor<Self>) {
        let _span = tracing::trace_span!("rasterize").entered();

        let device = &project_output.projected_splats.device.clone();
        let client = project_output.projected_splats.client.clone();
        let img_size = project_output.img_size;

        // Divide screen into tiles.
        let tile_bounds = calc_tile_bounds(img_size);
        let num_tiles = tile_bounds.x * tile_bounds.y;

        let rasterize_uniforms = shaders::helpers::RasterizeUniforms {
            tile_bounds: tile_bounds.into(),
            img_size: img_size.into(),
            background: [background.x, background.y, background.z, 1.0],
        };

        // Step 1: Allocate intersection buffers with exact size (minimum 1 to avoid zero-size allocation)
        let buffer_size = (num_intersections as usize).max(1);
        let tile_id_from_isect = create_tensor([buffer_size], device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], device, DType::U32);

        // Step 2: MapGaussiansToIntersect (fill pass)
        let num_vis_map_wg = create_dispatch_buffer_1d(
            project_output.num_visible.clone(),
            MapGaussiansToIntersect::WORKGROUP_SIZE[0],
        );

        let map_uniforms = shaders::map_gaussians_to_intersect::Uniforms {
            tile_bounds: tile_bounds.into(),
        };

        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        MapGaussiansToIntersect::task(),
                        CubeCount::Dynamic(num_vis_map_wg.handle.clone().binding()),
                        Bindings::new()
                            .with_buffers(vec![
                                project_output.num_visible.handle.clone().binding(),
                                project_output.projected_splats.handle.clone().binding(),
                                project_output.cum_tiles_hit.handle.clone().binding(),
                                tile_id_from_isect.handle.clone().binding(),
                                compact_gid_from_isect.handle.clone().binding(),
                            ])
                            .with_metadata(create_meta_binding(map_uniforms)),
                    )
                    .expect("Failed to render splats");
            }
        });

        // Step 3: Tile sort - use static dispatch with full buffer (num_intersections)
        let bits = u32::BITS - num_tiles.leading_zeros();

        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits, None));

        // Step 4: GetTileOffsets
        let cube_dim = CubeDim::new_1d(256);

        let tile_offsets = Self::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            device,
            IntDType::U32,
        );

        // Create a tensor for num_intersections
        let num_inter_tensor = {
            let data: [u32; 1] = [num_intersections];
            CubeTensor::new_contiguous(
                client.clone(),
                device.clone(),
                Shape::new([1]),
                client.create_from_slice(bytemuck::cast_slice(&data)),
                DType::U32,
            )
        };

        // SAFETY: Safe kernel.
        unsafe {
            get_tile_offsets::launch_unchecked::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_intersections, cube_dim.x * CHECKS_PER_ITER),
                cube_dim,
                tile_id_from_isect.as_tensor_arg(1),
                tile_offsets.as_tensor_arg(1),
                num_inter_tensor.as_tensor_arg(1),
            )
            .expect("Failed to render splats");
        }

        // Step 5: Rasterize
        let out_dim = if bwd_info { 4 } else { 1 };
        let out_img = create_tensor(
            [img_size.y as usize, img_size.x as usize, out_dim],
            device,
            DType::F32,
        );

        // Get total_splats from the shape of projected_splats
        let total_splats = project_output.projected_splats.shape.dims[0];

        let (bindings, visible) = if bwd_info {
            let visible = Self::float_zeros([total_splats].into(), device, FloatDType::F32);
            let bindings = Bindings::new()
                .with_buffers(vec![
                    compact_gid_from_isect.handle.clone().binding(),
                    tile_offsets.handle.clone().binding(),
                    project_output.projected_splats.handle.clone().binding(),
                    out_img.handle.clone().binding(),
                    project_output
                        .global_from_compact_gid
                        .handle
                        .clone()
                        .binding(),
                    visible.handle.clone().binding(),
                ])
                .with_metadata(create_meta_binding(rasterize_uniforms));
            (bindings, visible)
        } else {
            let bindings = Bindings::new()
                .with_buffers(vec![
                    compact_gid_from_isect.handle.clone().binding(),
                    tile_offsets.handle.clone().binding(),
                    project_output.projected_splats.handle.clone().binding(),
                    out_img.handle.clone().binding(),
                ])
                .with_metadata(create_meta_binding(rasterize_uniforms));
            (bindings, create_tensor([1], device, DType::F32))
        };

        let raster_task = Rasterize::task(bwd_info);

        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client
                .launch_unchecked(
                    raster_task,
                    CubeCount::Static(tile_bounds.x * tile_bounds.y, 1, 1),
                    bindings,
                )
                .expect("Failed to render splats");
        }

        (
            out_img,
            RenderAux {
                num_visible: project_output.num_visible.clone(),
                num_intersections,
                visible,
                tile_offsets,
                img_size: project_output.img_size,
            },
            compact_gid_from_isect,
        )
    }
}
