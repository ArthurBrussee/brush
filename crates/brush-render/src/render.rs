use crate::{
    MainBackendBase, SplatForward, SplatProjectPrepare, SplatRasterize,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::SplatRenderMode,
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    render_aux::{ProjectAux, RasterizeAux, RenderAux},
    sh::sh_degree_from_coeffs,
    shaders::{self, MapGaussiansToIntersect, ProjectSplats, ProjectVisible, Rasterize},
};
use brush_kernel::create_dispatch_buffer_1d;
use brush_kernel::create_tensor;
use brush_kernel::create_uniform_buffer;
use brush_kernel::{CubeCount, calc_cube_count_1d};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::{DType, IntDType, ops::FloatTensor};
use burn::tensor::{
    FloatDType,
    ops::{FloatTensorOps, IntTensorOps},
};
use burn_cubecl::cubecl::server::Bindings;

use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use burn_wgpu::CubeDim;
use glam::{Vec3, uvec2};
use std::mem::offset_of;

pub(crate) fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

// Implement the first pass: ProjectPrepare
impl SplatProjectPrepare<Self> for MainBackendBase {
    fn project_prepare(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
    ) -> ProjectAux<Self> {
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

        let uniforms = shaders::helpers::RenderUniforms {
            viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            focal: camera.focal(img_size).into(),
            pixel_center: camera.center(img_size).into(),
            img_size: img_size.into(),
            tile_bounds: tile_bounds.into(),
            sh_degree,
            total_splats: total_splats as u32,
            background: [background.x, background.y, background.z, 1.0],
            num_visible: 0,
            pad_a: 0,
        };

        let uniforms_buffer = create_uniform_buffer(uniforms, device, &client);

        let client = &means.client.clone();
        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        // Step 1: ProjectSplats - culling pass
        let (global_from_compact_gid, num_visible) = {
            let global_from_presort_gid =
                Self::int_zeros([total_splats].into(), device, IntDType::U32);
            let depths = create_tensor([total_splats], device, DType::F32);

            tracing::trace_span!("ProjectSplats").in_scope(||
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
            client.launch_unchecked(
                ProjectSplats::task(mip_splat),
                calc_cube_count_1d(total_splats as u32, ProjectSplats::WORKGROUP_SIZE[0]),
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
            ).expect("Failed to render splats");
        });

            // Get just the number of visible splats from the uniforms buffer.
            let num_vis_field_offset =
                offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
            let num_visible = Self::int_slice(
                uniforms_buffer.clone(),
                &[(num_vis_field_offset..num_vis_field_offset + 1).into()],
            );

            // Step 2: DepthSort
            let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort").in_scope(|| {
                radix_argsort(depths, global_from_presort_gid, &num_visible, 32)
            });

            (global_from_compact_gid, num_visible)
        };

        // Step 3: ProjectVisible with intersection counting
        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / size_of::<f32>();
        let projected_splats = create_tensor([total_splats, proj_size], device, DType::F32);
        let splat_intersect_counts =
            Self::int_zeros([total_splats + 1].into(), device, IntDType::U32);

        tracing::trace_span!("ProjectVisibleWithCounting").in_scope(|| {
            let num_vis_wg =
                create_dispatch_buffer_1d(num_visible.clone(), ProjectVisible::WORKGROUP_SIZE[0]);
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        ProjectVisible::task(mip_splat, true), // count_intersections = true
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
                            splat_intersect_counts.handle.clone().binding(),
                        ]),
                    )
                    .expect("Failed to render splats");
            }
        });

        // Step 4: PrefixSum to get cumulative tile hits
        let cum_tiles_hit = tracing::trace_span!("PrefixSumGaussHits")
            .in_scope(|| prefix_sum(splat_intersect_counts));

        // Sanity check
        assert!(
            uniforms_buffer.is_contiguous(),
            "Uniforms must be contiguous"
        );
        assert!(
            global_from_compact_gid.is_contiguous(),
            "Global from compact gid must be contiguous"
        );
        assert!(
            projected_splats.is_contiguous(),
            "Projected splats must be contiguous"
        );

        ProjectAux {
            projected_splats,
            uniforms_buffer,
            global_from_compact_gid,
            cum_tiles_hit,
            img_size,
        }
    }
}

// Implement the second pass: Rasterize
impl SplatRasterize<Self> for MainBackendBase {
    fn rasterize(
        project_aux: &ProjectAux<Self>,
        num_intersections: u32,
        _background: Vec3, // Background is read from uniforms_buffer
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RasterizeAux<Self>) {
        let _span = tracing::trace_span!("rasterize").entered();

        let device = &project_aux.projected_splats.device.clone();
        let client = project_aux.projected_splats.client.clone();
        let img_size = project_aux.img_size;

        // Divide screen into tiles.
        let tile_bounds = calc_tile_bounds(img_size);
        let num_tiles = tile_bounds.x * tile_bounds.y;

        // Get num_visible from uniforms buffer
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        let num_visible = Self::int_slice(
            project_aux.uniforms_buffer.clone(),
            &[(num_vis_field_offset..num_vis_field_offset + 1).into()],
        );


        // Step 1: Allocate intersection buffers with exact size (minimum 1 to avoid zero-size allocation)
        let buffer_size = (num_intersections as usize).max(1);
        let tile_id_from_isect = create_tensor([buffer_size], device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], device, DType::U32);

        // Step 2: MapGaussiansToIntersect (fill pass, prepass=false)
        let num_vis_map_wg =
            create_dispatch_buffer_1d(num_visible.clone(), MapGaussiansToIntersect::WORKGROUP_SIZE[0]);

        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        MapGaussiansToIntersect::task(false),
                        CubeCount::Dynamic(num_vis_map_wg.handle.clone().binding()),
                        Bindings::new().with_buffers(vec![
                            project_aux.uniforms_buffer.handle.clone().binding(),
                            project_aux.projected_splats.handle.clone().binding(),
                            project_aux.cum_tiles_hit.handle.clone().binding(),
                            tile_id_from_isect.handle.clone().binding(),
                            compact_gid_from_isect.handle.clone().binding(),
                        ]),
                    )
                    .expect("Failed to render splats");
            }
        });

        // Step 3: Tile sort - use static dispatch with actual num_intersections
        let bits = u32::BITS - num_tiles.leading_zeros();

        // Create a tensor holding num_intersections for the sort
        // Get the last element from cum_tiles_hit
        let cum_len = project_aux.cum_tiles_hit.shape[0];
        let num_intersections_tensor = Self::int_slice(
            project_aux.cum_tiles_hit.clone(),
            &[(cum_len - 1..cum_len).into()],
        );

        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| {
                radix_argsort(
                    tile_id_from_isect,
                    compact_gid_from_isect,
                    &num_intersections_tensor,
                    bits,
                )
            });

        // Step 4: GetTileOffsets
        let cube_dim = CubeDim::new_1d(256);
        let num_vis_map_wg =
            create_dispatch_buffer_1d(num_intersections_tensor.clone(), 256 * CHECKS_PER_ITER);
        let cube_count = CubeCount::Dynamic(num_vis_map_wg.handle.binding());

        let tile_offsets = Self::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            device,
            IntDType::U32,
        );

        // SAFETY: Safe kernel.
        unsafe {
            get_tile_offsets::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                tile_id_from_isect.as_tensor_arg(1),
                tile_offsets.as_tensor_arg(1),
                num_intersections_tensor.as_tensor_arg(1),
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

        // Update background in uniforms - we need to create a modified uniforms buffer
        // For now, we'll pass the background through the existing buffer structure
        // The rasterize kernel reads background from uniforms, so we need to ensure it's set

        let mut bindings = Bindings::new().with_buffers(vec![
            project_aux.uniforms_buffer.handle.clone().binding(),
            compact_gid_from_isect.handle.clone().binding(),
            tile_offsets.handle.clone().binding(),
            project_aux.projected_splats.handle.clone().binding(),
            out_img.handle.clone().binding(),
        ]);

        // Get total_splats from the shape of projected_splats
        let total_splats = project_aux.projected_splats.shape.dims[0];

        let visible = if bwd_info {
            let visible = Self::float_zeros([total_splats].into(), device, FloatDType::F32);
            bindings = bindings.with_buffers(vec![
                project_aux.global_from_compact_gid.handle.clone().binding(),
                visible.handle.clone().binding(),
            ]);
            visible
        } else {
            create_tensor([1], device, DType::F32)
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

        // Sanity checks
        assert!(tile_offsets.is_contiguous(), "Tile offsets must be contiguous");
        assert!(visible.is_contiguous(), "Visible must be contiguous");

        (
            out_img,
            RasterizeAux {
                tile_offsets,
                compact_gid_from_isect,
                visible,
            },
        )
    }
}

// Implement backwards-compatible render_splats using the split pipeline
impl SplatForward<Self> for MainBackendBase {
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>) {
        // First pass: project and prepare (includes background in uniforms)
        let project_aux = Self::project_prepare(
            camera,
            img_size,
            means,
            log_scales,
            quats,
            sh_coeffs,
            raw_opacities,
            render_mode,
            background,
        );

        // Sync readback of num_intersections
        #[cfg(not(target_family = "wasm"))]
        let num_intersections = project_aux.num_intersections();

        #[cfg(target_family = "wasm")]
        let num_intersections = {
            // On wasm, estimate max intersections
            let tile_bounds = calc_tile_bounds(img_size);
            let num_tiles = tile_bounds[0] * tile_bounds[1];
            let total_splats = project_aux.projected_splats.shape.dims[0] as u32;
            let max_possible = num_tiles.saturating_mul(total_splats);
            max_possible.min(2 * 512 * 65535)
        };

        // Second pass: rasterize
        let (out_img, rasterize_aux) =
            Self::rasterize(&project_aux, num_intersections, background, bwd_info);

        // Combine into RenderAux for backwards compatibility
        let render_aux = RenderAux::from_parts(project_aux, rasterize_aux);

        (out_img, render_aux)
    }
}
