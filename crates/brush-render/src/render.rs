use crate::{
    MainBackendBase, RenderAux, SplatOps,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::SplatRenderMode,
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    render_aux::RenderOutput,
    sh::sh_degree_from_coeffs,
    shaders::{self, MapGaussiansToIntersect, ProjectSplats, ProjectVisible, Rasterize},
};
use brush_kernel::bytemuck;
use brush_kernel::calc_cube_count_1d;
use brush_kernel::create_meta_binding;
use brush_kernel::create_tensor;
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::ops::{FloatTensor, FloatTensorOps, IntTensorOps};
use burn::tensor::{DType, FloatDType, Int, IntDType, Shape, Tensor, TensorMetadata, Transaction};
use burn_cubecl::cubecl::server::KernelArguments;
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
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        bwd_info: bool,
    ) -> RenderOutput<Self> {
        assert!(
            img_size[0] > 0 && img_size[1] > 0,
            "Can't render images with 0 size."
        );

        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opacities = into_contiguous(raw_opacities);

        let device = &transforms.device.clone();
        let client = transforms.client.clone();

        DimCheck::new()
            .check_dims("transforms", &transforms, &["D".into(), 10.into()])
            .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

        let tile_bounds = calc_tile_bounds(img_size);
        let total_splats = transforms.shape()[0];

        let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape()[1] as u32);
        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        let mut project_uniforms = shaders::helpers::ProjectUniforms {
            viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            focal: camera.focal(img_size).into(),
            pixel_center: camera.center(img_size).into(),
            img_size: img_size.into(),
            tile_bounds: tile_bounds.into(),
            sh_degree,
            total_splats: total_splats as u32,
            num_visible: 0,
            pad_a: 0,
        };

        let num_visible_buffer = Self::int_zeros([1].into(), device, IntDType::U32);
        let num_intersections_buffer = Self::int_zeros([1].into(), device, IntDType::U32);
        let intersect_counts = Self::int_zeros([total_splats].into(), device, IntDType::U32);

        // Phase 1: Project & cull. Writes num_visible, num_intersections atomically.
        let global_from_presort_gid = create_tensor([total_splats], device, DType::U32);
        let depths = create_tensor([total_splats], device, DType::F32);

        tracing::trace_span!("ProjectSplats").in_scope(||
        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client.launch_unchecked(
                ProjectSplats::task(mip_splat),
                calc_cube_count_1d(total_splats as u32, ProjectSplats::WORKGROUP_SIZE[0]),
                KernelArguments::new()
                    .with_buffers(vec![
                        transforms.handle.clone().binding(),
                        raw_opacities.handle.clone().binding(),
                        global_from_presort_gid.handle.clone().binding(),
                        depths.handle.clone().binding(),
                        num_visible_buffer.handle.clone().binding(),
                        intersect_counts.handle.clone().binding(),
                        num_intersections_buffer.handle.clone().binding(),
                    ])
                    .with_info(create_meta_binding(project_uniforms)),
            );
        });

        // Read both atomic counts in one transaction BEFORE the sort.
        let (num_visible, num_intersections) = if total_splats == 0 {
            (0, 0)
        } else {
            let data = Transaction::default()
                .register(Tensor::<Self, 1, Int>::from_primitive(num_visible_buffer))
                .register(Tensor::<Self, 1, Int>::from_primitive(
                    num_intersections_buffer,
                ))
                .execute_async()
                .await
                .expect("Failed to read counts");
            let num_visible = data[0].clone().into_vec::<u32>().expect("num_visible")[0];
            let num_intersections = data[1]
                .clone()
                .into_vec::<u32>()
                .expect("num_intersections")[0];
            (num_visible, num_intersections)
        };

        project_uniforms.num_visible = num_visible;
        let num_visible_sz = (num_visible as usize).max(1);

        let global_from_compact_gid = {
            // Depth sort only the valid [0..num_visible] entries.
            // Slice to exactly num_visible before sorting — no dynamic dispatch needed.
            let depths = Self::float_slice(depths, &[(0..num_visible_sz).into()]);
            let global_from_presort_gid =
                Self::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);

            let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort")
                .in_scope(|| radix_argsort(depths, global_from_presort_gid, 32));
            global_from_compact_gid
        };

        // Reorder intersection counts from global_gid to compact (depth-sorted) order.
        let compact_counts = Self::int_gather(0, intersect_counts, global_from_compact_gid.clone());

        let cum_tiles_hit =
            tracing::trace_span!("PrefixSumGaussHits").in_scope(|| prefix_sum(compact_counts));
        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / size_of::<f32>();
        let projected_splats = create_tensor([num_visible_sz, proj_size], device, DType::F32);

        tracing::trace_span!("ProjectVisible").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client.launch_unchecked(
                    ProjectVisible::task(mip_splat),
                    calc_cube_count_1d(num_visible, ProjectVisible::WORKGROUP_SIZE[0]),
                    KernelArguments::new()
                        .with_buffers(vec![
                            transforms.handle.clone().binding(),
                            sh_coeffs.handle.binding(),
                            raw_opacities.handle.binding(),
                            global_from_compact_gid.handle.clone().binding(),
                            projected_splats.handle.clone().binding(),
                        ])
                        .with_info(create_meta_binding(project_uniforms)),
                );
            }
        });

        let num_tiles = tile_bounds.x * tile_bounds.y;
        let buffer_size = (num_intersections as usize).max(1);
        let tile_id_from_isect = create_tensor([buffer_size], device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], device, DType::U32);

        let map_uniforms = shaders::map_gaussians_to_intersect::Uniforms {
            tile_bounds: tile_bounds.into(),
            num_visible,
            pad_a: 0,
        };

        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            client.launch(
                MapGaussiansToIntersect::task(),
                calc_cube_count_1d(num_visible, MapGaussiansToIntersect::WORKGROUP_SIZE[0]),
                KernelArguments::new()
                    .with_buffers(vec![
                        projected_splats.handle.clone().binding(),
                        cum_tiles_hit.handle.clone().binding(),
                        tile_id_from_isect.handle.clone().binding(),
                        compact_gid_from_isect.handle.clone().binding(),
                    ])
                    .with_info(create_meta_binding(map_uniforms)),
            );
        });

        // ---- Tile sort ----
        let bits = u32::BITS - num_tiles.leading_zeros();
        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits));

        // ---- GetTileOffsets ----
        let cube_dim = CubeDim::new_1d(256);
        let tile_offsets = Self::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            device,
            IntDType::U32,
        );

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
                tile_id_from_isect.into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
                num_inter_tensor.into_tensor_arg(),
            );
        }

        // ---- Rasterize ----
        let rasterize_uniforms = shaders::helpers::RasterizeUniforms {
            tile_bounds: tile_bounds.into(),
            img_size: img_size.into(),
            background: [background.x, background.y, background.z, 1.0],
        };

        let out_dim = if bwd_info { 4 } else { 1 };
        let out_img = create_tensor(
            [img_size.y as usize, img_size.x as usize, out_dim],
            device,
            DType::F32,
        );

        let total_splats = project_uniforms.total_splats as usize;
        let (bindings, visible) = if bwd_info {
            let visible = Self::float_zeros([total_splats].into(), device, FloatDType::F32);
            let bindings = KernelArguments::new()
                .with_buffers(vec![
                    compact_gid_from_isect.handle.clone().binding(),
                    tile_offsets.handle.clone().binding(),
                    projected_splats.handle.clone().binding(),
                    out_img.handle.clone().binding(),
                    global_from_compact_gid.handle.clone().binding(),
                    visible.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(rasterize_uniforms));
            (bindings, visible)
        } else {
            let bindings = KernelArguments::new()
                .with_buffers(vec![
                    compact_gid_from_isect.handle.clone().binding(),
                    tile_offsets.handle.clone().binding(),
                    projected_splats.handle.clone().binding(),
                    out_img.handle.clone().binding(),
                ])
                .with_info(create_meta_binding(rasterize_uniforms));
            (bindings, create_tensor([1], device, DType::F32))
        };

        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client.launch_unchecked(
                Rasterize::task(bwd_info),
                calc_cube_count_1d(
                    num_tiles * (shaders::helpers::TILE_WIDTH * shaders::helpers::TILE_WIDTH),
                    shaders::helpers::TILE_WIDTH * shaders::helpers::TILE_WIDTH,
                ),
                bindings,
            );
        }

        RenderOutput {
            out_img,
            aux: RenderAux {
                num_visible,
                num_intersections,
                visible,
                tile_offsets,
                img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            project_uniforms,
            global_from_compact_gid,
        }
    }
}
