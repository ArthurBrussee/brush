use crate::camera::calculate_jacobian_clamp_limits;
use crate::{
    RenderAuxInner, SplatOps,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::{RasterPass, SplatRenderMode},
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    kernels,
    render_aux::RenderOutput,
    sh::sh_degree_from_coeffs,
    shaders,
};
use brush_cube::create_tensor;
use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::backend::TensorMetadata;
use burn::backend::ops::TransactionPrimitive;
use burn::backend::ops::{FloatTensorOps, IntTensorOps, TransactionOps};
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, FloatDType, IntDType};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use glam::{Vec3, uvec2};
use kernels::types::RasterizeUniformsLaunch;
use std::f32::consts::PI;

#[doc(hidden)]
pub fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
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
        pass: RasterPass,
    ) -> RenderOutput<Self> {
        assert!(
            img_size[0] > 0 && img_size[1] > 0,
            "Can't render images with 0 size."
        );
        let bwd_info = pass.bwd_info();
        let smooth_cutoff = pass.smooth_cutoff();

        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opacities = into_contiguous(raw_opacities);

        DimCheck::new()
            .check_dims("transforms", &transforms, &["D".into(), 10.into()])
            .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
            .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

        let total_splats = transforms.shape()[0] as u32;
        let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape()[1] as u32);
        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        let half_max_render_fov =
            ((camera.fov_x as f32).hypot(camera.fov_y as f32) * 1.05).min(2.0 * PI - 1e-6) * 0.5;
        let pinhole_params = camera.build_pinhole_params(img_size);

        let (screen_area_penalty, screen_area_threshold) = crate::screen_area_penalty_config();
        let mut project_uniforms = shaders::helpers::ProjectUniforms {
            viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
            camera_model: camera.camera_model,
            half_max_render_fov,
            pinhole_params,
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            img_size: img_size.into(),
            tile_bounds: calc_tile_bounds(img_size).into(),
            sh_degree,
            total_splats,
            num_visible: 0, // num_visible — not yet known.
            screen_area_penalty,
            screen_area_threshold,
            jacobian_clamp_limits: calculate_jacobian_clamp_limits(
                img_size,
                pinhole_params,
                camera.camera_model,
            ),
        };

        let device = transforms.device.clone();
        let client = transforms.client.clone();

        let (
            global_from_presort_gid,
            depths,
            intersect_counts,
            max_radius,
            num_visible_buf,
            num_intersections_buf,
        ) = {
            let project_uniforms: &shaders::helpers::ProjectUniforms = &project_uniforms;
            let _span = tracing::trace_span!("ProjectSplats").entered();

            let total_splats = project_uniforms.total_splats as usize;
            let num_visible_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let num_intersections_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let intersect_counts = Self::int_zeros([total_splats].into(), &device, IntDType::U32);
            let max_radius = Self::float_zeros([total_splats].into(), &device, FloatDType::F32);

            let global_from_presort_gid = create_tensor([total_splats], &device, DType::U32);
            let depths = create_tensor([total_splats], &device, DType::F32);

            let uniforms = project_uniforms.to_launch_object();

            kernels::project_forward::project_forward_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(
                    project_uniforms.total_splats,
                    kernels::project_forward::WG_SIZE,
                ),
                CubeDim::new_1d(kernels::project_forward::WG_SIZE),
                transforms.clone().into_tensor_arg(),
                raw_opacities.clone().into_tensor_arg(),
                global_from_presort_gid.clone().into_tensor_arg(),
                depths.clone().into_tensor_arg(),
                num_visible_buf.clone().into_tensor_arg(),
                intersect_counts.clone().into_tensor_arg(),
                num_intersections_buf.clone().into_tensor_arg(),
                max_radius.clone().into_tensor_arg(),
                uniforms,
                mip_splat,
                camera.camera_model,
            );
            (
                global_from_presort_gid,
                depths,
                intersect_counts,
                max_radius,
                num_visible_buf,
                num_intersections_buf,
            )
        };

        // Read both atomic counts in one transaction BEFORE the sort.
        let (num_visible, num_intersections) = if total_splats == 0 {
            (0, 0)
        } else {
            let tp = TransactionPrimitive::<Self>::new(
                vec![],
                vec![],
                vec![num_visible_buf, num_intersections_buf],
                vec![],
            );
            let data = <Self as TransactionOps<Self>>::tr_execute(tp)
                .await
                .expect("Failed to read counts");
            let num_visible = data.read_ints[0]
                .clone()
                .into_vec::<u32>()
                .expect("num_visible")[0];
            let num_intersections = data.read_ints[1]
                .clone()
                .into_vec::<u32>()
                .expect("num_intersections")[0];
            (num_visible, num_intersections)
        };

        project_uniforms.num_visible = num_visible;

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);
        let img_size: glam::UVec2 = project_uniforms.img_size.into();
        let tile_bounds: glam::UVec2 = project_uniforms.tile_bounds.into();
        let num_visible_sz = (num_visible as usize).max(1);

        let global_from_compact_gid = {
            let depths = Self::float_slice(depths, &[(0..num_visible_sz).into()]);
            let global_from_presort_gid =
                Self::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);
            let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort")
                .in_scope(|| radix_argsort(depths, global_from_presort_gid, 32));
            global_from_compact_gid
        };
        let compact_counts = Self::int_gather(0, intersect_counts, global_from_compact_gid.clone());
        let cum_tiles_hit =
            tracing::trace_span!("PrefixSumGaussHits").in_scope(|| prefix_sum(compact_counts));
        let projected_splats = create_tensor(
            [num_visible_sz, kernels::helpers::PROJECTED_LANES_USIZE],
            &device,
            DType::F32,
        );
        tracing::trace_span!("ProjectVisible").in_scope(|| {
            let uniforms = project_uniforms.to_launch_object();
            kernels::project_visible::project_visible_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, kernels::project_visible::WG_SIZE),
                CubeDim::new_1d(kernels::project_visible::WG_SIZE),
                transforms.into_tensor_arg(),
                sh_coeffs.into_tensor_arg(),
                raw_opacities.into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                uniforms,
                mip_splat,
                sh_degree,
                camera.camera_model,
            );
        });
        let num_tiles = tile_bounds.x * tile_bounds.y;
        let buffer_size = (num_intersections as usize).max(1);
        let tile_id_from_isect = create_tensor([buffer_size], &device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], &device, DType::U32);
        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            kernels::map_gaussians::map_gaussians_to_intersect_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, kernels::map_gaussians::WG_SIZE),
                CubeDim::new_1d(kernels::map_gaussians::WG_SIZE),
                projected_splats.clone().into_tensor_arg(),
                cum_tiles_hit.clone().into_tensor_arg(),
                tile_id_from_isect.clone().into_tensor_arg(),
                compact_gid_from_isect.clone().into_tensor_arg(),
                project_uniforms.tile_bounds[0],
                project_uniforms.tile_bounds[1],
                num_visible,
            );
        });
        let bits = u32::BITS - num_tiles.leading_zeros();
        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits));
        let cube_dim = CubeDim::new_1d(256);
        let tile_offsets = Self::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            &device,
            IntDType::U32,
        );
        tracing::trace_span!("GetTileOffsets").in_scope(|| {
            get_tile_offsets::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_intersections, cube_dim.x * CHECKS_PER_ITER),
                cube_dim,
                num_intersections,
                num_tiles,
                tile_id_from_isect.into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
            );
        });
        let out_dim = if bwd_info { 4 } else { 1 };
        let out_img = create_tensor(
            [img_size.y as usize, img_size.x as usize, out_dim],
            &device,
            DType::F32,
        );
        let (out_packed_arg, out_f32_arg) = if bwd_info {
            (create_tensor([1], &device, DType::U32), out_img.clone())
        } else {
            (out_img.clone(), create_tensor([1], &device, DType::F32))
        };
        let total_splats = project_uniforms.total_splats as usize;
        let visible = if bwd_info {
            Self::float_zeros([total_splats].into(), &device, FloatDType::F32)
        } else {
            // Zero-init the dummy — `create_tensor` doesn't initialise, and
            // validate() may read this tensor to check its invariants.
            // Using `float_zeros` makes that read a well-defined no-op.
            Self::float_zeros([1].into(), &device, FloatDType::F32)
        };
        tracing::trace_span!("Rasterize").in_scope(|| {
            let uniforms = RasterizeUniformsLaunch::new(
                project_uniforms.tile_bounds[0],
                project_uniforms.img_size[0],
                project_uniforms.img_size[1],
                background.x,
                background.y,
                background.z,
            );
            kernels::rasterize::rasterize_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(
                    num_tiles * (shaders::helpers::TILE_WIDTH * shaders::helpers::TILE_WIDTH),
                    shaders::helpers::TILE_WIDTH * shaders::helpers::TILE_WIDTH,
                ),
                CubeDim::new_1d(shaders::helpers::TILE_SIZE),
                compact_gid_from_isect.clone().into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                out_packed_arg.into_tensor_arg(),
                out_f32_arg.into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                visible.clone().into_tensor_arg(),
                uniforms,
                bwd_info,
                smooth_cutoff,
            );
        });
        RenderOutput {
            out_img,
            aux: RenderAuxInner {
                num_visible,
                num_intersections,
                visible,
                max_radius,
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
