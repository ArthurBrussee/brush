//! Forward pipeline for rasterizing per-splat feature vectors (`DiG`).
//!
//! Mirrors [`crate::render`]'s cull → sort → project → rasterize
//! orchestration, reusing the same kernels for everything up to the
//! rasterize itself, which is swapped for
//! [`crate::kernels::rasterize_features`]. Kept separate from the RGB
//! path because the feature pass runs at its own (lower) resolution and
//! treats geometry as a constant — `DiG` detaches means/rotations/scales/
//! opacities for feature supervision, so no projection gradients are
//! ever needed here.

use crate::camera::calculate_jacobian_clamp_limits;
use crate::{
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::SplatRenderMode,
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    kernels,
    render::calc_tile_bounds,
    shaders,
};
use brush_cube::create_tensor;
use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::backend::TensorMetadata;
use burn::backend::ops::TransactionPrimitive;
use burn::backend::ops::{FloatTensorOps, IntTensorOps, TransactionOps};
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::tensor::{DType, FloatDType, IntDType};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use kernels::types::RasterizeUniformsLaunch;
use std::f32::consts::PI;

/// Everything the feature backward needs to replay the forward walk.
#[derive(Debug, Clone)]
pub struct FeatureRenderOutput<B: burn::backend::Backend> {
    /// `[img_h, img_w, feat_dim + 1]`: composited features + alpha.
    pub out_img: FloatTensor<B>,
    /// `[num_visible, PROJECTED_LANES]` projected splats.
    pub projected_splats: FloatTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub num_visible: u32,
}

/// Rasterize `[N, feat_dim]` per-splat features to `[h, w, feat_dim + 1]`
/// (features + alpha). Geometry inputs are constants; the only
/// differentiable input is `features` (handled by brush-render-bwd).
pub async fn render_features_base(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<MainBackendBase>,
    raw_opacities: FloatTensor<MainBackendBase>,
    features: FloatTensor<MainBackendBase>,
    render_mode: SplatRenderMode,
) -> FeatureRenderOutput<MainBackendBase> {
    assert!(
        img_size[0] > 0 && img_size[1] > 0,
        "Can't render images with 0 size."
    );

    let transforms = into_contiguous(transforms);
    let raw_opacities = into_contiguous(raw_opacities);
    let features = into_contiguous(features);

    DimCheck::new()
        .check_dims("transforms", &transforms, &["D".into(), 10.into()])
        .check_dims("raw_opacities", &raw_opacities, &["D".into()])
        .check_dims("features", &features, &["D".into(), "C".into()]);

    let total_splats = transforms.shape()[0] as u32;
    let feat_dim = features.shape()[1];
    let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

    let half_max_render_fov =
        ((camera.fov_x as f32).hypot(camera.fov_y as f32) * 1.05).min(2.0 * PI - 1e-6) * 0.5;
    let pinhole_params = camera.build_pinhole_params(img_size);

    let mut project_uniforms = shaders::helpers::ProjectUniforms {
        viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
        camera_model: camera.camera_model,
        half_max_render_fov,
        pinhole_params,
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        img_size: img_size.into(),
        tile_bounds: calc_tile_bounds(img_size).into(),
        sh_degree: 0,
        total_splats,
        num_visible: 0,
        jacobian_clamp_limits: calculate_jacobian_clamp_limits(
            img_size,
            pinhole_params,
            camera.camera_model,
        ),
    };

    let device = transforms.device.clone();
    let client = transforms.client.clone();

    let num_visible_buf = MainBackendBase::int_zeros([1].into(), &device, IntDType::U32);
    let num_intersections_buf = MainBackendBase::int_zeros([1].into(), &device, IntDType::U32);
    let intersect_counts =
        MainBackendBase::int_zeros([total_splats as usize].into(), &device, IntDType::U32);
    let max_radius =
        MainBackendBase::float_zeros([total_splats as usize].into(), &device, FloatDType::F32);
    let global_from_presort_gid = create_tensor([total_splats as usize], &device, DType::U32);
    let depths = create_tensor([total_splats as usize], &device, DType::F32);

    tracing::trace_span!("ProjectSplats (features)").in_scope(|| {
        let uniforms = project_uniforms.to_launch_object();
        kernels::project_forward::project_forward_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(total_splats, kernels::project_forward::WG_SIZE),
            CubeDim::new_1d(kernels::project_forward::WG_SIZE),
            transforms.clone().into_tensor_arg(),
            raw_opacities.clone().into_tensor_arg(),
            global_from_presort_gid.clone().into_tensor_arg(),
            depths.clone().into_tensor_arg(),
            num_visible_buf.clone().into_tensor_arg(),
            intersect_counts.clone().into_tensor_arg(),
            num_intersections_buf.clone().into_tensor_arg(),
            max_radius.into_tensor_arg(),
            uniforms,
            mip_splat,
            camera.camera_model,
        );
    });

    let (num_visible, num_intersections) = if total_splats == 0 {
        (0, 0)
    } else {
        let tp = TransactionPrimitive::<MainBackendBase>::new(
            vec![],
            vec![],
            vec![num_visible_buf, num_intersections_buf],
            vec![],
        );
        let data = <MainBackendBase as TransactionOps<MainBackendBase>>::tr_execute(tp)
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
    let tile_bounds: glam::UVec2 = project_uniforms.tile_bounds.into();
    let num_visible_sz = (num_visible as usize).max(1);

    let global_from_compact_gid = {
        let depths = MainBackendBase::float_slice(depths, &[(0..num_visible_sz).into()]);
        let global_from_presort_gid =
            MainBackendBase::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);
        let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort (features)")
            .in_scope(|| radix_argsort(depths, global_from_presort_gid, 32));
        global_from_compact_gid
    };
    let compact_counts =
        MainBackendBase::int_gather(0, intersect_counts, global_from_compact_gid.clone());
    let cum_tiles_hit = prefix_sum(compact_counts);

    // The visible-projection kernel evaluates SH into the color lanes; the
    // feature pass never reads them, so feed a degree-0 all-zero SH tensor.
    let dummy_sh = MainBackendBase::float_zeros(
        [total_splats as usize, 1, 3].into(),
        &device,
        FloatDType::F32,
    );
    let projected_splats = create_tensor(
        [num_visible_sz, kernels::helpers::PROJECTED_LANES_USIZE],
        &device,
        DType::F32,
    );
    tracing::trace_span!("ProjectVisible (features)").in_scope(|| {
        let uniforms = project_uniforms.to_launch_object();
        kernels::project_visible::project_visible_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_visible, kernels::project_visible::WG_SIZE),
            CubeDim::new_1d(kernels::project_visible::WG_SIZE),
            transforms.into_tensor_arg(),
            dummy_sh.into_tensor_arg(),
            raw_opacities.into_tensor_arg(),
            global_from_compact_gid.clone().into_tensor_arg(),
            projected_splats.clone().into_tensor_arg(),
            uniforms,
            mip_splat,
            0,
            camera.camera_model,
        );
    });

    let num_tiles = tile_bounds.x * tile_bounds.y;
    let buffer_size = (num_intersections as usize).max(1);
    let tile_id_from_isect = create_tensor([buffer_size], &device, DType::U32);
    let compact_gid_from_isect = create_tensor([buffer_size], &device, DType::U32);
    tracing::trace_span!("MapGaussiansToIntersect (features)").in_scope(|| {
        kernels::map_gaussians::map_gaussians_to_intersect_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_visible, kernels::map_gaussians::WG_SIZE),
            CubeDim::new_1d(kernels::map_gaussians::WG_SIZE),
            projected_splats.clone().into_tensor_arg(),
            cum_tiles_hit.into_tensor_arg(),
            tile_id_from_isect.clone().into_tensor_arg(),
            compact_gid_from_isect.clone().into_tensor_arg(),
            project_uniforms.tile_bounds[0],
            project_uniforms.tile_bounds[1],
            num_visible,
        );
    });
    let bits = u32::BITS - num_tiles.leading_zeros();
    let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort (features)")
        .in_scope(|| radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits));
    let cube_dim = CubeDim::new_1d(256);
    let tile_offsets = MainBackendBase::int_zeros(
        [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
        &device,
        IntDType::U32,
    );
    tracing::trace_span!("GetTileOffsets (features)").in_scope(|| {
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

    let out_img = create_tensor(
        [img_size.y as usize, img_size.x as usize, feat_dim + 1],
        &device,
        DType::F32,
    );
    tracing::trace_span!("RasterizeFeatures").in_scope(|| {
        let uniforms = RasterizeUniformsLaunch::new(
            project_uniforms.tile_bounds[0],
            project_uniforms.img_size[0],
            project_uniforms.img_size[1],
            0.0,
            0.0,
            0.0,
        );
        kernels::rasterize_features::rasterize_features_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(
                num_tiles * (shaders::helpers::TILE_WIDTH * shaders::helpers::TILE_WIDTH),
                shaders::helpers::TILE_WIDTH * shaders::helpers::TILE_WIDTH,
            ),
            CubeDim::new_1d(shaders::helpers::TILE_SIZE),
            compact_gid_from_isect.clone().into_tensor_arg(),
            tile_offsets.clone().into_tensor_arg(),
            projected_splats.clone().into_tensor_arg(),
            features.into_tensor_arg(),
            global_from_compact_gid.clone().into_tensor_arg(),
            out_img.clone().into_tensor_arg(),
            uniforms,
            feat_dim,
        );
    });

    FeatureRenderOutput {
        out_img,
        projected_splats,
        compact_gid_from_isect,
        tile_offsets,
        global_from_compact_gid,
        num_visible,
    }
}
