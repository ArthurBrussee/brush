use brush_cube::calc_cube_count_1d;
use brush_render::MainBackendBase;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::kernels::types::RasterizeUniformsLaunch;
use brush_render::render::build_project_uniforms_launch;
use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::FloatDType;
use burn::tensor::TensorMetadata;
use burn::tensor::ops::IntTensor;
use burn::tensor::ops::{FloatTensor, FloatTensorOps};
use burn_cubecl::cubecl::CubeCount;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::cubecl::features::AtomicUsage;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, StorageType, Type};
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use glam::{Vec3, uvec2};

use crate::burn_glue::{RasterizeGrads, SplatBwdOps, SplatGrads};
use crate::kernels;
use brush_render::shaders::helpers::ProjectUniforms;

impl SplatBwdOps<Self> for MainBackendBase {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
    ) -> RasterizeGrads<Self> {
        let _span = tracing::trace_span!("rasterize_bwd").entered();

        let v_output = into_contiguous(v_output);
        let device = out_img.device.clone();
        let num_visible = projected_splats.shape()[0].max(1);
        let client = projected_splats.client.clone();

        // Sparse [num_visible, 10] indexed by compact_gid.
        let v_combined = Self::float_zeros([num_visible, 10].into(), &device, FloatDType::F32);

        let tile_bounds = uvec2(
            img_size
                .x
                .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
            img_size
                .y
                .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
        );

        let hard_floats = client
            .properties()
            .atomic_type_usage(Type::Scalar(StorageType::Atomic(ElemType::Float(
                FloatKind::F32,
            ))))
            .contains(AtomicUsage::Add);

        let cube_count = CubeCount::Static(tile_bounds.x, tile_bounds.y, 1);
        let cube_dim = CubeDim::new_1d(kernels::rasterize_backwards::SPLAT_BATCH);
        let uniforms = RasterizeUniformsLaunch::new(
            tile_bounds.x,
            img_size.x,
            img_size.y,
            background.x,
            background.y,
            background.z,
        );

        tracing::trace_span!("RasterizeBackwards").in_scope(|| {
            use kernels::rasterize_backwards::{
                CasAtomicAdd, HfAtomicAdd, rasterize_backwards_kernel,
            };
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                if hard_floats {
                    rasterize_backwards_kernel::launch_unchecked::<HfAtomicAdd, WgpuRuntime>(
                        &client,
                        cube_count,
                        cube_dim,
                        compact_gid_from_isect.into_tensor_arg(),
                        tile_offsets.into_tensor_arg(),
                        projected_splats.into_tensor_arg(),
                        out_img.into_tensor_arg(),
                        v_output.into_tensor_arg(),
                        v_combined.clone().into_tensor_arg(),
                        uniforms,
                    );
                } else {
                    rasterize_backwards_kernel::launch_unchecked::<CasAtomicAdd, WgpuRuntime>(
                        &client,
                        cube_count,
                        cube_dim,
                        compact_gid_from_isect.into_tensor_arg(),
                        tile_offsets.into_tensor_arg(),
                        projected_splats.into_tensor_arg(),
                        out_img.into_tensor_arg(),
                        v_output.into_tensor_arg(),
                        v_combined.clone().into_tensor_arg(),
                        uniforms,
                    );
                }
            }
        });

        RasterizeGrads { v_combined }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        let _span = tracing::trace_span!("project_bwd").entered();

        let transforms = into_contiguous(transforms);
        let raw_opac = into_contiguous(raw_opac);

        let device = transforms.device.clone();
        let num_points = transforms.shape()[0];
        let client = transforms.client.clone();

        // Dense outputs, the kernel scatters compact→global internally.
        let v_transforms = Self::float_zeros([num_points, 10].into(), &device, FloatDType::F32);
        let v_coeffs = Self::float_zeros(
            [
                num_points,
                sh_coeffs_for_degree(project_uniforms.sh_degree) as usize,
                3,
            ]
            .into(),
            &device,
            FloatDType::F32,
        );
        let v_raw_opac = Self::float_zeros([num_points].into(), &device, FloatDType::F32);
        let v_refine_weight = Self::float_zeros([num_points].into(), &device, FloatDType::F32);

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        let num_visible = project_uniforms.num_visible;
        let total_splats = project_uniforms.total_splats;

        let uniforms = build_project_uniforms_launch(
            &project_uniforms.viewmat,
            project_uniforms.focal,
            project_uniforms.pixel_center,
            project_uniforms.camera_position,
            project_uniforms.img_size,
            project_uniforms.tile_bounds,
            project_uniforms.sh_degree,
            total_splats,
            num_visible,
        );

        tracing::trace_span!("ProjectBackwards").in_scope(|| {
            // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
            unsafe {
                kernels::project_backwards::project_backwards_kernel::launch_unchecked::<
                    WgpuRuntime,
                >(
                    &client,
                    calc_cube_count_1d(num_visible, kernels::project_backwards::WG_SIZE),
                    CubeDim::new_1d(kernels::project_backwards::WG_SIZE),
                    transforms.into_tensor_arg(),
                    raw_opac.into_tensor_arg(),
                    global_from_compact_gid.into_tensor_arg(),
                    v_combined.into_tensor_arg(),
                    v_transforms.clone().into_tensor_arg(),
                    v_coeffs.clone().into_tensor_arg(),
                    v_raw_opac.clone().into_tensor_arg(),
                    v_refine_weight.clone().into_tensor_arg(),
                    uniforms,
                    mip_splat,
                    project_uniforms.sh_degree,
                );
            }
        });

        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}
