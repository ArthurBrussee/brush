use crate::gaussian_splats::SplatRenderMode;
use crate::kernels::types::RasterizeUniformsLaunch;
use crate::sh::sh_coeffs_for_degree;
use brush_cube::create_tensor;
use burn::backend::TensorMetadata;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::cubecl::CubeCount;
use burn::cubecl::CubeDim;
use burn::cubecl::calculate_cube_count_elemwise;
use burn::cubecl::features::AtomicUsage;
use burn::cubecl::ir::{ElemType, FloatKind, Type};
use burn::tensor::{DType, FloatDType};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use glam::{Vec3, uvec2};

use crate::bwd::burn_glue::{SplatBwdOps, SplatGrads};
use crate::bwd::kernels;
use crate::shaders::helpers::ProjectUniforms;

impl SplatBwdOps for CubeBackend {
    fn rasterize_bwd(
        out_img: FloatTensor<Self>,
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
        smooth_cutoff: bool,
    ) -> FloatTensor<Self> {
        let _span = tracing::trace_span!("rasterize_bwd").entered();

        let v_output = into_contiguous(v_output);
        let device = out_img.device.clone();
        let num_visible = projected_splats.shape()[0];
        let client = projected_splats.client.clone();

        // Sparse [num_visible, 10] indexed by compact_gid.
        let v_combined = Self::float_zeros([num_visible, 10].into(), &device, FloatDType::F32);

        let tile_bounds = uvec2(
            img_size.x.div_ceil(crate::shaders::helpers::TILE_WIDTH),
            img_size.y.div_ceil(crate::shaders::helpers::TILE_WIDTH),
        );

        let hard_floats = client
            .properties()
            .atomic_type_usage(Type::atomic(Type::new(ElemType::Float(FloatKind::F32))))
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
            if hard_floats {
                rasterize_backwards_kernel::launch::<HfAtomicAdd>(
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
                    smooth_cutoff,
                );
            } else {
                rasterize_backwards_kernel::launch::<CasAtomicAdd>(
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
                    smooth_cutoff,
                );
            }
        });

        v_combined
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        let _span = tracing::trace_span!("project_bwd").entered();

        // The screen-area regulariser only acts in this backward kernel, so we
        // stamp the weight onto the uniforms here rather than in the forward.
        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opac = into_contiguous(raw_opac);
        let min_scale = into_contiguous(min_scale);

        let device = transforms.device.clone();
        let client = transforms.client.clone();

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        // Compact outputs with the zero row in front (see the kernel); the
        // kernel writes every row, so no fill.
        let rows = project_uniforms.num_visible as usize + 1;
        let coeffs = sh_coeffs_for_degree(project_uniforms.sh_degree) as usize;
        let v_transforms = create_tensor([rows, 10], &device, DType::F32);
        let v_coeffs = create_tensor([rows, coeffs, 3], &device, DType::F32);
        let v_raw_opac = create_tensor([rows], &device, DType::F32);
        let v_refine_weight = create_tensor([rows], &device, DType::F32);

        let uniforms = project_uniforms.to_launch_object();
        let cube_dim = CubeDim::new_1d(kernels::project_backwards::WG_SIZE);

        tracing::trace_span!("ProjectBackwards").in_scope(|| {
            kernels::project_backwards::project_backwards_kernel::launch(
                &client,
                calculate_cube_count_elemwise(&client, rows, cube_dim),
                cube_dim,
                transforms.into_tensor_arg(),
                sh_coeffs.into_tensor_arg(),
                raw_opac.into_tensor_arg(),
                min_scale.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_combined.into_tensor_arg(),
                v_transforms.clone().into_tensor_arg(),
                v_coeffs.clone().into_tensor_arg(),
                v_raw_opac.clone().into_tensor_arg(),
                v_refine_weight.clone().into_tensor_arg(),
                uniforms,
                mip_splat,
                has_min_scale,
                project_uniforms.sh_degree,
                project_uniforms.camera_model,
            );
        });

        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}
