use brush_kernel::{CubeTensor, bytemuck, calc_cube_count_1d, create_meta_binding};
use brush_render::MainBackendBase;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::shaders::helpers::RasterizeUniforms;
use brush_wgsl::wgsl_kernel;

use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::ops::IntTensor;
use burn::tensor::ops::{FloatTensor, FloatTensorOps};
use burn::tensor::{DType, FloatDType, Shape, TensorMetadata};
use burn_cubecl::cubecl::features::TypeUsage;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, StorageType};
use burn_cubecl::cubecl::server::KernelArguments;
use burn_cubecl::kernel::into_contiguous;
use glam::{Vec3, uvec2};

use crate::burn_glue::{RasterizeGrads, SplatBwdOps, SplatGrads};
use brush_render::shaders::helpers::ProjectUniforms;

// Kernel definitions using proc macro
#[wgsl_kernel(
    source = "src/shaders/project_backwards.wgsl",
    includes = ["../brush-render/src/shaders/helpers.wgsl"],
)]
pub struct ProjectBackwards {
    mip_filter: bool,
}

#[wgsl_kernel(
    source = "src/shaders/rasterize_backwards.wgsl",
    includes = ["../brush-render/src/shaders/helpers.wgsl"],
)]
pub struct RasterizeBackwards {
    pub hard_float: bool,
    pub webgpu: bool,
}

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

        let device = &out_img.device;
        let num_visible = projected_splats.shape()[0].max(1);

        let client = &projected_splats.client;

        // Sparse [num_visible, 10] indexed by compact_gid.
        let v_combined = Self::float_zeros([num_visible, 10].into(), device, FloatDType::F32);

        let tile_bounds = uvec2(
            img_size
                .x
                .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
            img_size
                .y
                .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
        );

        // Create RasterizeUniforms for the backward rasterize pass
        let rasterize_uniforms = RasterizeUniforms {
            tile_bounds: tile_bounds.into(),
            img_size: img_size.into(),
            background: [background.x, background.y, background.z, 1.0],
        };

        let hard_floats = client
            .properties()
            .type_usage(StorageType::Atomic(ElemType::Float(FloatKind::F32)))
            .contains(TypeUsage::AtomicAdd);

        let webgpu = cfg!(target_family = "wasm");

        tracing::trace_span!("RasterizeBackwards").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client.launch_unchecked(
                    RasterizeBackwards::task(hard_floats, webgpu),
                    calc_cube_count_1d(
                        tile_bounds.x * tile_bounds.y * RasterizeBackwards::WORKGROUP_SIZE[0],
                        RasterizeBackwards::WORKGROUP_SIZE[0],
                    ),
                    KernelArguments::new()
                        .with_buffers(vec![
                            compact_gid_from_isect.handle.binding(),
                            tile_offsets.handle.binding(),
                            projected_splats.handle.binding(),
                            out_img.handle.binding(),
                            v_output.handle.binding(),
                            v_combined.handle.clone().binding(),
                        ])
                        .with_info(create_meta_binding(rasterize_uniforms)),
                );
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
        sh_degree: u32,
        render_mode: SplatRenderMode,
        v_combined: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        let _span = tracing::trace_span!("project_bwd").entered();

        let transforms = into_contiguous(transforms);
        let raw_opac = into_contiguous(raw_opac);

        let device = &transforms.device;
        let num_points = transforms.shape()[0];
        let client = &transforms.client;

        // Dense outputs, the kernel scatters compact→global internally.
        let v_transforms = Self::float_zeros([num_points, 10].into(), device, FloatDType::F32);
        let v_coeffs = Self::float_zeros(
            [num_points, sh_coeffs_for_degree(sh_degree) as usize, 3].into(),
            device,
            FloatDType::F32,
        );
        let v_raw_opac = Self::float_zeros([num_points].into(), device, FloatDType::F32);
        let v_refine_weight = Self::float_zeros([num_points].into(), device, FloatDType::F32);

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        let num_visible = project_uniforms.num_visible;
        // Create GPU buffer from CPU num_visible for the kernel binding.
        let num_visible_buf = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            Shape::new([1]),
            client.create_from_slice(bytemuck::cast_slice(&[num_visible])),
            DType::U32,
        );

        tracing::trace_span!("ProjectBackwards").in_scope(|| {
            // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
            unsafe {
                client.launch_unchecked(
                    ProjectBackwards::task(mip_splat),
                    calc_cube_count_1d(num_visible, ProjectBackwards::WORKGROUP_SIZE[0]),
                    KernelArguments::new()
                        .with_buffers(vec![
                            num_visible_buf.handle.binding(),
                            transforms.handle.binding(),
                            raw_opac.handle.binding(),
                            global_from_compact_gid.handle.binding(),
                            v_combined.handle.binding(),
                            v_transforms.handle.clone().binding(),
                            v_coeffs.handle.clone().binding(),
                            v_raw_opac.handle.clone().binding(),
                            v_refine_weight.handle.clone().binding(),
                        ])
                        .with_info(create_meta_binding(project_uniforms)),
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
