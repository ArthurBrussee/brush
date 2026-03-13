use brush_kernel::{calc_cube_count_1d, create_meta_binding};
use brush_render::MainBackendBase;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::shaders::helpers::RasterizeUniforms;
use brush_wgsl::wgsl_kernel;

use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::ops::IntTensor;
use burn::tensor::ops::{FloatTensor, FloatTensorOps};
use burn::tensor::{FloatDType, TensorMetadata};
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
        global_from_compact_gid: IntTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        background: Vec3,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
    ) -> RasterizeGrads<Self> {
        let _span = tracing::trace_span!("rasterize_bwd").entered();

        // Comes from loss, might not be contiguous.
        let v_output = into_contiguous(v_output);

        let device = &out_img.device;
        let num_points = projected_splats.shape()[0];

        let client = &projected_splats.client;

        // Setup combined output tensor: [projected_splat_grads(8) + opac(1) + refine(1)] per splat
        let v_combined = Self::float_zeros([num_points, 10].into(), device, FloatDType::F32);

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
                            global_from_compact_gid.handle.binding(),
                            tile_offsets.handle.binding(),
                            projected_splats.handle.binding(),
                            out_img.handle.binding(),
                            v_output.handle.binding(),
                            v_projected_splats.handle.clone().binding(),
                            v_raw_opac.handle.clone().binding(),
                            v_refine_weight.handle.clone().binding(),
                        ])
                        .with_metadata(create_meta_binding(rasterize_uniforms)),
                );
            }
        });

        RasterizeGrads { v_combined }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_bwd(
        transforms: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        num_visible: IntTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        sh_degree: u32,
        render_mode: SplatRenderMode,
        rasterize_grads: RasterizeGrads<Self>,
    ) -> SplatGrads<Self> {
        let _span = tracing::trace_span!("project_bwd").entered();

        // Comes from params, might not be contiguous.
        let transforms = into_contiguous(transforms);
        let raw_opac = into_contiguous(raw_opac);

        let device = &transforms.device;
        let num_points = transforms.shape()[0];
        let client = &transforms.client;

        // Setup output tensors.
        let v_transforms = Self::float_zeros([num_points, 10].into(), device, FloatDType::F32);
        let v_coeffs = Self::float_zeros(
            [num_points, sh_coeffs_for_degree(sh_degree) as usize, 3].into(),
            device,
            FloatDType::F32,
        );

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        tracing::trace_span!("ProjectBackwards").in_scope(|| {
            // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        ProjectBackwards::task(mip_splat),
                        calc_cube_count_1d(num_points as u32, ProjectBackwards::WORKGROUP_SIZE[0]),
                        Bindings::new()
                            .with_buffers(vec![
                                num_visible.handle.binding(),
                                transforms.handle.binding(),
                                raw_opac.handle.binding(),
                                global_from_compact_gid.handle.binding(),
                                rasterize_grads.v_combined.handle.clone().binding(),
                                v_transforms.handle.clone().binding(),
                                v_coeffs.handle.clone().binding(),
                            ])
                            .with_metadata(create_meta_binding(project_uniforms)),
                    )
                    .expect("Failed to bwd-diff splats");
            }
        });

        // Extract v_raw_opac and v_refine_weight from the combined rasterize grads buffer.
        // These are zero-copy view operations.
        let v_raw_opac = Self::float_reshape(
            Self::float_slice(
                rasterize_grads.v_combined.clone(),
                &[(0..num_points).into(), (8..9).into()],
            ),
            [num_points].into(),
        );
        let v_refine_weight = Self::float_reshape(
            Self::float_slice(
                rasterize_grads.v_combined,
                &[(0..num_points).into(), (9..10).into()],
            ),
            [num_points].into(),
        );

        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}
