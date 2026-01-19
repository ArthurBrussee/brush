use brush_kernel::{CubeCount, calc_cube_count_1d, create_meta_binding};
use brush_render::MainBackendBase;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::shaders::helpers::RasterizeUniforms;
use brush_wgsl::wgsl_kernel;

use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::FloatDType;
use burn::tensor::ops::{FloatTensor, FloatTensorOps};
use burn_cubecl::cubecl::features::TypeUsage;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, StorageType};
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::kernel::into_contiguous;
use glam::uvec2;

use crate::burn_glue::{
    ProjectBwdState, RasterizeBwdState, RasterizeGrads, SplatBwdOps, SplatGrads,
};

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
    fn rasterize_bwd(
        state: RasterizeBwdState<Self>,
        v_output: FloatTensor<Self>,
    ) -> RasterizeGrads<Self> {
        let _span = tracing::trace_span!("rasterize_bwd").entered();

        // Comes from loss, might not be contiguous.
        let v_output = into_contiguous(v_output);

        // We're in charge of these, SHOULD be contiguous but might as well.
        let projected_splats = into_contiguous(state.projected_splats);
        let compact_gid_from_isect = into_contiguous(state.compact_gid_from_isect);
        let global_from_compact_gid = into_contiguous(state.global_from_compact_gid);
        let tile_offsets = into_contiguous(state.tile_offsets);
        let out_img = into_contiguous(state.out_img);

        let device = &out_img.device;
        let img_size = state.img_size;
        let num_points = projected_splats.shape.dims[0];

        let client = &projected_splats.client;

        // Setup output tensors.
        let v_projected_splats = Self::float_zeros([num_points, 8].into(), device, FloatDType::F32);
        let v_raw_opac = Self::float_zeros([num_points].into(), device, FloatDType::F32);
        let v_refine_weight = Self::float_zeros([num_points].into(), device, FloatDType::F32);

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
            background: [
                state.background.x,
                state.background.y,
                state.background.z,
                1.0,
            ],
        };

        let hard_floats = client
            .properties()
            .type_usage(StorageType::Atomic(ElemType::Float(FloatKind::F32)))
            .contains(TypeUsage::AtomicAdd);

        let webgpu = cfg!(target_family = "wasm");

        tracing::trace_span!("RasterizeBackwards").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        RasterizeBackwards::task(hard_floats, webgpu),
                        CubeCount::Static(tile_bounds.x * tile_bounds.y, 1, 1),
                        Bindings::new()
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
                    )
                    .expect("Failed to bwd-diff splats");
            }
        });

        RasterizeGrads {
            v_projected_splats,
            v_raw_opac,
            v_refine_weight,
        }
    }

    fn project_bwd(
        state: ProjectBwdState<Self>,
        rasterize_grads: RasterizeGrads<Self>,
    ) -> SplatGrads<Self> {
        let _span = tracing::trace_span!("project_bwd").entered();

        // Comes from params, might not be contiguous.
        let means = into_contiguous(state.means);
        let log_scales = into_contiguous(state.log_scales);
        let quats = into_contiguous(state.quats);
        let raw_opac = into_contiguous(state.raw_opac);
        let global_from_compact_gid = into_contiguous(state.global_from_compact_gid);

        let device = &means.device;
        let num_points = means.shape.dims[0];
        let client = &means.client;

        // Setup output tensors.
        let v_means = Self::float_zeros([num_points, 3].into(), device, FloatDType::F32);
        let v_scales = Self::float_zeros([num_points, 3].into(), device, FloatDType::F32);
        let v_quats = Self::float_zeros([num_points, 4].into(), device, FloatDType::F32);
        let v_coeffs = Self::float_zeros(
            [
                num_points,
                sh_coeffs_for_degree(state.sh_degree) as usize,
                3,
            ]
            .into(),
            device,
            FloatDType::F32,
        );

        let mip_splat = matches!(state.render_mode, SplatRenderMode::Mip);

        tracing::trace_span!("ProjectBackwards").in_scope(|| {
            // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        ProjectBackwards::task(mip_splat),
                        calc_cube_count_1d(num_points as u32, ProjectBackwards::WORKGROUP_SIZE[0]),
                        Bindings::new()
                            .with_buffers(vec![
                                state.num_visible.handle.binding(),
                                means.handle.binding(),
                                log_scales.handle.binding(),
                                quats.handle.binding(),
                                raw_opac.handle.binding(),
                                global_from_compact_gid.handle.binding(),
                                rasterize_grads.v_projected_splats.handle.binding(),
                                v_means.handle.clone().binding(),
                                v_scales.handle.clone().binding(),
                                v_quats.handle.clone().binding(),
                                v_coeffs.handle.clone().binding(),
                                rasterize_grads.v_raw_opac.handle.clone().binding(),
                            ])
                            .with_metadata(create_meta_binding(state.project_uniforms)),
                    )
                    .expect("Failed to bwd-diff splats");
            }
        });

        SplatGrads {
            v_means,
            v_quats,
            v_scales,
            v_coeffs,
            v_raw_opac: rasterize_grads.v_raw_opac,
            v_refine_weight: rasterize_grads.v_refine_weight,
        }
    }
}
