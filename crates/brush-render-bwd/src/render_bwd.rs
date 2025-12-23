use brush_kernel::{CubeCount, calc_cube_count_1d};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_wgsl::wgsl_kernel;

use brush_render::MainBackendBase;
use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::FloatDType;
use burn::tensor::ops::FloatTensorOps;
use burn::{prelude::Backend, tensor::ops::FloatTensor};
use burn_cubecl::cubecl::features::TypeUsage;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, StorageType};
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::kernel::into_contiguous;
use glam::uvec2;

use crate::burn_glue::{GaussianBackwardState, SplatBackwardOps};

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

#[derive(Debug, Clone)]
pub struct SplatGrads<B: Backend> {
    pub v_means: FloatTensor<B>,
    pub v_quats: FloatTensor<B>,
    pub v_scales: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

impl SplatBackwardOps<Self> for MainBackendBase {
    fn render_splats_bwd(
        state: GaussianBackwardState<Self>,
        v_output: FloatTensor<Self>,
    ) -> SplatGrads<Self> {
        // Comes from loss, might not be contiguous.
        let v_output = into_contiguous(v_output);

        // Comes from params, might not be contiguous.
        let means = into_contiguous(state.means);
        let log_scales = into_contiguous(state.log_scales);
        let quats = into_contiguous(state.quats);
        let raw_opac = into_contiguous(state.raw_opac);

        // We're in charge of these, SHOULD be contiguous but might as well.
        let projected_splats = into_contiguous(state.projected_splats);
        let uniforms_buffer = into_contiguous(state.uniforms_buffer);
        let compact_gid_from_isect = into_contiguous(state.compact_gid_from_isect);
        let global_from_compact_gid = into_contiguous(state.global_from_compact_gid);
        let tile_offsets = into_contiguous(state.tile_offsets);

        let device = &state.out_img.device;
        let img_dimgs = state.out_img.shape.dims;
        let img_size = glam::uvec2(img_dimgs[1] as u32, img_dimgs[0] as u32);

        let num_points = means.shape.dims[0];

        let client = &means.client;

        // Setup tensors.
        // Nb: these are packed vec3 values, special care is taken in the kernel to respect alignment.
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
        let v_raw_opac = Self::float_zeros([num_points].into(), device, FloatDType::F32);
        let v_grads = Self::float_zeros([num_points, 8].into(), device, FloatDType::F32);
        let v_refine_weight = Self::float_zeros([num_points].into(), device, FloatDType::F32);

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
            .type_usage(StorageType::Atomic(ElemType::Float(FloatKind::F32)))
            .contains(TypeUsage::AtomicAdd);

        let webgpu = cfg!(target_family = "wasm");
        let mip_splat = matches!(state.render_mode, SplatRenderMode::Mip);

        // Use checked execution, as the atomic loops are potentially unbounded.
        tracing::trace_span!("RasterizeBackwards").in_scope(|| {
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        RasterizeBackwards::task(hard_floats, webgpu),
                        CubeCount::Static(tile_bounds.x * tile_bounds.y, 1, 1),
                        Bindings::new().with_buffers(vec![
                            uniforms_buffer.handle.clone().binding(),
                            compact_gid_from_isect.handle.binding(),
                            global_from_compact_gid.handle.clone().binding(),
                            tile_offsets.handle.binding(),
                            projected_splats.handle.binding(),
                            state.out_img.handle.binding(),
                            v_output.handle.binding(),
                            v_grads.handle.clone().binding(),
                            v_raw_opac.handle.clone().binding(),
                            v_refine_weight.handle.clone().binding(),
                        ]),
                    )
                    .expect("Failed to bwd-diff splats");
            }
        });

        tracing::trace_span!("ProjectBackwards").in_scope(||
        // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
        unsafe {
        client.launch_unchecked(
            ProjectBackwards::task(mip_splat),
            calc_cube_count_1d(num_points as u32, ProjectBackwards::WORKGROUP_SIZE[0]),
            Bindings::new().with_buffers(
            vec![
                uniforms_buffer.handle.binding(),
                means.handle.binding(),
                log_scales.handle.binding(),
                quats.handle.binding(),
                raw_opac.handle.binding(),
                global_from_compact_gid.handle.binding(),
                v_grads.handle.binding(),
                v_means.handle.clone().binding(),
                v_scales.handle.clone().binding(),
                v_quats.handle.clone().binding(),
                v_coeffs.handle.clone().binding(),
                v_raw_opac.handle.clone().binding(),
            ]),
        ).expect("Failed to bwd-diff splats");
    });

        assert!(v_means.is_contiguous(), "Grads must be contiguous");
        assert!(v_quats.is_contiguous(), "Grads must be contiguous");
        assert!(v_scales.is_contiguous(), "Grads must be contiguous");
        assert!(v_coeffs.is_contiguous(), "Grads must be contiguous");
        assert!(v_raw_opac.is_contiguous(), "Grads must be contiguous");
        assert!(v_refine_weight.is_contiguous(), "Grads must be contiguous");

        SplatGrads {
            v_means,
            v_quats,
            v_scales,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        }
    }
}
