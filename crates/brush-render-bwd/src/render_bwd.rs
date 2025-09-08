use super::shaders::{project_backwards, rasterize_backwards};
use brush_kernel::{CubeCount, CubeTensor, calc_cube_count, kernel_source_gen};

use brush_render::MainBackendBase;
use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::FloatDType;
use burn::tensor::ops::FloatTensorOps;
use burn::{backend::wgpu::WgpuRuntime, prelude::Backend, tensor::ops::FloatTensor};
use burn_cubecl::cubecl::AtomicFeature;
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::kernel::into_contiguous;
use glam::uvec2;

kernel_source_gen!(ProjectBackwards {}, project_backwards);
kernel_source_gen!(RasterizeBackwards { hard_float }, rasterize_backwards);

#[derive(Debug, Clone)]
pub struct SplatGrads<B: Backend> {
    pub v_means: FloatTensor<B>,
    pub v_quats: FloatTensor<B>,
    pub v_scales: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_backward(
    v_output: CubeTensor<WgpuRuntime>,

    means: CubeTensor<WgpuRuntime>,
    quats: CubeTensor<WgpuRuntime>,
    log_scales: CubeTensor<WgpuRuntime>,
    out_img: CubeTensor<WgpuRuntime>,

    projected_splats: CubeTensor<WgpuRuntime>,
    uniforms_buffer: CubeTensor<WgpuRuntime>,
    compact_gid_from_isect: CubeTensor<WgpuRuntime>,
    global_from_compact_gid: CubeTensor<WgpuRuntime>,
    tile_offsets: CubeTensor<WgpuRuntime>,
    sh_degree: u32,
) -> SplatGrads<MainBackendBase> {
    // Comes from loss, might not be contiguous.
    let v_output = into_contiguous(v_output);

    // Comes from params, might not be contiguous.
    let means = into_contiguous(means);
    let log_scales = into_contiguous(log_scales);
    let quats = into_contiguous(quats);

    // We're in charge of these, SHOULD be contiguous but might as well.
    let projected_splats = into_contiguous(projected_splats);
    let uniforms_buffer = into_contiguous(uniforms_buffer);
    let compact_gid_from_isect = into_contiguous(compact_gid_from_isect);
    let global_from_compact_gid = into_contiguous(global_from_compact_gid);
    let tile_offsets = into_contiguous(tile_offsets);

    let device = &out_img.device;
    let img_dimgs = out_img.shape.dims;
    let img_size = glam::uvec2(img_dimgs[1] as u32, img_dimgs[0] as u32);

    let num_points = means.shape.dims[0];

    let client = &means.client;

    // Setup tensors.
    // Nb: these are packed vec3 values, special care is taken in the kernel to respect alignment.
    let v_means = MainBackendBase::float_zeros([num_points, 3].into(), device, FloatDType::F32);

    let v_scales = MainBackendBase::float_zeros([num_points, 3].into(), device, FloatDType::F32);
    let v_quats = MainBackendBase::float_zeros([num_points, 4].into(), device, FloatDType::F32);
    let v_coeffs = MainBackendBase::float_zeros(
        [num_points, sh_coeffs_for_degree(sh_degree) as usize, 3].into(),
        device,
        FloatDType::F32,
    );
    let v_raw_opac = MainBackendBase::float_zeros([num_points].into(), device, FloatDType::F32);
    let v_grads = MainBackendBase::float_zeros([num_points, 8].into(), device, FloatDType::F32);
    let v_refine_weight =
        MainBackendBase::float_zeros([num_points].into(), device, FloatDType::F32);

    let tile_bounds = uvec2(
        img_size
            .x
            .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
        img_size
            .y
            .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
    );

    let hard_floats =
        client
            .properties()
            .feature_enabled(burn_cubecl::cubecl::Feature::AtomicFloat(
                AtomicFeature::Add,
            ));

    // Use checked execution, as the atomic loops are potentially unbounded.
    tracing::trace_span!("RasterizeBackwards").in_scope(|| {
        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client.execute_unchecked(
                RasterizeBackwards::task(hard_floats),
                CubeCount::Static(tile_bounds.x * tile_bounds.y, 1, 1),
                Bindings::new().with_buffers(vec![
                    uniforms_buffer.handle.clone().binding(),
                    compact_gid_from_isect.handle.binding(),
                    global_from_compact_gid.handle.clone().binding(),
                    tile_offsets.handle.binding(),
                    projected_splats.handle.binding(),
                    out_img.handle.binding(),
                    v_output.handle.binding(),
                    v_grads.handle.clone().binding(),
                    v_raw_opac.handle.clone().binding(),
                    v_refine_weight.handle.clone().binding(),
                ]),
            );
        }
    });

    tracing::trace_span!("ProjectBackwards").in_scope(||
        // SAFETY: Kernel has to contain no OOB indexing, bounded loops.
        unsafe {
        client.execute_unchecked(
            ProjectBackwards::task(),
            calc_cube_count([num_points as u32], ProjectBackwards::WORKGROUP_SIZE),
            Bindings::new().with_buffers(
            vec![
                uniforms_buffer.handle.binding(),
                means.handle.binding(),
                log_scales.handle.binding(),
                quats.handle.binding(),
                global_from_compact_gid.handle.binding(),
                v_grads.handle.binding(),
                v_means.handle.clone().binding(),
                v_scales.handle.clone().binding(),
                v_quats.handle.clone().binding(),
                v_coeffs.handle.clone().binding()
            ]),
        );
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
