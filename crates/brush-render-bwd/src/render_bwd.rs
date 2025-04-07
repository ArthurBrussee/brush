use super::shaders::{project_backwards, rasterize_backwards};
use crate::shaders::gather_grads;
use brush_kernel::{CubeCount, CubeTensor, calc_cube_count, kernel_source_gen};
use brush_render::BBase;
use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::ops::FloatTensorOps;
use burn::{backend::wgpu::WgpuRuntime, prelude::Backend, tensor::ops::FloatTensor};
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::{BoolElement, cubecl::AtomicFeature};
use glam::uvec2;

kernel_source_gen!(GatherGrads {}, gather_grads);
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
pub(crate) fn render_backward<BT: BoolElement>(
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
    final_index: CubeTensor<WgpuRuntime>,
    sh_degree: u32,
) -> SplatGrads<BBase<BT>> {
    let device = &out_img.device;
    let img_dimgs = out_img.shape.dims;
    let img_size = glam::uvec2(img_dimgs[1] as u32, img_dimgs[0] as u32);

    let num_points = means.shape.dims[0];

    let client = &means.client;

    // Create tensors to hold gradients.

    // Nb: these are packed vec3 values, special care is taken in the kernel to respect alignment.
    // Nb: These have to be zeroed out - as we only write to visible splats.
    let v_means = BBase::<BT>::float_zeros([num_points, 3].into(), device);
    let v_scales = BBase::<BT>::float_zeros([num_points, 3].into(), device);
    let v_quats = BBase::<BT>::float_zeros([num_points, 4].into(), device);

    let v_coeffs = BBase::<BT>::float_zeros(
        [num_points, sh_coeffs_for_degree(sh_degree) as usize, 3].into(),
        device,
    );
    let v_opac = BBase::<BT>::float_zeros([num_points].into(), device);

    let tile_bounds = uvec2(
        img_size
            .x
            .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
        img_size
            .y
            .div_ceil(brush_render::shaders::helpers::TILE_WIDTH),
    );
    let invocations = tile_bounds.x * tile_bounds.y;

    // These gradients are atomically added to so important to zero them.
    let v_grads = BBase::<BT>::float_zeros([num_points, 9].into(), device);
    let v_refine_weight = BBase::<BT>::float_zeros([num_points, 2].into(), device);

    let hard_floats =
        client
            .properties()
            .feature_enabled(burn_cubecl::cubecl::Feature::AtomicFloat(
                AtomicFeature::Add,
            ));

    tracing::trace_span!("RasterizeBackwards", sync_burn = true).in_scope(||
            // SAFETY: Kernel has to contain no OOB indexing.
            unsafe {
                client.execute_unchecked(
                    RasterizeBackwards::task(hard_floats),
                    CubeCount::Static(invocations, 1, 1),
                    Bindings::new().with_buffers(
                    vec![
                        uniforms_buffer.clone().handle.binding(),
                        compact_gid_from_isect.handle.binding(),
                        tile_offsets.handle.binding(),
                        projected_splats.handle.binding(),
                        final_index.handle.binding(),
                        out_img.handle.binding(),
                        v_output.handle.binding(),
                        v_grads.clone().handle.binding(),
                        v_refine_weight.clone().handle.binding(),
                    ]),
                );
            });
    let _span = tracing::trace_span!("GatherGrads", sync_burn = true).entered();

    // SAFETY: Kernel has to contain no OOB indexing.
    unsafe {
        client.execute_unchecked(
            GatherGrads::task(),
            calc_cube_count([num_points as u32], GatherGrads::WORKGROUP_SIZE),
            Bindings::new().with_buffers(vec![
                uniforms_buffer.clone().handle.binding(),
                global_from_compact_gid.clone().handle.binding(),
                // raw_opac.handle.binding(),
                means.clone().handle.binding(),
                v_grads.clone().handle.binding(),
                v_coeffs.handle.clone().binding(),
                v_opac.handle.clone().binding(),
            ]),
        );
    }

    tracing::trace_span!("ProjectBackwards", sync_burn = true).in_scope(||
        // SAFETY: Kernel has to contain no OOB indexing.
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
            ]),
        );
    });

    SplatGrads {
        v_means,
        v_quats,
        v_scales,
        v_coeffs,
        v_raw_opac: v_opac,
        v_refine_weight,
    }
}
