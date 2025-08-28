use super::shaders::{project_backwards, rasterize_backwards};
use brush_kernel::{
    CubeCount, CubeDim, CubeTensor, calc_cube_count, create_tensor, kernel_source_gen,
};

use brush_render::MainBackendBase;
use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::DType;
use burn::{backend::wgpu::WgpuRuntime, prelude::Backend, tensor::ops::FloatTensor};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::prelude::{ABSOLUTE_POS, Line, Tensor};
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::cubecl::{AtomicFeature, calculate_cube_count_elemwise, cube, terminate};
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
    let device = &out_img.device;
    let img_dimgs = out_img.shape.dims;
    let img_size = glam::uvec2(img_dimgs[1] as u32, img_dimgs[0] as u32);

    let num_points = means.shape.dims[0];

    let client = &means.client;

    // Setup tensors.
    // Nb: these are packed vec3 values, special care is taken in the kernel to respect alignment.
    let v_means = create_tensor([num_points, 3], device, DType::F32);
    let v_scales = create_tensor([num_points, 3], device, DType::F32);
    let v_quats = create_tensor([num_points, 4], device, DType::F32);
    let v_coeffs = create_tensor(
        [num_points, sh_coeffs_for_degree(sh_degree) as usize, 3],
        device,
        DType::F32,
    );
    let v_raw_opac = create_tensor([num_points], device, DType::F32);
    let v_grads = create_tensor([num_points, 8], device, DType::F32);
    let v_refine_weight = create_tensor([num_points, 2], device, DType::F32);

    #[cube(launch_unchecked)]
    pub fn zero_all_grads(
        means: &mut Tensor<f32>,
        scales: &mut Tensor<f32>,
        v_quats: &mut Tensor<Line<f32>>,
        v_coeffs: &mut Tensor<f32>,
        v_opac: &mut Tensor<f32>,
        v_grads: &mut Tensor<Line<f32>>,
        v_refine_weight: &mut Tensor<Line<f32>>,
        #[comptime] num_points: u32,
        #[comptime] total_coeffs: u32,
    ) {
        if ABSOLUTE_POS >= num_points {
            terminate!();
        }
        #[unroll]
        for i in 0..3 {
            means[ABSOLUTE_POS * 3 + i] = 0.0;
            scales[ABSOLUTE_POS * 3 + i] = 0.0;
        }
        #[unroll]
        for i in 0..2 {
            v_grads[ABSOLUTE_POS * 2 + i] = Line::empty(4u32).fill(0.0);
        }
        #[unroll]
        for i in 0..total_coeffs {
            v_coeffs[ABSOLUTE_POS * total_coeffs + i] = 0.0;
        }
        v_quats[ABSOLUTE_POS] = Line::empty(4u32).fill(0.0);
        v_opac[ABSOLUTE_POS] = 0.0;
        v_refine_weight[ABSOLUTE_POS] = Line::empty(2u32).fill(0.0);
    }

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_points, cube_dim);

    // SAFETY: No OOB
    unsafe {
        zero_all_grads::launch_unchecked::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            v_means.as_tensor_arg::<f32>(1),
            v_scales.as_tensor_arg::<f32>(1),
            v_quats.as_tensor_arg::<f32>(4),
            v_coeffs.as_tensor_arg::<f32>(1),
            v_raw_opac.as_tensor_arg::<f32>(1),
            v_grads.as_tensor_arg::<f32>(4),
            v_refine_weight.as_tensor_arg::<f32>(2),
            num_points as u32,
            sh_coeffs_for_degree(sh_degree) * 3,
        );
    }

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

    SplatGrads {
        v_means,
        v_quats,
        v_scales,
        v_coeffs,
        v_raw_opac,
        v_refine_weight,
    }
}
