use burn::tensor::{DType, ops::FloatTensor};
use burn_cubecl::{BoolElement, fusion::FusionCubeRuntime};
use burn_fusion::{Fusion, FusionHandle, client::FusionClient, stream::Operation};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr};
use burn_wgpu::WgpuRuntime;

use crate::{
    MainBackendBase, SplatForward,
    camera::Camera,
    render::{calc_tile_bounds, max_intersections, render_forward},
    render_aux::RenderAux,
    shaders,
};

// Implement forward functions for the inner wgpu backend.
impl SplatForward<Self> for MainBackendBase {
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        opacity: FloatTensor<Self>,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>) {
        render_forward(
            camera, img_size, means, log_scales, quats, sh_coeffs, opacity, bwd_info,
        )
    }
}

impl SplatForward<Self> for Fusion<MainBackendBase> {
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        opacity: FloatTensor<Self>,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>) {
        struct CustomOp {
            cam: Camera,
            img_size: glam::UVec2,
            bwd_info: bool,
            desc: CustomOpIr,
        }

        impl<BT: BoolElement> Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp {
            fn execute(
                self: Box<Self>,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (inputs, outputs) = self.desc.consume();

                let [means, log_scales, quats, sh_coeffs, opacity] = inputs;
                let [
                    projected_splats,
                    uniforms_buffer,
                    num_intersections,
                    num_visible,
                    tile_offsets,
                    compact_gid_from_isect,
                    global_from_compact_gid,
                    out_img,
                    visible,
                    final_index,
                ] = outputs;

                let (img, aux) = MainBackendBase::render_splats(
                    &self.cam,
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(&means),
                    h.get_float_tensor::<MainBackendBase>(&log_scales),
                    h.get_float_tensor::<MainBackendBase>(&quats),
                    h.get_float_tensor::<MainBackendBase>(&sh_coeffs),
                    h.get_float_tensor::<MainBackendBase>(&opacity),
                    self.bwd_info,
                );

                // Register output.
                h.register_float_tensor::<MainBackendBase>(&out_img.id, img);
                h.register_float_tensor::<MainBackendBase>(
                    &projected_splats.id,
                    aux.projected_splats,
                );
                h.register_int_tensor::<MainBackendBase>(&uniforms_buffer.id, aux.uniforms_buffer);
                h.register_int_tensor::<MainBackendBase>(
                    &num_intersections.id,
                    aux.num_intersections,
                );
                h.register_int_tensor::<MainBackendBase>(&num_visible.id, aux.num_visible);
                h.register_int_tensor::<MainBackendBase>(&tile_offsets.id, aux.tile_offsets);
                h.register_int_tensor::<MainBackendBase>(
                    &compact_gid_from_isect.id,
                    aux.compact_gid_from_isect,
                );
                h.register_int_tensor::<MainBackendBase>(
                    &global_from_compact_gid.id,
                    aux.global_from_compact_gid,
                );

                h.register_float_tensor::<MainBackendBase>(&visible.id, aux.visible);
                h.register_int_tensor::<MainBackendBase>(&final_index.id, aux.final_index);
            }
        }

        let stream = means.stream;
        let client = means.client.clone();

        let num_points = means.shape[0];

        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / 4;
        let uniforms_size = size_of::<shaders::helpers::RenderUniforms>() / 4;
        let tile_bounds = calc_tile_bounds(img_size);
        let max_intersects = max_intersections(img_size, num_points as u32);

        // If render_u32_buffer is true, we render a packed buffer of u32 values, otherwise
        // render RGBA f32 values.
        let channels = if bwd_info { 4 } else { 1 };

        let out_img = client.tensor_uninitialized(
            vec![img_size.y as usize, img_size.x as usize, channels],
            if bwd_info { DType::F32 } else { DType::U32 },
        );

        let final_index_shape = if bwd_info {
            vec![img_size.y as usize, img_size.x as usize]
        } else {
            vec![1, 1]
        };
        let visible_shape = if bwd_info { vec![num_points] } else { vec![1] };

        let aux = RenderAux::<Self> {
            projected_splats: client.tensor_uninitialized(vec![num_points, proj_size], DType::F32),
            uniforms_buffer: client.tensor_uninitialized(vec![uniforms_size], DType::I32),
            num_intersections: client.tensor_uninitialized(vec![1], DType::I32),
            num_visible: client.tensor_uninitialized(vec![1], DType::I32),

            tile_offsets: client.tensor_uninitialized(
                vec![(tile_bounds.y * tile_bounds.x) as usize + 1],
                DType::I32,
            ),
            compact_gid_from_isect: client
                .tensor_uninitialized(vec![max_intersects as usize], DType::I32),
            global_from_compact_gid: client.tensor_uninitialized(vec![num_points], DType::I32),

            visible: client.tensor_uninitialized(visible_shape, DType::F32),
            final_index: client.tensor_uninitialized(final_index_shape, DType::I32),
        };

        let desc = CustomOpIr::new(
            "render_splats",
            &[
                means.into_ir(),
                log_scales.into_ir(),
                quats.into_ir(),
                sh_coeffs.into_ir(),
                opacity.into_ir(),
            ],
            &[
                aux.projected_splats.to_ir_out(),
                aux.uniforms_buffer.to_ir_out(),
                aux.num_intersections.to_ir_out(),
                aux.num_visible.to_ir_out(),
                aux.tile_offsets.to_ir_out(),
                aux.compact_gid_from_isect.to_ir_out(),
                aux.global_from_compact_gid.to_ir_out(),
                out_img.to_ir_out(),
                aux.visible.to_ir_out(),
                aux.final_index.to_ir_out(),
            ],
        );

        let op = CustomOp {
            cam: cam.clone(),
            img_size,
            bwd_info,
            desc: desc.clone(),
        };

        client.register(vec![stream], OperationIr::Custom(desc), op);
        (out_img, aux)
    }
}
