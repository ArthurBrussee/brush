use std::marker::PhantomData;

use burn::tensor::{DType, ops::FloatTensor};
use burn_cubecl::{BoolElement, FloatElement, IntElement, fusion::FusionCubeRuntime};
use burn_fusion::{Fusion, FusionHandle, client::FusionClient, stream::Operation};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr};
use burn_wgpu::WgpuRuntime;

use crate::{
    BBase, RenderAuxPrimitive, SplatForward,
    camera::Camera,
    render::{calc_tile_bounds, max_intersections, render_forward},
    shaders,
};

// Implement forward functions for the inner wgpu backend.
impl<F: FloatElement, I: IntElement, BT: BoolElement> SplatForward<Self> for BBase<F, I, BT> {
    fn render_splats(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacity: FloatTensor<Self>,
        render_u32_buffer: bool,
    ) -> (FloatTensor<Self>, RenderAuxPrimitive<Self>) {
        render_forward(
            camera,
            img_size,
            means,
            log_scales,
            quats,
            sh_coeffs,
            raw_opacity,
            render_u32_buffer,
        )
    }
}

impl<F: FloatElement, I: IntElement, BT: BoolElement> SplatForward<Self>
    for Fusion<BBase<F, I, BT>>
{
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacity: FloatTensor<Self>,
        render_u32_buffer: bool,
    ) -> (FloatTensor<Self>, RenderAuxPrimitive<Self>) {
        struct CustomOp<F: FloatElement, I: IntElement, BT: BoolElement> {
            cam: Camera,
            img_size: glam::UVec2,
            render_u32_buffer: bool,
            desc: CustomOpIr,
            _c: PhantomData<(F, I, BT)>,
        }

        impl<F: FloatElement, I: IntElement, BT: BoolElement>
            Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp<F, I, BT>
        {
            fn execute(
                self: Box<Self>,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (
                    [means, log_scales, quats, sh_coeffs, raw_opacity],
                    [
                        projected_splats,
                        uniforms_buffer,
                        num_intersections,
                        num_visible,
                        final_index,
                        tile_offsets,
                        compact_gid_from_isect,
                        global_from_compact_gid,
                        radii,
                        out_img,
                    ],
                ) = self.desc.consume();

                let (img, aux) = BBase::<F, I, BT>::render_splats(
                    &self.cam,
                    self.img_size,
                    h.get_float_tensor::<BBase<F, I, BT>>(&means),
                    h.get_float_tensor::<BBase<F, I, BT>>(&log_scales),
                    h.get_float_tensor::<BBase<F, I, BT>>(&quats),
                    h.get_float_tensor::<BBase<F, I, BT>>(&sh_coeffs),
                    h.get_float_tensor::<BBase<F, I, BT>>(&raw_opacity),
                    self.render_u32_buffer,
                );

                // Register output.
                h.register_float_tensor::<BBase<F, I, BT>>(&out_img.id, img);
                h.register_float_tensor::<BBase<F, I, BT>>(
                    &projected_splats.id,
                    aux.projected_splats,
                );
                h.register_int_tensor::<BBase<F, I, BT>>(&uniforms_buffer.id, aux.uniforms_buffer);
                h.register_int_tensor::<BBase<F, I, BT>>(
                    &num_intersections.id,
                    aux.num_intersections,
                );
                h.register_int_tensor::<BBase<F, I, BT>>(&num_visible.id, aux.num_visible);
                h.register_int_tensor::<BBase<F, I, BT>>(&final_index.id, aux.final_index);
                h.register_int_tensor::<BBase<F, I, BT>>(&tile_offsets.id, aux.tile_offsets);
                h.register_int_tensor::<BBase<F, I, BT>>(
                    &compact_gid_from_isect.id,
                    aux.compact_gid_from_isect,
                );
                h.register_int_tensor::<BBase<F, I, BT>>(
                    &global_from_compact_gid.id,
                    aux.global_from_compact_gid,
                );
                h.register_float_tensor::<BBase<F, I, BT>>(&radii.id, aux.radii);
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
        let channels = if render_u32_buffer { 1 } else { 4 };

        let out_img = client.tensor_uninitialized(
            vec![img_size.y as usize, img_size.x as usize, channels],
            DType::F32,
        );

        let aux = RenderAuxPrimitive::<Self> {
            projected_splats: client.tensor_uninitialized(vec![num_points, proj_size], DType::F32),
            uniforms_buffer: client.tensor_uninitialized(vec![uniforms_size], DType::I32),
            num_intersections: client.tensor_uninitialized(vec![1], DType::I32),
            num_visible: client.tensor_uninitialized(vec![1], DType::I32),
            final_index: client
                .tensor_uninitialized(vec![img_size.y as usize, img_size.x as usize], DType::I32),
            tile_offsets: client.tensor_uninitialized(
                vec![(tile_bounds.y * tile_bounds.x) as usize + 1],
                DType::I32,
            ),
            compact_gid_from_isect: client
                .tensor_uninitialized(vec![max_intersects as usize], DType::I32),
            global_from_compact_gid: client.tensor_uninitialized(vec![num_points], DType::I32),
            radii: client.tensor_uninitialized(vec![num_points], DType::F32),
        };

        let desc = CustomOpIr::new(
            "render_splats",
            &[
                means.into_ir(),
                log_scales.into_ir(),
                quats.into_ir(),
                sh_coeffs.into_ir(),
                raw_opacity.into_ir(),
            ],
            &[
                aux.projected_splats.to_ir_out(),
                aux.uniforms_buffer.to_ir_out(),
                aux.num_intersections.to_ir_out(),
                aux.num_visible.to_ir_out(),
                aux.final_index.to_ir_out(),
                aux.tile_offsets.to_ir_out(),
                aux.compact_gid_from_isect.to_ir_out(),
                aux.global_from_compact_gid.to_ir_out(),
                aux.radii.to_ir_out(),
                out_img.to_ir_out(),
            ],
        );

        let op = CustomOp::<F, I, BT> {
            cam: cam.clone(),
            img_size,
            render_u32_buffer,
            desc: desc.clone(),
            _c: PhantomData {},
        };

        client.register(vec![stream], OperationIr::Custom(desc), op);

        (out_img, aux)
    }
}
