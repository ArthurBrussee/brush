use burn::tensor::{
    ops::FloatTensor,
    repr::{CustomOpDescription, HandleContainer, OperationDescription},
    DType,
};
use burn_fusion::{client::FusionClient, stream::Operation, Fusion};
use burn_jit::fusion::{FusionJitRuntime, JitFusionHandle};
use burn_wgpu::WgpuRuntime;

use crate::{
    camera::Camera,
    render::{calc_tile_bounds, max_intersections, render_forward},
    shaders, BBase, RenderAuxPrimitive, SplatForward,
};

// Implement forward functions for the inner wgpu backend.
impl SplatForward<Self> for BBase {
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

impl SplatForward<Self> for Fusion<BBase> {
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
        struct CustomOp {
            cam: Camera,
            img_size: glam::UVec2,
            render_u32_buffer: bool,
            desc: CustomOpDescription,
        }

        impl Operation<FusionJitRuntime<WgpuRuntime, u32>> for CustomOp {
            fn execute(self: Box<Self>, h: &mut HandleContainer<JitFusionHandle<WgpuRuntime>>) {
                let (
                    [means, log_scales, quats, sh_coeffs, raw_opacity],
                    [projected_splats, uniforms_buffer, num_intersections, num_visible, final_index, tile_offsets, compact_gid_from_isect, global_from_compact_gid, radii, out_img],
                ) = self.desc.consume();

                let (img, aux) = BBase::render_splats(
                    &self.cam,
                    self.img_size,
                    h.get_float_tensor::<BBase>(&means),
                    h.get_float_tensor::<BBase>(&log_scales),
                    h.get_float_tensor::<BBase>(&quats),
                    h.get_float_tensor::<BBase>(&sh_coeffs),
                    h.get_float_tensor::<BBase>(&raw_opacity),
                    self.render_u32_buffer,
                );

                // Register output.
                h.register_float_tensor::<BBase>(&out_img.id, img);
                h.register_float_tensor::<BBase>(&projected_splats.id, aux.projected_splats);
                h.register_int_tensor::<BBase>(&uniforms_buffer.id, aux.uniforms_buffer);
                h.register_int_tensor::<BBase>(&num_intersections.id, aux.num_intersections);
                h.register_int_tensor::<BBase>(&num_visible.id, aux.num_visible);
                h.register_int_tensor::<BBase>(&final_index.id, aux.final_index);
                h.register_int_tensor::<BBase>(&tile_offsets.id, aux.tile_offsets);
                h.register_int_tensor::<BBase>(
                    &compact_gid_from_isect.id,
                    aux.compact_gid_from_isect,
                );
                h.register_int_tensor::<BBase>(
                    &global_from_compact_gid.id,
                    aux.global_from_compact_gid,
                );
                h.register_float_tensor::<BBase>(&radii.id, aux.radii);
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

        let desc = CustomOpDescription::new(
            "render_splats",
            &[
                means.into_description(),
                log_scales.into_description(),
                quats.into_description(),
                sh_coeffs.into_description(),
                raw_opacity.into_description(),
            ],
            &[
                aux.projected_splats.to_description_out(),
                aux.uniforms_buffer.to_description_out(),
                aux.num_intersections.to_description_out(),
                aux.num_visible.to_description_out(),
                aux.final_index.to_description_out(),
                aux.tile_offsets.to_description_out(),
                aux.compact_gid_from_isect.to_description_out(),
                aux.global_from_compact_gid.to_description_out(),
                aux.radii.to_description_out(),
                out_img.to_description_out(),
            ],
        );

        let op = CustomOp {
            cam: cam.clone(),
            img_size,
            render_u32_buffer,
            desc: desc.clone(),
        };

        client.register(vec![stream], OperationDescription::Custom(desc), op);

        (out_img, aux)
    }
}
