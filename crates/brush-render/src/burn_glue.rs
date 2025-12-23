use burn::tensor::{DType, Shape, ops::FloatTensor};
use burn_cubecl::{BoolElement, fusion::FusionCubeRuntime};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;
use glam::Vec3;

use crate::{
    MainBackendBase, SplatForward, camera::Camera, gaussian_splats::SplatRenderMode,
    render_aux::RenderAux, shaders,
};

impl SplatForward<Self> for Fusion<MainBackendBase> {
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        opacity: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>) {
        #[derive(Debug)]
        struct CustomOp {
            cam: Camera,
            img_size: glam::UVec2,
            render_mode: SplatRenderMode,
            bwd_info: bool,
            background: Vec3,
            desc: CustomOpIr,
        }

        impl<BT: BoolElement> Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [means, log_scales, quats, sh_coeffs, opacity] = inputs;
                let [
                    // Img
                    out_img,
                    // Aux
                    projected_splats,
                    uniforms_buffer,
                    global_from_compact_gid,
                    num_visible,
                    visible,
                ] = outputs;

                let (img, aux) = MainBackendBase::render_splats(
                    &self.cam,
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(means),
                    h.get_float_tensor::<MainBackendBase>(log_scales),
                    h.get_float_tensor::<MainBackendBase>(quats),
                    h.get_float_tensor::<MainBackendBase>(sh_coeffs),
                    h.get_float_tensor::<MainBackendBase>(opacity),
                    self.render_mode,
                    self.background,
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
                    &global_from_compact_gid.id,
                    aux.global_from_compact_gid,
                );
                h.register_int_tensor::<MainBackendBase>(&num_visible.id, aux.num_visible);
                h.register_float_tensor::<MainBackendBase>(&visible.id, aux.visible);
            }
        }

        let client = means.client.clone();

        let num_points = means.shape[0];

        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / 4;
        let uniforms_size = size_of::<shaders::helpers::RenderUniforms>() / 4;

        // If render_u32_buffer is true, we render a packed buffer of u32 values, otherwise
        // render RGBA f32 values.
        let channels = if bwd_info { 4 } else { 1 };

        let out_img = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([img_size.y as usize, img_size.x as usize, channels]),
            if bwd_info { DType::F32 } else { DType::U32 },
        );

        let visible_shape = if bwd_info {
            Shape::new([num_points])
        } else {
            Shape::new([1])
        };

        let projected_splats = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, proj_size]),
            DType::F32,
        );
        let uniforms_buffer = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([uniforms_size]),
            DType::U32,
        );
        let global_from_compact_gid = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points]),
            DType::U32,
        );
        let num_visible =
            TensorIr::uninit(client.create_empty_handle(), Shape::new([1]), DType::U32);
        let visible = TensorIr::uninit(client.create_empty_handle(), visible_shape, DType::F32);

        let input_tensors = [means, log_scales, quats, sh_coeffs, opacity];
        let stream = OperationStreams::with_inputs(&input_tensors);
        let desc = CustomOpIr::new(
            "render_splats",
            &input_tensors.map(|t| t.into_ir()),
            &[
                out_img,
                projected_splats,
                uniforms_buffer,
                global_from_compact_gid,
                num_visible,
                visible,
            ],
        );
        let op = CustomOp {
            cam: cam.clone(),
            img_size,
            bwd_info,
            background,
            render_mode,
            desc: desc.clone(),
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [
            // Img
            out_img,
            // Aux
            projected_splats,
            uniforms_buffer,
            global_from_compact_gid,
            num_visible,
            visible,
        ] = outputs;

        (
            out_img,
            RenderAux::<Self> {
                projected_splats,
                uniforms_buffer,
                global_from_compact_gid,
                num_visible,
                visible,
                img_size,
            },
        )
    }
}
