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
    MainBackendBase, SplatOps,
    camera::Camera,
    gaussian_splats::SplatRenderMode,
    render::calc_tile_bounds,
    render_aux::{ProjectAux, RasterizeAux},
    sh::sh_degree_from_coeffs,
    shaders::{self, helpers::ProjectUniforms},
};

impl SplatOps<Self> for Fusion<MainBackendBase> {
    fn project(
        cam: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        opacity: FloatTensor<Self>,
        render_mode: SplatRenderMode,
    ) -> ProjectAux<Self> {
        #[derive(Debug)]
        struct CustomOp {
            cam: Camera,
            img_size: glam::UVec2,
            render_mode: SplatRenderMode,
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
                    projected_splats,
                    num_visible,
                    global_from_compact_gid,
                    cum_tiles_hit,
                ] = outputs;

                let aux = MainBackendBase::project(
                    &self.cam,
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(means),
                    h.get_float_tensor::<MainBackendBase>(log_scales),
                    h.get_float_tensor::<MainBackendBase>(quats),
                    h.get_float_tensor::<MainBackendBase>(sh_coeffs),
                    h.get_float_tensor::<MainBackendBase>(opacity),
                    self.render_mode,
                );

                // Register outputs (project_uniforms is stored on ProjectAux directly)
                h.register_float_tensor::<MainBackendBase>(
                    &projected_splats.id,
                    aux.projected_splats,
                );
                h.register_int_tensor::<MainBackendBase>(&num_visible.id, aux.num_visible);
                h.register_int_tensor::<MainBackendBase>(
                    &global_from_compact_gid.id,
                    aux.global_from_compact_gid,
                );
                h.register_int_tensor::<MainBackendBase>(&cum_tiles_hit.id, aux.cum_tiles_hit);
            }
        }

        let client = means.client.clone();
        let num_points = means.shape[0];
        let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape[1] as u32);
        let tile_bounds = calc_tile_bounds(img_size);

        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / 4;

        let projected_splats = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points, proj_size]),
            DType::F32,
        );
        let num_visible =
            TensorIr::uninit(client.create_empty_handle(), Shape::new([1]), DType::U32);
        let global_from_compact_gid = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points]),
            DType::U32,
        );
        let cum_tiles_hit = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_points]),
            DType::U32,
        );

        // Create project_uniforms from camera and img_size (stored on ProjectAux directly)
        let project_uniforms = ProjectUniforms {
            viewmat: glam::Mat4::from(cam.world_to_local()).to_cols_array_2d(),
            camera_position: [cam.position.x, cam.position.y, cam.position.z, 0.0],
            focal: cam.focal(img_size).into(),
            pixel_center: cam.center(img_size).into(),
            img_size: img_size.into(),
            tile_bounds: tile_bounds.into(),
            sh_degree,
            total_splats: num_points as u32,
            pad_a: 0,
            pad_b: 0,
        };

        let input_tensors = [means, log_scales, quats, sh_coeffs, opacity];
        let stream = OperationStreams::with_inputs(&input_tensors);
        let desc = CustomOpIr::new(
            "project_prepare",
            &input_tensors.map(|t| t.into_ir()),
            &[
                projected_splats,
                num_visible,
                global_from_compact_gid,
                cum_tiles_hit,
            ],
        );
        let op = CustomOp {
            cam: cam.clone(),
            img_size,
            render_mode,
            desc: desc.clone(),
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [
            projected_splats,
            num_visible,
            global_from_compact_gid,
            cum_tiles_hit,
        ] = outputs;

        ProjectAux::<Self> {
            projected_splats,
            project_uniforms,
            num_visible,
            global_from_compact_gid,
            cum_tiles_hit,
            img_size,
        }
    }

    fn rasterize(
        project_aux: &ProjectAux<Self>,
        num_intersections: u32,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RasterizeAux<Self>) {
        #[derive(Debug)]
        struct CustomOp {
            img_size: glam::UVec2,
            num_intersections: u32,
            background: Vec3,
            bwd_info: bool,
            project_uniforms: ProjectUniforms,
            desc: CustomOpIr,
        }

        impl<BT: BoolElement> Operation<FusionCubeRuntime<WgpuRuntime, BT>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime, BT>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [
                    projected_splats,
                    num_visible,
                    global_from_compact_gid,
                    cum_tiles_hit,
                ] = inputs;
                let [out_img, tile_offsets, compact_gid_from_isect, visible] = outputs;

                let inner_aux = ProjectAux::<MainBackendBase> {
                    projected_splats: h.get_float_tensor::<MainBackendBase>(projected_splats),
                    project_uniforms: self.project_uniforms,
                    num_visible: h.get_int_tensor::<MainBackendBase>(num_visible),
                    global_from_compact_gid: h
                        .get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    cum_tiles_hit: h.get_int_tensor::<MainBackendBase>(cum_tiles_hit),
                    img_size: self.img_size,
                };

                let (img, aux) = MainBackendBase::rasterize(
                    &inner_aux,
                    self.num_intersections,
                    self.background,
                    self.bwd_info,
                );

                // Register outputs
                h.register_float_tensor::<MainBackendBase>(&out_img.id, img);
                h.register_int_tensor::<MainBackendBase>(&tile_offsets.id, aux.tile_offsets);
                h.register_int_tensor::<MainBackendBase>(
                    &compact_gid_from_isect.id,
                    aux.compact_gid_from_isect,
                );
                h.register_float_tensor::<MainBackendBase>(&visible.id, aux.visible);
            }
        }

        let client = project_aux.projected_splats.client.clone();
        let img_size = project_aux.img_size;
        let tile_bounds = calc_tile_bounds(img_size);

        let num_points = project_aux.projected_splats.shape[0];

        let channels = if bwd_info { 4 } else { 1 };
        let out_img = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([img_size.y as usize, img_size.x as usize, channels]),
            if bwd_info { DType::F32 } else { DType::U32 },
        );

        let tile_offsets = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([tile_bounds.y as usize, tile_bounds.x as usize, 2]),
            DType::U32,
        );

        // Use actual num_intersections for buffer size
        let compact_gid_from_isect = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([num_intersections as usize]),
            DType::U32,
        );

        let visible_shape = if bwd_info {
            Shape::new([num_points])
        } else {
            Shape::new([1])
        };
        let visible = TensorIr::uninit(client.create_empty_handle(), visible_shape, DType::F32);

        let input_tensors = [
            project_aux.projected_splats.clone(),
            project_aux.num_visible.clone(),
            project_aux.global_from_compact_gid.clone(),
            project_aux.cum_tiles_hit.clone(),
        ];
        let stream = OperationStreams::with_inputs(&input_tensors);
        let desc = CustomOpIr::new(
            "rasterize",
            &input_tensors.map(|t| t.into_ir()),
            &[out_img, tile_offsets, compact_gid_from_isect, visible],
        );
        let op = CustomOp {
            img_size,
            num_intersections,
            background,
            bwd_info,
            project_uniforms: project_aux.project_uniforms,
            desc: desc.clone(),
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [out_img, tile_offsets, compact_gid_from_isect, visible] = outputs;

        (
            out_img,
            RasterizeAux::<Self> {
                tile_offsets,
                compact_gid_from_isect,
                visible,
                img_size,
            },
        )
    }
}
