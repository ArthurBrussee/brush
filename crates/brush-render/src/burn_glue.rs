use burn::tensor::{DType, Int, Tensor, Transaction, ops::FloatTensor};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::{
    Client, Fusion, FusionHandle,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;
use glam::Vec3;

use crate::{
    MainBackendBase, RenderAux, SplatOps,
    camera::Camera,
    gaussian_splats::SplatRenderMode,
    kernels::helpers::PROJECTED_LANES_USIZE,
    render::{RenderPrep, build_project_uniforms, render_main, render_prep},
    render_aux::RenderOutput,
    sh::sh_degree_from_coeffs,
    shaders,
};
use burn::prelude::Shape;

type FusionRt = FusionCubeRuntime<WgpuRuntime>;

#[derive(Debug)]
struct PrepOp {
    desc: CustomOpIr,
    project_uniforms: shaders::helpers::ProjectUniforms,
    mip_splat: bool,
}

impl Operation<FusionRt> for PrepOp {
    fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionRt>>) {
        let (inputs, outputs) = self.desc.as_fixed::<2, 6>();
        let [transforms_ir, raw_opacities_ir] = inputs;
        let [
            global_from_presort_gid_ir,
            depths_ir,
            num_visible_buf_ir,
            intersect_counts_ir,
            num_intersections_buf_ir,
            max_radius_ir,
        ] = outputs;

        let (prep, num_visible_buf, num_intersections_buf) = render_prep(
            h.get_float_tensor::<MainBackendBase>(transforms_ir),
            h.get_float_tensor::<MainBackendBase>(raw_opacities_ir),
            &self.project_uniforms,
            self.mip_splat,
        );

        h.register_int_tensor::<MainBackendBase>(
            &global_from_presort_gid_ir.id,
            prep.global_from_presort_gid,
        );
        h.register_float_tensor::<MainBackendBase>(&depths_ir.id, prep.depths);
        h.register_int_tensor::<MainBackendBase>(&num_visible_buf_ir.id, num_visible_buf);
        h.register_int_tensor::<MainBackendBase>(&intersect_counts_ir.id, prep.intersect_counts);
        h.register_int_tensor::<MainBackendBase>(
            &num_intersections_buf_ir.id,
            num_intersections_buf,
        );
        h.register_float_tensor::<MainBackendBase>(&max_radius_ir.id, prep.max_radius);
    }
}

#[derive(Debug)]
struct MainOp {
    desc: CustomOpIr,
    project_uniforms: shaders::helpers::ProjectUniforms,
    render_mode: SplatRenderMode,
    background: Vec3,
    bwd_info: bool,
    num_visible: u32,
    num_intersections: u32,
}

impl Operation<FusionRt> for MainOp {
    fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionRt>>) {
        let (inputs, outputs) = self.desc.as_fixed::<7, 7>();
        let [
            transforms_ir,
            sh_coeffs_ir,
            raw_opacities_ir,
            global_from_presort_gid_ir,
            depths_ir,
            intersect_counts_ir,
            max_radius_ir,
        ] = inputs;
        let [
            out_img_ir,
            visible_ir,
            max_radius_out_ir,
            projected_splats_ir,
            tile_offsets_ir,
            compact_gid_from_isect_ir,
            global_from_compact_gid_ir,
        ] = outputs;

        let prep = RenderPrep {
            global_from_presort_gid: h
                .get_int_tensor::<MainBackendBase>(global_from_presort_gid_ir),
            depths: h.get_float_tensor::<MainBackendBase>(depths_ir),
            intersect_counts: h.get_int_tensor::<MainBackendBase>(intersect_counts_ir),
            max_radius: h.get_float_tensor::<MainBackendBase>(max_radius_ir),
        };

        let out = render_main(
            h.get_float_tensor::<MainBackendBase>(transforms_ir),
            h.get_float_tensor::<MainBackendBase>(sh_coeffs_ir),
            h.get_float_tensor::<MainBackendBase>(raw_opacities_ir),
            prep,
            self.num_visible,
            self.num_intersections,
            self.project_uniforms,
            self.render_mode,
            self.background,
            self.bwd_info,
        );

        h.register_float_tensor::<MainBackendBase>(&out_img_ir.id, out.out_img);
        h.register_float_tensor::<MainBackendBase>(&visible_ir.id, out.aux.visible);
        h.register_float_tensor::<MainBackendBase>(&max_radius_out_ir.id, out.aux.max_radius);
        h.register_float_tensor::<MainBackendBase>(&projected_splats_ir.id, out.projected_splats);
        h.register_int_tensor::<MainBackendBase>(&tile_offsets_ir.id, out.aux.tile_offsets);
        h.register_int_tensor::<MainBackendBase>(
            &compact_gid_from_isect_ir.id,
            out.compact_gid_from_isect,
        );
        h.register_int_tensor::<MainBackendBase>(
            &global_from_compact_gid_ir.id,
            out.global_from_compact_gid,
        );
    }
}

/// Build an uninitialized fusion IR tensor of the given shape + dtype.
/// Used as a one-liner per stage-output slot.
fn uninit<const N: usize>(client: &Client<FusionRt>, shape: [usize; N], dtype: DType) -> TensorIr {
    TensorIr::uninit(client.create_empty_handle(), Shape::new(shape), dtype)
}

impl SplatOps<Self> for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        bwd_info: bool,
    ) -> RenderOutput<Self> {
        assert!(
            img_size[0] > 0 && img_size[1] > 0,
            "Can't render images with 0 size."
        );

        let client = transforms.client.clone();
        let total_splats = transforms.shape[0] as u32;
        let total_splats_sz = total_splats as usize;
        let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape[1] as u32);
        let mut project_uniforms =
            build_project_uniforms(camera, img_size, total_splats, sh_degree);
        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        // === Stage 1: project + cull (sync fusion register) ===
        let prep_streams =
            OperationStreams::with_inputs::<FusionRt, _>([&transforms, &raw_opacities]);
        let prep_desc = CustomOpIr::new(
            "render_prep",
            &[
                transforms.clone().into_ir(),
                raw_opacities.clone().into_ir(),
            ],
            &[
                uninit(&client, [total_splats_sz], DType::U32), // global_from_presort_gid
                uninit(&client, [total_splats_sz], DType::F32), // depths
                uninit(&client, [1], DType::U32),               // num_visible_buf
                uninit(&client, [total_splats_sz], DType::U32), // intersect_counts
                uninit(&client, [1], DType::U32),               // num_intersections_buf
                uninit(&client, [total_splats_sz], DType::F32), // max_radius_prep
            ],
        );
        let [
            global_from_presort_gid,
            depths,
            num_visible_buf,
            intersect_counts,
            num_intersections_buf,
            max_radius_prep,
        ] = client
            .register(
                prep_streams,
                OperationIr::Custom(prep_desc.clone()),
                PrepOp {
                    desc: prep_desc,
                    project_uniforms,
                    mip_splat,
                },
            )
            .outputs();

        // === Readback (the only `.await`) — stage 2 needs the counts ===
        let (num_visible, num_intersections) = if total_splats == 0 {
            (0, 0)
        } else {
            let data = Transaction::default()
                .register(Tensor::<Self, 1, Int>::from_primitive(num_visible_buf))
                .register(Tensor::<Self, 1, Int>::from_primitive(
                    num_intersections_buf,
                ))
                .execute_async()
                .await
                .expect("read prep counts");
            (
                data[0].clone().into_vec::<u32>().expect("num_visible")[0],
                data[1]
                    .clone()
                    .into_vec::<u32>()
                    .expect("num_intersections")[0],
            )
        };
        project_uniforms.num_visible = num_visible;

        // === Stage 2: depth sort + rasterize (sync fusion register) ===
        let num_visible_sz = (num_visible as usize).max(1);
        let num_intersections_sz = (num_intersections as usize).max(1);
        let tile_bounds = glam::UVec2::new(
            project_uniforms.tile_bounds[0],
            project_uniforms.tile_bounds[1],
        );
        let out_dim = if bwd_info { 4 } else { 1 };
        let visible_len = if bwd_info { total_splats_sz } else { 1 };

        let main_streams = OperationStreams::with_inputs::<FusionRt, _>([
            &transforms,
            &sh_coeffs,
            &raw_opacities,
            &global_from_presort_gid,
            &depths,
            &intersect_counts,
            &max_radius_prep,
        ]);
        let main_desc = CustomOpIr::new(
            "render_main",
            &[
                transforms.into_ir(),
                sh_coeffs.into_ir(),
                raw_opacities.into_ir(),
                global_from_presort_gid.into_ir(),
                depths.into_ir(),
                intersect_counts.into_ir(),
                max_radius_prep.into_ir(),
            ],
            &[
                uninit(
                    &client,
                    [img_size.y as usize, img_size.x as usize, out_dim],
                    DType::F32,
                ), // out_img
                uninit(&client, [visible_len], DType::F32), // visible
                uninit(&client, [total_splats_sz], DType::F32), // max_radius
                uninit(&client, [num_visible_sz, PROJECTED_LANES_USIZE], DType::F32), // projected_splats
                uninit(
                    &client,
                    [tile_bounds.y as usize, tile_bounds.x as usize, 2],
                    DType::U32,
                ), // tile_offsets
                uninit(&client, [num_intersections_sz], DType::U32), // compact_gid_from_isect
                uninit(&client, [num_visible_sz], DType::U32),       // global_from_compact_gid
            ],
        );
        let [
            out_img,
            visible,
            max_radius,
            projected_splats,
            tile_offsets,
            compact_gid_from_isect,
            global_from_compact_gid,
        ] = client
            .register(
                main_streams,
                OperationIr::Custom(main_desc.clone()),
                MainOp {
                    desc: main_desc,
                    project_uniforms,
                    render_mode,
                    background,
                    bwd_info,
                    num_visible,
                    num_intersections,
                },
            )
            .outputs();

        RenderOutput {
            out_img,
            aux: RenderAux {
                num_visible,
                num_intersections,
                visible,
                max_radius,
                tile_offsets,
                img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            project_uniforms,
            global_from_compact_gid,
        }
    }
}
