use std::mem::offset_of;

use burn::{
    prelude::Backend,
    tensor::{
        ElementConversion, Int, Tensor, TensorPrimitive,
        ops::{FloatTensor, IntTensor},
        s,
    },
};

use crate::{
    INTERSECTS_UPPER_BOUND,
    render::max_intersections,
    shaders::{self, helpers::TILE_WIDTH},
    validation::validate_tensor_val,
};

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    /// The packed projected splat information, see `ProjectedSplat` in helpers.wgsl
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub num_intersections: IntTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub visible: FloatTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> RenderAux<B> {
    pub fn calc_tile_depth(&self) -> Tensor<B, 2, Int> {
        let tile_offsets: Tensor<B, 3, Int> = Tensor::from_primitive(self.tile_offsets.clone());
        let max = tile_offsets.clone().slice(s![.., .., 1]);
        let min = tile_offsets.slice(s![.., .., 0]);
        let [w, h] = self.img_size.into();
        let [ty, tx] = [h.div_ceil(TILE_WIDTH), w.div_ceil(TILE_WIDTH)];
        (max - min).reshape([ty as usize, tx as usize])
    }

    pub fn num_intersections(&self) -> Tensor<B, 1, Int> {
        Tensor::from_primitive(self.num_intersections.clone())
    }

    pub fn num_visible(&self) -> Tensor<B, 1, Int> {
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        Tensor::from_primitive(self.uniforms_buffer.clone()).slice(s![num_vis_field_offset])
    }

    pub fn validate_values(&self) {
        let num_intersects: Tensor<B, 1, Int> = self.num_intersections();
        let compact_gid_from_isect: Tensor<B, 1, Int> =
            Tensor::from_primitive(self.compact_gid_from_isect.clone());
        let num_visible: Tensor<B, 1, Int> = self.num_visible();

        let num_intersections = num_intersects.into_scalar().elem::<i32>();
        let num_points = compact_gid_from_isect.dims()[0] as u32;
        let num_visible = num_visible.into_scalar().elem::<i32>() as u32;
        let img_size = self.img_size;

        let max_intersects = max_intersections(img_size, num_points);

        assert!(
            num_intersections < max_intersects as i32,
            "Too many intersections, estimated too low of a number. {num_intersections} / {max_intersects}"
        );

        assert!(
            num_intersections < INTERSECTS_UPPER_BOUND as i32,
            "Too many intersections, Brush currently can't handle this. {num_intersections} > {INTERSECTS_UPPER_BOUND}"
        );

        assert!(
            num_visible <= num_points,
            "Something went wrong when calculating the number of visible gaussians. {num_visible} > {num_points}"
        );

        // Projected splats is only valid up to num_visible and undefined for other values.
        if num_visible > 0 {
            let projected_splats: Tensor<B, 2> =
                Tensor::from_primitive(TensorPrimitive::Float(self.projected_splats.clone()));
            let projected_splats = projected_splats.slice(s![0..num_visible]);
            validate_tensor_val(&projected_splats, "projected_splats", None, None);
        }

        let visible: Tensor<B, 2> =
            Tensor::from_primitive(TensorPrimitive::Float(self.visible.clone()));
        validate_tensor_val(&visible, "visible", None, None);

        let tile_offsets: Tensor<B, 3, Int> = Tensor::from_primitive(self.tile_offsets.clone());

        let tile_offsets = tile_offsets
            .into_data()
            .into_vec::<u32>()
            .expect("Failed to fetch tile offsets");
        for &offsets in &tile_offsets {
            assert!(
                offsets as i32 <= num_intersections,
                "Tile offsets exceed bounds. Value: {offsets}, num_intersections: {num_intersections}"
            );
        }

        if num_intersections > 0 {
            for i in 0..(tile_offsets.len() - 1) / 2 {
                // Check pairs of start/end points.
                let start = tile_offsets[i * 2] as i32;
                let end = tile_offsets[i * 2 + 1] as i32;
                assert!(
                    start < num_intersections && end <= num_intersections,
                    "Invalid elements in tile offsets. Start {start} ending at {end}"
                );
                assert!(
                    end >= start,
                    "Invalid elements in tile offsets. Start {start} ending at {end}"
                );
                assert!(
                    end - start <= num_visible as i32,
                    "One tile has more hits than total visible splats. Start {start} ending at {end}"
                );
            }
        }

        if num_intersections > 0 {
            let compact_gid_from_isect = &compact_gid_from_isect
                .slice([0..num_intersections as usize])
                .into_data()
                .into_vec::<u32>()
                .expect("Failed to fetch compact_gid_from_isect");

            for (i, &compact_gid) in compact_gid_from_isect.iter().enumerate() {
                assert!(
                compact_gid < num_visible,
                "Invalid gaussian ID in intersection buffer. {compact_gid} out of {num_visible}. At {i} out of {num_intersections} intersections. \n

                {compact_gid_from_isect:?}

                \n\n\n"
            );
            }
        }

        // assert that every ID in global_from_compact_gid is valid.
        let global_from_compact_gid: Tensor<B, 1, Int> =
            Tensor::from_primitive(self.global_from_compact_gid.clone());
        let global_from_compact_gid = &global_from_compact_gid
            .into_data()
            .into_vec::<u32>()
            .expect("Failed to fetch global_from_compact_gid")[0..num_visible as usize];

        for &global_gid in global_from_compact_gid {
            assert!(
                global_gid < num_points,
                "Invalid gaussian ID in global_from_compact_gid buffer. {global_gid} out of {num_points}"
            );
        }
    }
}
