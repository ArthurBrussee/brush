use std::mem::offset_of;

use burn::{
    prelude::Backend,
    tensor::{
        ElementConversion, Int, Tensor, TensorMetadata,
        ops::{FloatTensor, IntTensor},
        s,
    },
};

use crate::{
    GAUSSIANS_UPPER_BOUND, INTERSECTS_UPPER_BOUND,
    shaders::{self, helpers::TILE_WIDTH},
};

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    /// The packed projected splat information, see `ProjectedSplat` in helpers.wgsl
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub visible: FloatTensor<B>,
    pub final_index: IntTensor<B>,
}

impl<B: Backend> RenderAux<B> {
    #[allow(clippy::single_range_in_vec_init)]
    pub fn calc_tile_depth(&self) -> Tensor<B, 2, Int> {
        let tile_offsets: Tensor<B, 1, Int> = Tensor::from_primitive(self.tile_offsets.clone());
        let final_index: Tensor<B, 2, Int> = Tensor::from_primitive(self.final_index.clone());

        let n_bins = tile_offsets.dims()[0];
        let max = tile_offsets.clone().slice([1..n_bins]);
        let min = tile_offsets.slice([0..n_bins - 1]);
        let [h, w] = final_index.shape().dims();
        let [ty, tx] = [
            h.div_ceil(TILE_WIDTH as usize),
            w.div_ceil(TILE_WIDTH as usize),
        ];
        (max - min).reshape([ty, tx])
    }

    pub fn num_intersections(&self) -> Tensor<B, 1, Int> {
        Tensor::from_primitive(self.tile_offsets.clone()).slice(s![-1])
    }

    pub fn num_visible(&self) -> Tensor<B, 1, Int> {
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        Tensor::from_primitive(self.uniforms_buffer.clone()).slice(s![num_vis_field_offset])
    }

    pub fn debug_assert_valid(&self) {
        let num_intersects: Tensor<B, 1, Int> = self.num_intersections();
        let compact_gid_from_isect: Tensor<B, 1, Int> =
            Tensor::from_primitive(self.compact_gid_from_isect.clone());
        let num_visible: Tensor<B, 1, Int> = self.num_visible();

        let num_intersections = num_intersects.into_scalar().elem::<i32>();
        let num_points = compact_gid_from_isect.dims()[0] as u32;
        let num_visible = num_visible.into_scalar().elem::<i32>();

        assert!(
            num_intersections >= 0 && num_intersections < INTERSECTS_UPPER_BOUND as i32,
            "Too many intersections, Brush currently can't handle this. {num_intersections} > {INTERSECTS_UPPER_BOUND}"
        );

        assert!(
            num_visible >= 0 && num_visible <= num_points as i32,
            "Something went wrong when calculating the number of visible gaussians. {num_visible} > {num_points}"
        );
        assert!(
            num_visible >= 0 && num_visible < GAUSSIANS_UPPER_BOUND as i32,
            "Brush doesn't support this many gaussians currently. {num_visible} > {GAUSSIANS_UPPER_BOUND}"
        );

        if self.final_index.shape().dims() != [1, 1] {
            let final_index: Tensor<B, 2, Int> = Tensor::from_primitive(self.final_index.clone());
            let final_index = final_index
                .into_data()
                .to_vec::<i32>()
                .expect("Failed to fetch final index");
            for &final_index in &final_index {
                assert!(
                    final_index >= 0 && final_index <= num_intersections,
                    "Final index exceeds bounds. Final index {final_index}, num_intersections: {num_intersections}"
                );
            }
        }

        let tile_offsets: Tensor<B, 1, Int> = Tensor::from_primitive(self.tile_offsets.clone());

        let tile_offsets = tile_offsets
            .into_data()
            .to_vec::<i32>()
            .expect("Failed to fetch tile offsets");
        for &offsets in &tile_offsets {
            assert!(
                offsets >= 0 && offsets <= num_intersections,
                "Tile offsets exceed bounds. Value: {offsets}, num_intersections: {num_intersections}"
            );
        }

        for i in 0..tile_offsets.len() - 1 {
            let start = tile_offsets[i];
            let end = tile_offsets[i + 1];
            assert!(
                start >= 0 && end >= 0,
                "Invalid elements in tile offsets. Start {start} ending at {end}"
            );
            assert!(
                end >= start,
                "Invalid elements in tile offsets. Start {start} ending at {end}"
            );
            assert!(
                end - start <= num_visible,
                "One tile has more hits than total visible splats. Start {start} ending at {end}"
            );
        }

        if num_intersections > 0 {
            let compact_gid_from_isect = &compact_gid_from_isect
                .slice([0..num_intersections as usize])
                .into_data()
                .to_vec::<i32>()
                .expect("Failed to fetch compact_gid_from_isect");

            for (i, &compact_gid) in compact_gid_from_isect.iter().enumerate() {
                assert!(
                compact_gid >= 0 && compact_gid < num_visible,
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
            .to_vec::<i32>()
            .expect("Failed to fetch global_from_compact_gid")[0..num_visible as usize];

        for &global_gid in global_from_compact_gid {
            assert!(
                global_gid >= 0 && global_gid < num_points as i32,
                "Invalid gaussian ID in global_from_compact_gid buffer. {global_gid} out of {num_points}"
            );
        }
    }
}
