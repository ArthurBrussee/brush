use std::mem::offset_of;

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        Int,
        ops::{FloatTensor, IntTensor},
        s,
    },
};

use crate::shaders::{self, helpers::TILE_WIDTH};

/// Output of the ProjectPrepare pass.
///
/// Contains all data needed to perform the Rasterize pass, including
/// the `cum_tiles_hit` buffer which can be used to extract the exact
/// number of intersections via sync readback.
#[derive(Debug, Clone)]
pub struct ProjectAux<B: Backend> {
    /// The packed projected splat information, see `ProjectedSplat` in helpers.wgsl
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    /// Cumulative sum of tiles hit per splat. Last element contains total num_intersections.
    pub cum_tiles_hit: IntTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> ProjectAux<B> {
    /// Extract the total number of intersections from the cum_tiles_hit buffer.
    ///
    /// This requires a sync readback from the GPU.
    #[cfg(not(target_family = "wasm"))]
    pub fn num_intersections(&self) -> u32 {
        use burn::tensor::ElementConversion;
        let cum: Tensor<B, 1, Int> = Tensor::from_primitive(self.cum_tiles_hit.clone());
        let len = cum.dims()[0];
        cum.slice(s![len - 1..len]).into_scalar().elem::<u32>()
    }

    pub fn num_visible(&self) -> Tensor<B, 1, Int> {
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        Tensor::from_primitive(self.uniforms_buffer.clone()).slice(s![num_vis_field_offset])
    }
}

/// Output of the Rasterize pass.
#[derive(Debug, Clone)]
pub struct RasterizeAux<B: Backend> {
    pub tile_offsets: IntTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub visible: FloatTensor<B>,
}

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    /// The packed projected splat information, see `ProjectedSplat` in helpers.wgsl
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub visible: FloatTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> RenderAux<B> {
    /// Combine ProjectAux and RasterizeAux into a RenderAux for backwards compatibility.
    pub fn from_parts(project: ProjectAux<B>, rasterize: RasterizeAux<B>) -> Self {
        Self {
            projected_splats: project.projected_splats,
            uniforms_buffer: project.uniforms_buffer,
            global_from_compact_gid: project.global_from_compact_gid,
            tile_offsets: rasterize.tile_offsets,
            compact_gid_from_isect: rasterize.compact_gid_from_isect,
            visible: rasterize.visible,
            img_size: project.img_size,
        }
    }
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

    pub fn num_visible(&self) -> Tensor<B, 1, Int> {
        let num_vis_field_offset = offset_of!(shaders::helpers::RenderUniforms, num_visible) / 4;
        Tensor::from_primitive(self.uniforms_buffer.clone()).slice(s![num_vis_field_offset])
    }

    pub fn validate_values(&self) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            use burn::tensor::{ElementConversion, TensorPrimitive};

            use crate::validation::validate_tensor_val;

            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            let compact_gid_from_isect: Tensor<B, 1, Int> =
                Tensor::from_primitive(self.compact_gid_from_isect.clone());
            let num_visible: Tensor<B, 1, Int> = self.num_visible();

            // Get num_intersections from the last element of the flattened tile_offsets
            // tile_offsets has shape [ty, tx, 2] where each [i, j, :] is [start, end] for that tile
            // The last element (end offset of the last tile) is the total number of intersections
            let tile_offsets_3d: Tensor<B, 3, Int> = Tensor::from_primitive(self.tile_offsets.clone());
            let [ty, tx, _] = tile_offsets_3d.dims();
            // Get the end offset of the last tile: tile_offsets[ty-1, tx-1, 1]
            let num_intersections = tile_offsets_3d
                .slice(s![ty - 1..ty, tx - 1..tx, 1..2])
                .reshape([1])
                .into_scalar()
                .elem::<i32>() as u32;

            // Get total_splats from the uniforms buffer for validation
            let total_splats_field_offset =
                offset_of!(shaders::helpers::RenderUniforms, total_splats) / 4;
            let total_splats: Tensor<B, 1, Int> =
                Tensor::from_primitive(self.uniforms_buffer.clone());
            let total_splats = total_splats
                .slice(s![total_splats_field_offset..total_splats_field_offset + 1])
                .into_scalar()
                .elem::<i32>() as u32;

            let num_visible = num_visible.into_scalar().elem::<i32>() as u32;

            assert!(
                num_visible <= total_splats,
                "Something went wrong when calculating the number of visible gaussians. {num_visible} > {total_splats}"
            );

            // Projected splats is only valid up to num_visible and undefined for other values.
            if num_visible > 0 {
                use crate::validation::validate_tensor_val;

                let projected_splats: Tensor<B, 2> =
                    Tensor::from_primitive(TensorPrimitive::Float(self.projected_splats.clone()));
                let projected_splats = projected_splats.slice(s![0..num_visible]);
                validate_tensor_val(&projected_splats, "projected_splats", None, None);
            }

            let visible: Tensor<B, 1> =
                Tensor::from_primitive(TensorPrimitive::Float(self.visible.clone()));
            let visible_2d: Tensor<B, 2> = visible.unsqueeze_dim(1);
            validate_tensor_val(&visible_2d, "visible", None, None);

            // Only validate tile_offsets when there are intersections to validate
            if num_intersections > 0 {
                let tile_offsets: Tensor<B, 3, Int> =
                    Tensor::from_primitive(self.tile_offsets.clone());

                let tile_offsets = tile_offsets
                    .into_data()
                    .into_vec::<u32>()
                    .expect("Failed to fetch tile offsets");
                for &offsets in &tile_offsets {
                    assert!(
                        offsets <= num_intersections,
                        "Tile offsets exceed bounds. Value: {offsets}, num_intersections: {num_intersections}"
                    );
                }

                for i in 0..(tile_offsets.len() - 1) / 2 {
                    // Check pairs of start/end points.
                    let start = tile_offsets[i * 2];
                    let end = tile_offsets[i * 2 + 1];
                    assert!(
                        start < num_intersections && end <= num_intersections,
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
            }

            // Skip validation of compact_gid_from_isect when shape is 0 (fusion placeholder)
            let declared_size = compact_gid_from_isect.dims()[0];
            if num_intersections > 0 && declared_size > 0 {
                let data = compact_gid_from_isect
                    .slice([0..num_intersections as usize])
                    .into_data();

                // Handle both I32 and U32 tensor types
                let compact_gid_vec: Vec<u32> = data
                    .clone()
                    .into_vec::<u32>()
                    .or_else(|_| {
                        data.into_vec::<i32>()
                            .map(|v| v.into_iter().map(|x| x as u32).collect())
                    })
                    .expect("Failed to fetch compact_gid_from_isect");

                for (i, compact_gid) in compact_gid_vec.iter().enumerate() {
                    assert!(
                *compact_gid < num_visible,
                "Invalid gaussian ID in intersection buffer. {compact_gid} out of {num_visible}. At {i} out of {num_intersections} intersections. \n

                {compact_gid_vec:?}

                \n\n\n"
            );
                }
            }

            // assert that every ID in global_from_compact_gid is valid.
            // Only validate when there are visible splats
            if num_visible > 0 && total_splats > 0 {
                let global_from_compact_gid: Tensor<B, 1, Int> =
                    Tensor::from_primitive(self.global_from_compact_gid.clone());
                let global_from_compact_gid = &global_from_compact_gid
                    .into_data()
                    .into_vec::<u32>()
                    .expect("Failed to fetch global_from_compact_gid")[0..num_visible as usize];

                for &global_gid in global_from_compact_gid {
                    assert!(
                        global_gid < total_splats,
                        "Invalid gaussian ID in global_from_compact_gid buffer. {global_gid} out of {total_splats}"
                    );
                }
            }
        }
    }
}
