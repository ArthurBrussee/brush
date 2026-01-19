use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        Int,
        ops::{FloatTensor, IntTensor},
    },
};

use crate::shaders::helpers::ProjectUniforms;

/// Validate both ProjectAux and RasterizeAux outputs together.
/// This is a no-op in release builds unless `debug-validation` feature is enabled.
/// Also skipped when running benchmarks (detected via `--bench` arg).
pub fn validate_render_output<B: Backend>(project_aux: &ProjectAux<B>, rasterize_aux: &RasterizeAux<B>) {
    #[cfg(any(test, feature = "debug-validation"))]
    {
        if std::env::args().any(|a| a == "--bench") {
            return;
        }
        project_aux.validate_values();
        use burn::tensor::ElementConversion;
        let num_visible = project_aux.get_num_visible().into_scalar().elem::<u32>();
        rasterize_aux.validate_values(num_visible);
    }
    #[cfg(not(any(test, feature = "debug-validation")))]
    {
        let _ = project_aux;
        let _ = rasterize_aux;
    }
}

#[derive(Debug, Clone)]
pub struct ProjectAux<B: Backend> {
    pub project_uniforms: ProjectUniforms,

    /// The packed projected splat information, see `ProjectedSplat` in helpers.wgsl
    pub projected_splats: FloatTensor<B>,
    pub num_visible: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub cum_tiles_hit: IntTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> ProjectAux<B> {
    /// Extract the total number of intersections.
    ///
    /// This requires a sync readback from the GPU.
    pub fn num_intersections(&self) -> u32 {
        use burn::tensor::ElementConversion;
        let cum_tiles_hit: Tensor<B, 1, Int> = Tensor::from_primitive(self.cum_tiles_hit.clone());
        let total = self.project_uniforms.total_splats as usize;
        // The prefix sum is inclusive, so the last element is the total number of intersections
        if total > 0 {
            cum_tiles_hit
                .slice([total - 1..total])
                .into_scalar()
                .elem::<u32>()
        } else {
            0
        }
    }

    pub fn get_num_visible(&self) -> Tensor<B, 1, Int> {
        Tensor::from_primitive(self.num_visible.clone())
    }

    pub fn validate_values(&self) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            use burn::tensor::{ElementConversion, TensorPrimitive, s};

            use crate::validation::validate_tensor_val;

            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            let num_visible_tensor: Tensor<B, 1, Int> = self.get_num_visible();
            let total_splats = self.project_uniforms.total_splats;
            let num_visible = num_visible_tensor.into_scalar().elem::<i32>() as u32;

            assert!(
                num_visible <= total_splats,
                "Something went wrong when calculating the number of visible gaussians. {num_visible} > {total_splats}"
            );

            // Projected splats is only valid up to num_visible and undefined for other values.
            if num_visible > 0 {
                let projected_splats: Tensor<B, 2> =
                    Tensor::from_primitive(TensorPrimitive::Float(self.projected_splats.clone()));
                let projected_splats = projected_splats.slice(s![0..num_visible, ..]);
                validate_tensor_val(&projected_splats, "projected_splats", None, None);
            }

            // assert that every ID in global_from_compact_gid is valid.
            // Only validate when there are visible splats
            if num_visible > 0 && total_splats > 0 {
                let global_from_compact_gid: Tensor<B, 1, Int> =
                    Tensor::from_primitive(self.global_from_compact_gid.clone());
                let global_from_compact_gid = &global_from_compact_gid
                    .into_data()
                    .into_vec::<u32>()
                    .expect("Failed to fetch global_from_compact_gid")
                    [0..num_visible as usize];

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

/// Output of the Rasterize pass.
#[derive(Debug, Clone)]
pub struct RasterizeAux<B: Backend> {
    pub tile_offsets: IntTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub visible: FloatTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> RasterizeAux<B> {
    pub fn calc_tile_depth(&self) -> Tensor<B, 2, Int> {
        use crate::shaders::helpers::TILE_WIDTH;
        use burn::tensor::s;

        let tile_offsets: Tensor<B, 3, Int> = Tensor::from_primitive(self.tile_offsets.clone());
        let max = tile_offsets.clone().slice(s![.., .., 1]);
        let min = tile_offsets.slice(s![.., .., 0]);
        let [w, h] = self.img_size.into();
        let [ty, tx] = [h.div_ceil(TILE_WIDTH), w.div_ceil(TILE_WIDTH)];
        (max - min).reshape([ty as usize, tx as usize])
    }

    pub fn validate_values(&self, #[allow(unused)] num_visible: u32) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            use burn::tensor::TensorPrimitive;

            use crate::validation::validate_tensor_val;

            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            let compact_gid_from_isect: Tensor<B, 1, Int> =
                Tensor::from_primitive(self.compact_gid_from_isect.clone());

            let num_intersections = compact_gid_from_isect.shape()[0] as u32;

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

            // Validate compact_gid_from_isect
            // Only validate when there are visible splats (if num_visible=0, no valid intersections)
            if num_visible > 0 {
                let data = compact_gid_from_isect.into_data();

                // Handle both I32 and U32 tensor types
                let compact_gid_vec: Vec<u32> = data
                    .clone()
                    .into_vec::<u32>()
                    .or_else(|_| {
                        data.into_vec::<i32>()
                            .map(|v| v.into_iter().map(|x| x as u32).collect())
                    })
                    .expect("Failed to fetch compact_gid_from_isect");

                for compact_gid in &compact_gid_vec {
                    assert!(
                        *compact_gid < num_visible,
                        "Invalid gaussian ID in intersection buffer. {compact_gid} out of {num_visible}."
                    );
                }
            }
        }
    }
}
