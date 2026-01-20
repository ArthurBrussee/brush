use burn::tensor::ElementConversion;
use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        Int,
        ops::{FloatTensor, IntTensor},
    },
};

use crate::shaders::helpers::ProjectUniforms;

/// Output from the project pass. Consumed by rasterize.
#[derive(Debug, Clone)]
pub struct ProjectOutput<B: Backend> {
    pub project_uniforms: ProjectUniforms,
    pub projected_splats: FloatTensor<B>,
    pub num_visible: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub cum_tiles_hit: IntTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> ProjectOutput<B> {
    /// Get the total number of intersections (sync readback).
    pub fn read_num_intersections(&self) -> u32 {
        let cum_tiles_hit: Tensor<B, 1, Int> = Tensor::from_primitive(self.cum_tiles_hit.clone());
        let total = self.project_uniforms.total_splats as usize;
        if total > 0 {
            cum_tiles_hit
                .slice([total - 1..total])
                .into_scalar()
                .elem::<u32>()
        } else {
            0
        }
    }

    /// Validate project outputs. Call before consuming.
    pub fn validate(&self) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            use crate::validation::validate_tensor_val;
            use burn::tensor::{ElementConversion, TensorPrimitive, s};

            let num_visible_tensor: Tensor<B, 1, Int> =
                Tensor::from_primitive(self.num_visible.clone());
            let total_splats = self.project_uniforms.total_splats;
            let num_visible = num_visible_tensor.into_scalar().elem::<i32>() as u32;

            assert!(
                num_visible <= total_splats,
                "num_visible ({num_visible}) > total_splats ({total_splats})"
            );

            if total_splats > 0 && num_visible > 0 {
                let projected_splats: Tensor<B, 2> =
                    Tensor::from_primitive(TensorPrimitive::Float(self.projected_splats.clone()));
                let projected_splats = projected_splats.slice(s![0..num_visible, ..]);
                validate_tensor_val(&projected_splats, "projected_splats", None, None);

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
                        "Invalid gaussian ID in global_from_compact_gid: {global_gid} >= {total_splats}"
                    );
                }
            }
        }
    }
}

/// Minimal output from rendering. Contains only what callers typically need.
#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    pub num_visible: IntTensor<B>,
    pub num_intersections: u32,
    pub visible: FloatTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> RenderAux<B> {
    /// Get `num_visible` as a tensor.
    pub fn get_num_visible(&self) -> Tensor<B, 1, Int> {
        Tensor::from_primitive(self.num_visible.clone())
    }

    /// Calculate tile depth map for visualization.
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

    /// Validate rasterize outputs.
    pub fn validate(&self) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            use crate::validation::validate_tensor_val;
            use burn::tensor::{ElementConversion, TensorPrimitive};

            let num_visible = Tensor::<B, 1, Int>::from_primitive(self.num_visible.clone())
                .into_scalar()
                .elem::<u32>();

            let visible: Tensor<B, 1> =
                Tensor::from_primitive(TensorPrimitive::Float(self.visible.clone()));
            let visible_2d: Tensor<B, 2> = visible.unsqueeze_dim(1);
            validate_tensor_val(&visible_2d, "visible", None, None);

            // Validate tile_offsets
            let tile_offsets: Tensor<B, 3, Int> = Tensor::from_primitive(self.tile_offsets.clone());
            let tile_offsets_data = tile_offsets
                .into_data()
                .into_vec::<u32>()
                .expect("Failed to fetch tile offsets");

            for i in 0..(tile_offsets_data.len() / 2) {
                let start = tile_offsets_data[i * 2];
                let end = tile_offsets_data[i * 2 + 1];
                assert!(
                    end >= start,
                    "Invalid tile offsets: start {start} > end {end}"
                );
                assert!(
                    end - start <= num_visible,
                    "Tile has more hits ({}) than visible splats ({num_visible})",
                    end - start
                );
            }
        }
    }
}
