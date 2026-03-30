use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        Int,
        ops::{FloatTensor, IntTensor},
    },
};

use crate::shaders::helpers::ProjectUniforms;

/// Full output from the rendering pipeline.
#[derive(Debug, Clone)]
pub struct RenderOutput<B: Backend> {
    pub out_img: FloatTensor<B>,
    pub aux: RenderAux<B>,
    // State needed by the backward pass; non-diff callers can ignore these.
    pub projected_splats: FloatTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub project_uniforms: ProjectUniforms,
    pub global_from_compact_gid: IntTensor<B>,
}

impl<B: Backend> RenderOutput<B> {
    /// Validate all outputs. Debug-only, takes self by value to avoid Send issues.
    #[allow(unused_variables)]
    pub async fn validate(self) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            #[cfg(not(target_family = "wasm"))]
            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            let num_visible = self.aux.num_visible;
            let total_splats = self.project_uniforms.total_splats;

            // Validate cull outputs.
            assert!(
                num_visible <= total_splats,
                "num_visible ({num_visible}) > total_splats ({total_splats})"
            );

            if total_splats > 0 && num_visible > 0 {
                let global_from_compact_gid: Tensor<B, 1, Int> =
                    Tensor::from_primitive(self.global_from_compact_gid);
                let global_from_compact_gid = global_from_compact_gid
                    .into_data_async()
                    .await
                    .expect("readback")
                    .into_vec::<u32>()
                    .expect("Failed to fetch global_from_compact_gid");
                let global_from_compact_gid = &global_from_compact_gid[0..num_visible as usize];

                for &global_gid in global_from_compact_gid {
                    assert!(
                        global_gid < total_splats,
                        "Invalid gaussian ID in global_from_compact_gid: {global_gid} >= {total_splats}"
                    );
                }
            }

            // Validate rasterize outputs.
            use crate::validation::validate_tensor_val;
            use burn::tensor::TensorPrimitive;

            let visible: Tensor<B, 1> =
                Tensor::from_primitive(TensorPrimitive::Float(self.aux.visible));
            let visible_2d: Tensor<B, 2> = visible.unsqueeze_dim(1);
            validate_tensor_val(visible_2d, "visible", None, None).await;

            let tile_offsets: Tensor<B, 3, Int> = Tensor::from_primitive(self.aux.tile_offsets);
            let tile_offsets_data = tile_offsets
                .into_data_async()
                .await
                .expect("readback")
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

/// Minimal output from rendering. Contains only what callers typically need.
#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    pub num_visible: u32,
    pub num_intersections: u32,
    pub visible: FloatTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> RenderAux<B> {
    /// Get `num_visible` count.
    pub fn get_num_visible(&self) -> u32 {
        self.num_visible
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
}
