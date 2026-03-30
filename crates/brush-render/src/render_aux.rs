use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        Int, Transaction,
        ops::{FloatTensor, IntTensor},
    },
};

use crate::shaders::helpers::ProjectUniforms;

/// Output from the culling + depth sort pass, consumed by `rasterize`.
#[derive(Debug, Clone)]
pub struct CullOutput<B: Backend> {
    /// Uniforms shared across all GPU passes. `sh_degree` is set to 0 here
    /// because ProjectSplats doesn't use it; it's filled in by `rasterize`.
    pub project_uniforms: ProjectUniforms,
    pub global_from_compact_gid: IntTensor<B>,
    /// Inclusive prefix sum of per-splat tile intersection counts in compact (depth) order.
    pub cum_tiles_hit: IntTensor<B>,
    pub img_size: glam::UVec2,
}

/// GPU tensors for async readback of `num_visible` and `num_intersections`.
///
/// These are separate from [`CullOutput`] because after readback they become
/// plain `u32` values and are not needed for rasterization.
#[derive(Debug, Clone)]
pub struct CullReadback<B: Backend> {
    pub num_visible: IntTensor<B>,
    pub num_intersections: IntTensor<B>,
    total_splats: u32,
}

impl<B: Backend> CullReadback<B> {
    pub fn new(num_visible: IntTensor<B>, num_intersections: IntTensor<B>, total_splats: u32) -> Self {
        Self { num_visible, num_intersections, total_splats }
    }

    /// Read `num_visible` and `num_intersections` in a single batched GPU readback.
    pub async fn read_counts(&self) -> (u32, u32) {
        if self.total_splats == 0 {
            return (0, 0);
        }

        let num_visible_tensor: Tensor<B, 1, Int> =
            Tensor::from_primitive(self.num_visible.clone());
        let num_intersections_tensor: Tensor<B, 1, Int> =
            Tensor::from_primitive(self.num_intersections.clone());

        let data = Transaction::default()
            .register(num_visible_tensor)
            .register(num_intersections_tensor)
            .execute_async()
            .await
            .expect("Failed to read counts");

        let num_visible = data[0]
            .clone()
            .into_vec::<u32>()
            .expect("Failed to read num_visible")[0];
        let num_intersections = data[1]
            .clone()
            .into_vec::<u32>()
            .expect("Failed to read num_intersections")[0];

        (num_visible, num_intersections)
    }
}

impl<B: Backend> CullOutput<B> {
    /// Validate cull outputs. Takes self by value to avoid Send issues with async.
    #[allow(unused_variables)]
    pub async fn validate(self, num_visible: u32) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            #[cfg(not(target_family = "wasm"))]
            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            let total_splats = self.project_uniforms.total_splats;

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

    /// Validate rasterize outputs. Takes self by value to avoid Send issues with async.
    pub async fn validate(self) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            #[cfg(not(target_family = "wasm"))]
            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            use crate::validation::validate_tensor_val;
            use burn::tensor::TensorPrimitive;

            let num_visible = self.num_visible;

            let visible: Tensor<B, 1> =
                Tensor::from_primitive(TensorPrimitive::Float(self.visible));
            let visible_2d: Tensor<B, 2> = visible.unsqueeze_dim(1);
            validate_tensor_val(visible_2d, "visible", None, None).await;

            let tile_offsets: Tensor<B, 3, Int> = Tensor::from_primitive(self.tile_offsets);
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
