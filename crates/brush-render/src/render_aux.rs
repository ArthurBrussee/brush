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
    /// Count-only invariants — cheap (no readback), always on.
    pub fn validate_counts(&self) {
        let num_visible = self.aux.num_visible;
        let num_intersections = self.aux.num_intersections;
        let total_splats = self.project_uniforms.total_splats;
        assert!(
            num_visible <= total_splats,
            "num_visible ({num_visible}) > total_splats ({total_splats})",
        );
        let max_isects = (num_visible as u64)
            * (self.project_uniforms.tile_bounds[0] as u64)
            * (self.project_uniforms.tile_bounds[1] as u64);
        assert!(
            (num_intersections as u64) <= max_isects,
            "num_intersections ({num_intersections}) > max possible {max_isects}",
        );
    }

    /// Full validation; gated on `debug-validation` feature / `cfg(test)`.
    /// Takes self by value to avoid Send issues with the async readbacks.
    #[allow(unused_variables)]
    pub async fn validate(self) {
        self.validate_counts();

        #[cfg(any(test, feature = "debug-validation"))]
        {
            #[cfg(not(target_family = "wasm"))]
            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            let num_visible = self.aux.num_visible;
            let num_intersections = self.aux.num_intersections;
            let total_splats = self.project_uniforms.total_splats;

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
                        "global_from_compact_gid has {global_gid} >= total_splats {total_splats}",
                    );
                }
                // Compact is a gather-permutation — duplicates would mean the
                // sort emitted two visible slots for one splat.
                use std::collections::HashSet;
                let mut seen = HashSet::with_capacity(global_from_compact_gid.len());
                for &gid in global_from_compact_gid {
                    assert!(
                        seen.insert(gid),
                        "duplicate global_gid {gid} in compact list"
                    );
                }
            }

            if num_intersections > 0 && num_visible > 0 {
                let compact_gid_from_isect: Tensor<B, 1, Int> =
                    Tensor::from_primitive(self.compact_gid_from_isect);
                let compact_gid_from_isect = compact_gid_from_isect
                    .into_data_async()
                    .await
                    .expect("readback")
                    .into_vec::<u32>()
                    .expect("Failed to fetch compact_gid_from_isect");
                let compact_gid_from_isect = &compact_gid_from_isect[0..num_intersections as usize];

                for &compact_gid in compact_gid_from_isect {
                    assert!(
                        compact_gid < num_visible,
                        "compact_gid_from_isect[{compact_gid}] >= num_visible {num_visible}",
                    );
                }
            }

            use crate::validation::validate_tensor_val;
            use burn::tensor::TensorPrimitive;

            // Non-BWD renders use a [1]-sized dummy `visible`; only check
            // when it's actually sized to total_splats (bwd mode).
            let visible: Tensor<B, 1> =
                Tensor::from_primitive(TensorPrimitive::Float(self.aux.visible));
            if visible.dims()[0] == total_splats as usize {
                let vis_data = visible
                    .clone()
                    .into_data_async()
                    .await
                    .expect("readback")
                    .into_vec::<f32>()
                    .expect("visible vec");
                let marks = vis_data.iter().filter(|&&v| v > 0.5).count() as u32;
                assert!(
                    marks <= num_visible,
                    "visible mask has {marks} marks > num_visible {num_visible}",
                );
                for &v in &vis_data {
                    assert!(
                        v == 0.0 || v == 1.0,
                        "visible mask has non-{{0,1}} entry: {v}"
                    );
                }
                let visible_2d: Tensor<B, 2> = visible.unsqueeze_dim(1);
                validate_tensor_val(visible_2d, "visible", None, None).await;
            }

            // Only scan the slots project_visible actually wrote to. Past
            // num_visible the buffer is unused padding (num_visible_sz =
            // max(1, num_visible) for cubecl's FastDivmod) and may hold
            // uninitialized garbage on platforms that don't zero allocations.
            if num_visible > 0 {
                let projected: Tensor<B, 2> =
                    Tensor::from_primitive(TensorPrimitive::Float(self.projected_splats));
                let projected = projected.slice([0..(num_visible as usize)]);
                validate_tensor_val(projected, "projected_splats", None, None).await;
            }

            let tile_offsets: Tensor<B, 3, Int> = Tensor::from_primitive(self.aux.tile_offsets);
            let tile_offsets_data = tile_offsets
                .into_data_async()
                .await
                .expect("readback")
                .into_vec::<u32>()
                .expect("Failed to fetch tile offsets");

            // Each intersection belongs to exactly one tile's [start, end),
            // so sum-of-ranges must equal num_intersections. Strict equality
            // is the sharp trap for PF/MG tile-count disagreement: any gap in
            // the intersection buffer sorts to a tile_id outside [0, num_tiles)
            // and drops out of the sum.
            let mut sum_isects: u64 = 0;
            for i in 0..(tile_offsets_data.len() / 2) {
                let start = tile_offsets_data[i * 2];
                let end = tile_offsets_data[i * 2 + 1];
                assert!(end >= start, "tile {i} start {start} > end {end}");
                assert!(
                    end <= num_intersections,
                    "tile {i} end {end} > num_intersections {num_intersections}",
                );
                sum_isects += (end - start) as u64;
                assert!(
                    end - start <= num_visible,
                    "tile {i} hits ({}) > num_visible ({num_visible})",
                    end - start,
                );
            }
            assert_eq!(
                sum_isects, num_intersections as u64,
                "sum of tile ranges {sum_isects} != num_intersections {num_intersections} \
                 — project_forward and map_gaussians_to_intersects disagreed on tile hits",
            );
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
