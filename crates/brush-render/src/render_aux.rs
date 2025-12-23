use burn::{
    prelude::Backend,
    tensor::{
        ElementConversion, Int, Tensor, TensorPrimitive,
        ops::{FloatTensor, IntTensor},
        s,
    },
};

use crate::validation::validate_tensor_val;

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub num_visible: IntTensor<B>,
    pub visible: FloatTensor<B>,
    pub img_size: glam::UVec2,
}

impl<B: Backend> RenderAux<B> {
    pub fn num_visible(&self) -> Tensor<B, 1, Int> {
        Tensor::from_primitive(self.num_visible.clone())
    }

    pub fn validate_values(&self) {
        let num_visible: Tensor<B, 1, Int> = self.num_visible();
        let num_visible = num_visible.into_scalar().elem::<i32>() as u32;

        // Get total splats from projected_splats shape
        let projected_splats: Tensor<B, 2> =
            Tensor::from_primitive(TensorPrimitive::Float(self.projected_splats.clone()));
        let num_points = projected_splats.dims()[0] as u32;

        assert!(
            num_visible <= num_points,
            "Something went wrong when calculating the number of visible gaussians. {num_visible} > {num_points}"
        );

        // Projected splats is only valid up to num_visible and undefined for other values.
        if num_visible > 0 {
            let projected_splats = projected_splats.slice(s![0..num_visible]);
            validate_tensor_val(&projected_splats, "projected_splats", None, None);
        }

        let visible: Tensor<B, 2> =
            Tensor::from_primitive(TensorPrimitive::Float(self.visible.clone()));
        validate_tensor_val(&visible, "visible", None, None);

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
