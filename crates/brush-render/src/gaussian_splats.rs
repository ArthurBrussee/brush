use crate::{
    bounding_box::BoundingBox, camera::Camera, render::sh_coeffs_for_degree,
    safetensor_utils::safetensor_to_burn, Backend,
};
use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    tensor::{activation::sigmoid, Shape, Tensor, TensorData, TensorPrimitive},
};
use glam::{Quat, Vec3};
use kiddo::{KdTree, SquaredEuclidean};
use rand::Rng;
use safetensors::SafeTensors;

#[derive(Config)]
pub struct RandomSplatsConfig {
    #[config(default = 10000)]
    init_count: usize,
    #[config(default = 0)]
    sh_degree: u32,
}

#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub means: Param<Tensor<B, 2>>,
    pub sh_coeffs: Param<Tensor<B, 3>>,
    pub rotation: Param<Tensor<B, 2>>,
    pub raw_opacity: Param<Tensor<B, 1>>,
    pub log_scales: Param<Tensor<B, 2>>,

    // Dummy input to track screenspace gradient.
    pub xys_dummy: Tensor<B, 2>,
}

pub fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

impl<B: Backend> Splats<B> {
    pub fn from_random_config(
        config: RandomSplatsConfig,
        bounds: BoundingBox,
        rng: &mut impl Rng,
        device: &B::Device,
    ) -> Self {
        let num_points = config.init_count;

        let min = bounds.min();
        let max = bounds.max();

        let mut positions: Vec<Vec3> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let x = rng.gen_range(min.x..max.x);
            let y = rng.gen_range(min.y..max.y);
            let z = rng.gen_range(min.z..max.z);
            positions.push(Vec3::new(x, y, z));
        }

        let mut colors: Vec<f32> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let r = rng.gen_range(0.0..1.0);
            let g = rng.gen_range(0.0..1.0);
            let b = rng.gen_range(0.0..1.0);
            colors.push(r);
            colors.push(g);
            colors.push(b);
        }

        Splats::from_raw(positions, None, None, Some(colors), None, device)
    }

    pub fn from_raw(
        means: Vec<Vec3>,
        rotations: Option<Vec<Quat>>,
        log_scales: Option<Vec<Vec3>>,
        sh_coeffs: Option<Vec<f32>>,
        raw_opacities: Option<Vec<f32>>,
        device: &B::Device,
    ) -> Splats<B> {
        let n_splats = means.len();

        let means_tensor: Vec<f32> = means.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let means_tensor = Tensor::from_data(TensorData::new(means_tensor, [n_splats, 3]), device);

        let rotations = if let Some(rotations) = rotations {
            let rotations: Vec<f32> = rotations
                .into_iter()
                .flat_map(|v| [v.w, v.x, v.y, v.z])
                .collect();
            Tensor::from_data(TensorData::new(rotations, [n_splats, 4]), device)
        } else {
            Tensor::<_, 1>::from_floats([1.0, 0.0, 0.0, 0.0], device)
                .unsqueeze::<2>()
                .repeat_dim(0, n_splats)
        };

        let log_scales = if let Some(log_scales) = log_scales {
            let log_scales: Vec<f32> = log_scales
                .into_iter()
                .flat_map(|v| [v.x, v.y, v.z])
                .collect();
            Tensor::from_data(TensorData::new(log_scales, [n_splats, 3]), device)
        } else {
            let tree_pos: Vec<[f32; 3]> = means.iter().map(|v| [v.x, v.y, v.z]).collect();
            let tree: KdTree<_, 3> = (&tree_pos).into();
            let extents: Vec<_> = tree_pos
                .iter()
                .map(|p| {
                    // Get average of 3 nearest squared distances.
                    (tree
                        .nearest_n::<SquaredEuclidean>(p, 4)
                        .iter()
                        .map(|x| x.distance)
                        .sum::<f32>()
                        / 4.0)
                        .sqrt()
                        .max(1e-12)
                        .ln()
                })
                .collect();

            Tensor::<B, 1>::from_floats(extents.as_slice(), device)
                .reshape([n_splats, 1])
                .repeat_dim(1, 3)
        };

        let sh_coeffs = if let Some(sh_coeffs) = sh_coeffs {
            let n_coeffs = sh_coeffs.len() / n_splats;
            Tensor::from_data(
                TensorData::new(sh_coeffs, [n_splats, n_coeffs / 3, 3]),
                device,
            )
        } else {
            Tensor::<_, 1>::from_floats([0.5, 0.5, 0.5], device)
                .unsqueeze::<3>()
                .repeat_dim(0, n_splats)
        };

        let raw_opacities = if let Some(raw_opacities) = raw_opacities {
            Tensor::from_data(TensorData::new(raw_opacities, [n_splats]), device).require_grad()
        } else {
            Tensor::ones(Shape::new([n_splats]), device) * inverse_sigmoid(0.1)
        };

        Self::from_tensor_data(
            means_tensor,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
        )
    }

    pub fn with_sh_degree(mut self, sh_degree: u32) -> Self {
        let n_coeffs = sh_coeffs_for_degree(sh_degree) as usize;

        let [n, c, _] = self.sh_coeffs.dims();

        if self.sh_coeffs.dims()[1] < n_coeffs {
            Splats::map_param(&mut self.sh_coeffs, |coeffs| {
                let device = coeffs.device();
                Tensor::cat(
                    vec![coeffs, Tensor::zeros([n, n_coeffs - c, 3], &device)],
                    1,
                )
            });
        }

        self
    }

    pub fn from_tensor_data(
        means: Tensor<B, 2>,
        rotation: Tensor<B, 2>,
        log_scales: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 3>,
        raw_opacity: Tensor<B, 1>,
    ) -> Self {
        assert_eq!(means.dims()[1], 3);
        assert_eq!(rotation.dims()[1], 4);
        assert_eq!(log_scales.dims()[1], 3);

        let num_points = means.shape().dims[0];
        let device = means.device();

        Splats {
            means: Param::initialized(ParamId::new(), means.detach().require_grad()),
            sh_coeffs: Param::initialized(
                ParamId::new(),
                sh_coeffs.clone().detach().require_grad(),
            ),
            rotation: Param::initialized(ParamId::new(), rotation.detach().require_grad()),
            raw_opacity: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            log_scales: Param::initialized(ParamId::new(), log_scales.detach().require_grad()),
            xys_dummy: Tensor::zeros([num_points, 2], &device).require_grad(),
        }
    }

    pub fn map_param<const D: usize>(
        tensor: &mut Param<Tensor<B, D>>,
        f: impl Fn(Tensor<B, D>) -> Tensor<B, D>,
    ) {
        *tensor = tensor.clone().map(|x| f(x).detach().require_grad());
    }

    pub fn render(
        &self,
        camera: &Camera,
        img_size: glam::UVec2,
        render_u32_buffer: bool,
    ) -> (Tensor<B, 3>, crate::RenderAux<B>) {
        let (img, aux) = B::render_splats(
            camera,
            img_size,
            self.means.val().into_primitive().tensor(),
            self.xys_dummy.clone().into_primitive().tensor(),
            self.log_scales.val().into_primitive().tensor(),
            self.rotation.val().into_primitive().tensor(),
            self.sh_coeffs.val().into_primitive().tensor(),
            self.raw_opacity.val().into_primitive().tensor(),
            render_u32_buffer,
        );

        (Tensor::from_primitive(TensorPrimitive::Float(img)), aux)
    }

    pub fn opacity(&self) -> Tensor<B, 1> {
        sigmoid(self.raw_opacity.val())
    }

    pub fn scales(&self) -> Tensor<B, 2> {
        self.log_scales.val().exp()
    }

    pub fn num_splats(&self) -> usize {
        self.means.dims()[0]
    }

    pub fn norm_rotations(&mut self) {
        Self::map_param(&mut self.rotation, |x| {
            x.clone() / Tensor::clamp_min(Tensor::sum_dim(x.powf_scalar(2.0), 1).sqrt(), 1e-6)
        });
    }

    pub fn from_safetensors(tensors: &SafeTensors, device: &B::Device) -> anyhow::Result<Self> {
        Ok(Self::from_tensor_data(
            safetensor_to_burn::<B, 2>(tensors.tensor("means")?, device),
            safetensor_to_burn::<B, 2>(tensors.tensor("quats")?, device),
            safetensor_to_burn::<B, 2>(tensors.tensor("scales")?, device),
            safetensor_to_burn::<B, 3>(tensors.tensor("coeffs")?, device),
            safetensor_to_burn::<B, 1>(tensors.tensor("opacities")?, device),
        ))
    }
}
