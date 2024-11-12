use crate::{
    bounding_box::BoundingBox, camera::Camera, render::sh_coeffs_for_degree,
    safetensor_utils::safetensor_to_burn, shaders, Backend,
};
use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    tensor::{activation::sigmoid, Device, Shape, Tensor, TensorPrimitive},
};
use glam::Vec3;
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

        let mut colors: Vec<Vec3> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let r = rng.gen_range(0.0..1.0);
            let g = rng.gen_range(0.0..1.0);
            let b = rng.gen_range(0.0..1.0);
            colors.push(Vec3::new(r, g, b));
        }

        Splats::from_point_cloud(positions, colors, config.sh_degree, device)
    }

    pub fn from_point_cloud(
        positions: Vec<Vec3>,
        colors: Vec<Vec3>,
        sh_degree: u32,
        device: &B::Device,
    ) -> Splats<B> {
        let num_points = positions.len();
        let positions_arr: Vec<_> = positions.into_iter().map(|v| [v.x, v.y, v.z]).collect();
        let means: Vec<f32> = positions_arr.iter().copied().flatten().collect();
        let means = Tensor::<B, 1>::from_floats(means.as_slice(), device).reshape([num_points, 3]);

        let colors: Vec<f32> = colors.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let colors =
            Tensor::<B, 1>::from_floats(colors.as_slice(), device).reshape([num_points, 1, 3]);

        let sh_coeffs_dc = (colors - 0.5) / shaders::gather_grads::SH_C0;

        let sh_num = sh_coeffs_for_degree(sh_degree);
        let sh_coeffs = if sh_num == 1 {
            // Can't concatenate with 0 sized tensor.
            sh_coeffs_dc
        } else {
            Tensor::cat(
                vec![
                    sh_coeffs_dc,
                    Tensor::zeros([num_points, sh_num as usize - 1, 3], device),
                ],
                1,
            )
        };

        let init_rotation = Tensor::<_, 1>::from_floats([1.0, 0.0, 0.0, 0.0], device)
            .unsqueeze::<2>()
            .repeat_dim(0, num_points);

        let raw_opacities = Tensor::ones(Shape::new([num_points]), device) * inverse_sigmoid(0.1);

        let tree: KdTree<_, 3> = (&positions_arr).into();
        let extents: Vec<_> = positions_arr
            .iter()
            .map(|p| {
                // Get average of 3 nearest squared distances.
                tree.nearest_n::<SquaredEuclidean>(p, 3)
                    .iter()
                    .map(|x| x.distance)
                    .sum::<f32>()
                    .sqrt()
                    / 3.0
            })
            .collect();

        let scales = Tensor::<B, 1>::from_floats(extents.as_slice(), device)
            .reshape([num_points, 1])
            .repeat_dim(1, 3);

        let log_scales = scales.clamp(0.0000001, f32::MAX).log();

        Self::from_data(
            means,
            sh_coeffs,
            init_rotation,
            raw_opacities,
            log_scales,
            device,
        )
    }

    pub fn from_data(
        means: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 3>,
        rotation: Tensor<B, 2>,
        raw_opacity: Tensor<B, 1>,
        log_scales: Tensor<B, 2>,
        device: &Device<B>,
    ) -> Self {
        let num_points = means.shape().dims[0];
        Splats {
            means: Param::initialized(ParamId::new(), means.detach().require_grad()),
            sh_coeffs: Param::initialized(
                ParamId::new(),
                sh_coeffs.clone().detach().require_grad(),
            ),
            rotation: Param::initialized(ParamId::new(), rotation.detach().require_grad()),
            raw_opacity: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            log_scales: Param::initialized(ParamId::new(), log_scales.detach().require_grad()),
            xys_dummy: Tensor::zeros([num_points, 2], device).require_grad(),
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
        // TODO: Remove for forward only.
        let rotations = self.rotation.val();
        let norm_rot = rotations.clone() / Tensor::sum_dim(rotations.powi_scalar(2), 1).sqrt();

        let (img, aux) = B::render_splats(
            camera,
            img_size,
            self.means.val().into_primitive().tensor(),
            self.xys_dummy.clone().into_primitive().tensor(),
            self.log_scales.val().into_primitive().tensor(),
            norm_rot.into_primitive().tensor(),
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
        let means = safetensor_to_burn::<B, 2>(tensors.tensor("means")?, device);
        let log_scales = safetensor_to_burn::<B, 2>(tensors.tensor("scales")?, device);
        let sh_coeffs = safetensor_to_burn::<B, 3>(tensors.tensor("coeffs")?, device);
        let quats = safetensor_to_burn::<B, 2>(tensors.tensor("quats")?, device);
        let raw_opacity = safetensor_to_burn::<B, 1>(tensors.tensor("opacities")?, device);

        Ok(Self::from_data(
            means,
            sh_coeffs,
            quats,
            raw_opacity,
            log_scales,
            device,
        ))
    }
}
