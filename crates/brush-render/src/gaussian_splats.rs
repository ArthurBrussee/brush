use crate::{
    SplatForward,
    bounding_box::BoundingBox,
    camera::Camera,
    render_aux::RenderAux,
    sh::{sh_coeffs_for_degree, sh_degree_from_coeffs},
};
use ball_tree::BallTree;
use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    prelude::Backend,
    tensor::{
        Tensor, TensorData, TensorPrimitive, activation::sigmoid, backend::AutodiffBackend, s,
    },
};
use glam::Vec3;
use rand::Rng;

#[derive(Config)]
pub struct RandomSplatsConfig {
    #[config(default = 10000)]
    init_count: usize,
}

#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub means: Param<Tensor<B, 2>>,
    pub rotation: Param<Tensor<B, 2>>,
    pub log_scales: Param<Tensor<B, 2>>,
    pub sh_coeffs: Param<Tensor<B, 3>>,
    pub raw_opacity: Param<Tensor<B, 1>>,
}

fn norm_vec<B: Backend>(vec: Tensor<B, 2>) -> Tensor<B, 2> {
    let magnitudes =
        Tensor::clamp_min(Tensor::sum_dim(vec.clone().powi_scalar(2), 1).sqrt(), 1e-32);
    vec / magnitudes
}

pub fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

impl<B: Backend> Splats<B> {
    pub fn from_random_config(
        config: &RandomSplatsConfig,
        bounds: BoundingBox,
        rng: &mut impl Rng,
        device: &B::Device,
    ) -> Self {
        let num_points = config.init_count;

        let min = bounds.min();
        let max = bounds.max();

        let mut positions: Vec<f32> = Vec::with_capacity(num_points * 3);
        for _ in 0..num_points {
            let x = rng.random_range(min.x..max.x);
            let y = rng.random_range(min.y..max.y);
            let z = rng.random_range(min.z..max.z);
            positions.extend([x, y, z]);
        }

        let mut colors: Vec<f32> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let r = rng.random_range(0.0..1.0);
            let g = rng.random_range(0.0..1.0);
            let b = rng.random_range(0.0..1.0);
            colors.push(r);
            colors.push(g);
            colors.push(b);
        }
        Self::from_raw(positions, None, None, Some(colors), None, device)
    }

    pub fn from_raw(
        pos_data: Vec<f32>,
        rot_data: Option<Vec<f32>>,
        scale_data: Option<Vec<f32>>,
        coeffs_data: Option<Vec<f32>>,
        opac_data: Option<Vec<f32>>,
        device: &B::Device,
    ) -> Self {
        let n_splats = pos_data.len() / 3;

        let log_scales = if let Some(log_scales) = scale_data {
            Tensor::from_data(TensorData::new(log_scales, [n_splats, 3]), device)
        } else {
            let tree_pos: Vec<[f64; 3]> = pos_data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
                .collect();

            let empty = vec![(); tree_pos.len()];
            let tree = BallTree::new(tree_pos.clone(), empty);

            let extents: Vec<_> = tree_pos
                .iter()
                .map(|p| {
                    // Get average of 4 nearest distances.
                    0.5 * tree.query().nn(p).skip(1).take(2).map(|x| x.1).sum::<f64>() / 2.0
                })
                .map(|p| p.max(1e-12))
                .map(|p| p.ln() as f32)
                .collect();

            Tensor::<B, 1>::from_floats(extents.as_slice(), device)
                .reshape([n_splats, 1])
                .repeat_dim(1, 3)
        };

        let means_tensor = Tensor::from_data(TensorData::new(pos_data, [n_splats, 3]), device);

        let rotations = if let Some(rotations) = rot_data {
            Tensor::from_data(TensorData::new(rotations, [n_splats, 4]), device)
        } else {
            norm_vec(Tensor::random(
                [n_splats, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            ))
        };

        let sh_coeffs = if let Some(sh_coeffs) = coeffs_data {
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

        let raw_opacities = if let Some(raw_opacities) = opac_data {
            Tensor::from_data(TensorData::new(raw_opacities, [n_splats]), device).require_grad()
        } else {
            Tensor::random(
                [n_splats],
                burn::tensor::Distribution::Uniform(
                    inverse_sigmoid(0.1) as f64,
                    inverse_sigmoid(0.25) as f64,
                ),
                device,
            )
        };

        Self::from_tensor_data(
            means_tensor,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
        )
    }

    /// Set the SH degree of this splat to be equal to `sh_degree`
    pub fn with_sh_degree(mut self, sh_degree: u32) -> Self {
        let n_coeffs = sh_coeffs_for_degree(sh_degree) as usize;

        let [n, cur_coeffs, _] = self.sh_coeffs.dims();

        self.sh_coeffs = self.sh_coeffs.map(|coeffs| {
            let device = coeffs.device();
            let tens = if cur_coeffs < n_coeffs {
                Tensor::cat(
                    vec![
                        coeffs,
                        Tensor::zeros([n, n_coeffs - cur_coeffs, 3], &device),
                    ],
                    1,
                )
            } else {
                coeffs.slice(s![.., 0..n_coeffs])
            };
            tens.detach().require_grad()
        });
        self
    }

    pub fn from_tensor_data(
        means: Tensor<B, 2>,
        rotation: Tensor<B, 2>,
        log_scales: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 3>,
        raw_opacity: Tensor<B, 1>,
    ) -> Self {
        assert_eq!(means.dims()[1], 3, "Means must be 3D");
        assert_eq!(rotation.dims()[1], 4, "Rotation must be 4D");
        assert_eq!(log_scales.dims()[1], 3, "Scales must be 3D");

        Self {
            means: Param::initialized(ParamId::new(), means.detach().require_grad()),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs.detach().require_grad()),
            rotation: Param::initialized(ParamId::new(), rotation.detach().require_grad()),
            raw_opacity: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            log_scales: Param::initialized(ParamId::new(), log_scales.detach().require_grad()),
        }
    }

    pub fn opacities(&self) -> Tensor<B, 1> {
        sigmoid(self.raw_opacity.val())
    }

    pub fn scales(&self) -> Tensor<B, 2> {
        self.log_scales.val().exp()
    }

    pub fn num_splats(&self) -> u32 {
        self.means.dims()[0] as u32
    }

    pub fn rotations_normed(&self) -> Tensor<B, 2> {
        norm_vec(self.rotation.val())
    }

    pub fn with_normed_rotations(mut self) -> Self {
        self.rotation = self.rotation.map(|r| norm_vec(r));
        self
    }

    pub fn sh_degree(&self) -> u32 {
        let [_, coeffs, _] = self.sh_coeffs.dims();
        sh_degree_from_coeffs(coeffs as u32)
    }

    pub fn device(&self) -> B::Device {
        self.means.device()
    }

    pub async fn estimate_bounds(&self) -> BoundingBox {
        let means = self
            .means
            .val()
            .into_data_async()
            .await
            .into_vec::<f32>()
            .expect("Failed to convert means");

        let vec3_means: Vec<Vec3> = means
            .chunks_exact(3)
            .map(|chunk| Vec3::new(chunk[0], chunk[1], chunk[2]))
            .collect();

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for pos in &vec3_means {
            min = min.min(*pos);
            max = max.max(*pos);
        }

        BoundingBox::from_min_max(min, max)
    }

    // TODO: This should probably exist in Burn. Maybe make a PR.
    pub fn into_autodiff<BDiff: AutodiffBackend<InnerBackend = B>>(self) -> Splats<BDiff> {
        let (means_id, means, _) = self.means.consume();
        let (rotation_id, rotation, _) = self.rotation.consume();
        let (log_scales_id, log_scales, _) = self.log_scales.consume();
        let (sh_coeffs_id, sh_coeffs, _) = self.sh_coeffs.consume();
        let (raw_opacity_id, raw_opacity, _) = self.raw_opacity.consume();

        Splats::<BDiff> {
            means: Param::initialized(means_id, Tensor::from_inner(means).require_grad()),
            rotation: Param::initialized(rotation_id, Tensor::from_inner(rotation).require_grad()),
            log_scales: Param::initialized(
                log_scales_id,
                Tensor::from_inner(log_scales).require_grad(),
            ),
            sh_coeffs: Param::initialized(
                sh_coeffs_id,
                Tensor::from_inner(sh_coeffs).require_grad(),
            ),
            raw_opacity: Param::initialized(
                raw_opacity_id,
                Tensor::from_inner(raw_opacity).require_grad(),
            ),
        }
    }
}

impl<B: Backend + SplatForward<B>> Splats<B> {
    /// Render the splats.
    ///
    /// NB: This doesn't work on a differentiable backend.
    pub fn render(
        &self,
        camera: &Camera,
        img_size: glam::UVec2,
        background: Vec3,
        splat_scale: Option<f32>,
    ) -> (Tensor<B, 3>, RenderAux<B>) {
        let mut scales = self.log_scales.val();

        // Add in scaling if needed.
        if let Some(scale) = splat_scale {
            scales = scales + scale.ln();
        };

        let (img, aux) = B::render_splats(
            camera,
            img_size,
            self.means.val().into_primitive().tensor(),
            scales.into_primitive().tensor(),
            self.rotation.val().into_primitive().tensor(),
            self.sh_coeffs.val().into_primitive().tensor(),
            self.raw_opacity.val().into_primitive().tensor(),
            background,
            false,
        );
        let img = Tensor::from_primitive(TensorPrimitive::Float(img));
        #[cfg(feature = "debug-validation")]
        aux.debug_assert_valid();
        (img, aux)
    }
}
