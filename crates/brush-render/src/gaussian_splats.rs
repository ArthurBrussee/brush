use crate::{
    SplatForward,
    bounding_box::BoundingBox,
    camera::Camera,
    render_aux::RenderAux,
    sh::{sh_coeffs_for_degree, sh_degree_from_coeffs},
    validation::validate_tensor_val,
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

            let bounding_box = bounds_from_pos(0.75, &pos_data);
            let median_size = bounding_box.median_size() as f64;

            let extents: Vec<_> = tree_pos
                .iter()
                .map(|p| {
                    // Get half of the average of 2 nearest distances.
                    0.5 * tree.query().nn(p).skip(1).take(2).map(|x| x.1).sum::<f64>() / 2.0
                })
                .map(|p| p.clamp(1e-3, median_size * 0.1))
                .map(|p| p.ln() as f32)
                .flat_map(|p| [p, p, p])
                .collect();
            Tensor::<B, 1>::from_floats(extents.as_slice(), device).reshape([n_splats, 3])
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
            Tensor::ones([n_splats, 1, 3], device) * 0.5
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

    pub fn validate_values(&self) {
        let num_splats = self.num_splats();

        // Validate means (positions)
        validate_tensor_val(&self.means.val(), "means", None, None);

        // Validate raw rotations and normalized rotations
        validate_tensor_val(&self.rotation.val(), "raw_rotations", None, None);
        let rotations = self.rotations_normed();
        validate_tensor_val(&rotations, "normalized_rotations", None, None);

        // Validate pre-activation scales (log_scales) and post-activation scales
        validate_tensor_val(
            &self.log_scales.val(),
            "log_scales",
            Some(-10.0),
            Some(10.0),
        );

        let scales = self.scales();
        validate_tensor_val(&scales, "scales", Some(1e-20), Some(10000.0));

        // Validate SH coefficients
        validate_tensor_val(&self.sh_coeffs.val(), "sh_coeffs", Some(-5.0), Some(5.0));

        // Validate pre-activation opacity (raw_opacity) and post-activation opacity
        validate_tensor_val(
            &self.raw_opacity.val(),
            "raw_opacity",
            Some(-20.0),
            Some(20.0),
        );
        let opacities = self.opacities();
        validate_tensor_val(&opacities, "opacities", Some(0.0), Some(1.0));

        // Range validation if requested
        // Scales should be positive and reasonable
        validate_tensor_val(&scales, "scales", Some(1e-6), Some(100.0));

        // Normalized rotations should have unit magnitude (quaternion)
        let rot_norms = rotations.powi_scalar(2).sum_dim(1).sqrt();
        validate_tensor_val(&rot_norms, "rotation_magnitudes", Some(1e-12), Some(1000.0));

        // Additional logical checks
        assert!(num_splats > 0, "Splats must contain at least one splat");

        let [n_means, dims] = self.means.dims();
        assert_eq!(dims, 3, "Means must be 3D coordinates");
        assert_eq!(
            n_means, num_splats as usize,
            "Inconsistent number of splats in means"
        );
        let [n_rot, rot_dims] = self.rotation.dims();
        assert_eq!(rot_dims, 4, "Rotations must be quaternions (4D)");
        assert_eq!(
            n_rot, num_splats as usize,
            "Inconsistent number of splats in rotations"
        );
        let [n_scales, scale_dims] = self.log_scales.dims();
        assert_eq!(scale_dims, 3, "Scales must be 3D");
        assert_eq!(
            n_scales, num_splats as usize,
            "Inconsistent number of splats in scales"
        );
        let [n_opacity] = self.raw_opacity.dims();
        assert_eq!(
            n_opacity, num_splats as usize,
            "Inconsistent number of splats in opacity"
        );
        let [n_sh, _coeffs, sh_dims] = self.sh_coeffs.dims();
        assert_eq!(sh_dims, 3, "SH coefficients must have 3 color channels");
        assert_eq!(
            n_sh, num_splats as usize,
            "Inconsistent number of splats in SH coeffs"
        );
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

    pub async fn get_bounds(self, percentile: f32) -> BoundingBox {
        let means: Vec<f32> = self
            .means
            .val()
            .into_data_async()
            .await
            .to_vec()
            .expect("Failed to get means");

        bounds_from_pos(percentile, &means)
    }
}

fn bounds_from_pos(percentile: f32, means: &[f32]) -> BoundingBox {
    // Split into x, y, z values
    let (mut x_vals, mut y_vals, mut z_vals): (Vec<f32>, Vec<f32>, Vec<f32>) = means
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect();

    // Filter out NaN and infinite values before sorting
    x_vals.retain(|x| x.is_finite());
    y_vals.retain(|y| y.is_finite());
    z_vals.retain(|z| z.is_finite());

    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Get upper and lower percentiles.
    let lower_idx = ((1.0 - percentile) / 2.0 * x_vals.len() as f32) as usize;
    let upper_idx =
        (x_vals.len() - 1).min(((1.0 + percentile) / 2.0 * x_vals.len() as f32) as usize);

    BoundingBox::from_min_max(
        Vec3::new(x_vals[lower_idx], y_vals[lower_idx], z_vals[lower_idx]),
        Vec3::new(x_vals[upper_idx], y_vals[upper_idx], z_vals[upper_idx]),
    )
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

        #[cfg(any(feature = "debug-validation", test))]
        self.validate_values();

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
        #[cfg(any(feature = "debug-validation", test))]
        aux.validate_values();
        (img, aux)
    }
}
