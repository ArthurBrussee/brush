use burn::{
    LearningRate,
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{
        AdaptiveMomentumState, SimpleOptimizer,
        adaptor::OptimizerAdaptor,
        decay::{WeightDecay, WeightDecayConfig},
    },
    prelude::Backend,
    record::Record,
    tensor::{Device, ElementConversion, Tensor, backend::AutodiffBackend},
};

/// Adam optimizer as described in the paper [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf).
#[derive(Clone)]
pub struct AdamScaled {
    momentum: AdaptiveMomentum,
    weight_decay: Option<WeightDecay>,
}

/// Adam configuration.
#[derive(Config)]
pub struct AdamScaledConfig {
    /// Parameter for Adam.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Parameter for Adam.
    #[config(default = 0.999)]
    beta_2: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

#[derive(Clone)]
struct AdaptiveMomentum {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

/// Adam state.
#[derive(Record, Clone)]
pub struct AdamState<B: Backend, const D: usize> {
    /// The current adaptive momentum.
    pub momentum: Option<AdaptiveMomentumState<B, D>>,
    pub scaling: Option<Tensor<B, D>>,
}

impl AdamScaledConfig {
    /// Initialize Adam optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<AdamScaled, M, B> {
        let optim = AdamScaled {
            momentum: AdaptiveMomentum {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                epsilon: self.epsilon,
            },
            weight_decay: self.weight_decay.as_ref().map(WeightDecay::new),
        };

        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

impl<B: Backend> SimpleOptimizer<B> for AdamScaled {
    type State<const D: usize> = AdamState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let mut state_momentum = None;
        let mut scaling = None;

        if let Some(state) = state {
            state_momentum = state.momentum;
            scaling = state.scaling;
        }

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        let (grad, state_momentum) = self.momentum.transform(grad, state_momentum);

        let state = AdamState {
            momentum: Some(state_momentum),
            scaling: scaling.clone(),
        };

        let delta = if let Some(scale) = scaling {
            grad * (scale * lr).unsqueeze()
        } else {
            grad * lr
        };

        (tensor - delta, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.map(|m| m.to_device(device));
        state
    }
}

impl AdaptiveMomentum {
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        momentum_state: Option<AdaptiveMomentumState<B, D>>,
    ) -> (Tensor<B, D>, AdaptiveMomentumState<B, D>) {
        let state = if let Some(mut state) = momentum_state {
            let factor = 1.0 - self.beta_1;
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor));

            let factor = 1.0 - self.beta_2;
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.powf_scalar(2.0).mul_scalar(factor));

            state.time += 1;

            state
        } else {
            let factor = 1.0 - self.beta_1;
            let moment_1 = grad.clone().mul_scalar(factor);

            let factor = 1.0 - self.beta_2;
            let moment_2 = grad.powf_scalar(2.0).mul_scalar(factor);

            AdaptiveMomentumState::new(1, moment_1, moment_2)
        };

        let time = (state.time as i32).elem();
        let moment_1_corrected = state
            .moment_1
            .clone()
            .div_scalar(1f32 - self.beta_1.powi(time));
        let moment_2_corrected = state
            .moment_2
            .clone()
            .div_scalar(1f32 - self.beta_2.powi(time));
        let grad = moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));
        (grad, state)
    }
}
