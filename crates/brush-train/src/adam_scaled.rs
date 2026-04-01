use burn::{
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::LearningRate,
    optim::{
        SimpleOptimizer,
        adaptor::OptimizerAdaptor,
        decay::{WeightDecay, WeightDecayConfig},
    },
    prelude::Backend,
    record::Record,
    tensor::{Device, ElementConversion, Tensor, backend::AutodiffBackend},
};

/// Adam optimizer with optional per-parameter second moment reduction.
///
/// Second moment reduction is controlled per parameter via the
/// `reduce_moment_2` flag on [`AdamState`]. When enabled for a parameter,
/// `moment_2` stores a single scalar per row instead of per-element, saving
/// memory.
#[derive(Clone)]
pub(crate) struct AdamScaled {
    momentum: AdaptiveMomentum,
    weight_decay: Option<WeightDecay>,
}

#[derive(Config, Debug)]
pub(crate) struct AdamScaledConfig {
    #[config(default = 0.9)]
    beta_1: f32,
    #[config(default = 0.999)]
    beta_2: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    weight_decay: Option<WeightDecayConfig>,
    grad_clipping: Option<GradientClippingConfig>,
}

#[derive(Clone)]
struct AdaptiveMomentum {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

/// Per-parameter momentum state.
///
/// When the owning [`AdamState`] has `reduce_moment_2` set, `moment_2` has
/// size 1 in all trailing dimensions (e.g. \[N, 1, 1\] for D=3), while
/// `moment_1` retains the full shape. Operations in [`map_opt`](super::map_opt)
/// must be shape-agnostic along trailing dims to handle both tensors correctly.
#[derive(Record, Clone)]
pub(crate) struct MomentumState<B: Backend, const D: usize> {
    pub moment_1: Tensor<B, D>,
    pub moment_2: Tensor<B, D>,
    pub time: usize,
}

impl<B: Backend, const D: usize> MomentumState<B, D> {
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(self, device: &Device<B>) -> Self {
        Self {
            moment_1: self.moment_1.to_device(device),
            moment_2: self.moment_2.to_device(device),
            time: self.time,
        }
    }
}

/// Per-parameter optimizer state.
#[derive(Record, Clone)]
pub(crate) struct AdamState<B: Backend, const D: usize> {
    pub momentum: Option<MomentumState<B, D>>,
    /// Per-component learning rate scaling (e.g. different LR for means vs
    /// rotations vs scales within the transforms tensor).
    pub scaling: Option<Tensor<B, D>>,
    /// When true, the second moment is reduced to a scalar per row. Set by the
    /// caller when initializing state for parameters where per-element variance
    /// is not needed (e.g. SH coefficients). Lives in state (rather than on the
    /// optimizer) because `SimpleOptimizer` has no per-parameter config mechanism.
    pub reduce_moment_2: bool,
}

impl AdamScaledConfig {
    pub(crate) fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
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
        let reduce = state.as_ref().is_some_and(|s| s.reduce_moment_2);

        if let Some(state) = state {
            state_momentum = state.momentum;
            scaling = state.scaling;
        }

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        let (grad, state_momentum) = self.momentum.transform(&grad, state_momentum, reduce);

        let state = AdamState {
            momentum: Some(state_momentum),
            scaling: scaling.clone(),
            reduce_moment_2: reduce,
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

/// Reduce to a single mean per row by averaging across all trailing dims (1..D).
/// Result has size 1 in each trailing dim so it broadcasts back to the full shape.
fn mean_trailing_dims<B: Backend, const D: usize>(t: Tensor<B, D>) -> Tensor<B, D> {
    debug_assert!(D > 1, "mean_trailing_dims requires D > 1");
    let shape = t.dims();
    let n = shape[0];
    let trailing_count: usize = shape[1..].iter().product();

    // Single flatten + sum avoids one kernel launch per trailing dim.
    let flat: Tensor<B, 2> = t.flatten(1, D - 1);
    let reduced: Tensor<B, 2> = flat.sum_dim(1) / trailing_count as f32;

    let mut target = [1usize; D];
    target[0] = n;
    reduced.reshape(target)
}

impl AdaptiveMomentum {
    fn transform<B: Backend, const D: usize>(
        &self,
        grad: &Tensor<B, D>,
        momentum_state: Option<MomentumState<B, D>>,
        reduce_moment_2: bool,
    ) -> (Tensor<B, D>, MomentumState<B, D>) {
        let grad_sq = grad.clone().powi_scalar(2);
        let grad_sq_for_moment = if reduce_moment_2 && D > 1 {
            mean_trailing_dims(grad_sq)
        } else {
            grad_sq
        };

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
                .add(grad_sq_for_moment.mul_scalar(factor));

            state.time += 1;
            state
        } else {
            let factor = 1.0 - self.beta_1;
            let moment_1 = grad.clone().mul_scalar(factor);

            let factor = 1.0 - self.beta_2;
            let moment_2 = grad_sq_for_moment.mul_scalar(factor);

            MomentumState {
                moment_1,
                moment_2,
                time: 1,
            }
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
        // moment_2_corrected broadcasts when it has reduced trailing dims
        let grad = moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));
        (grad, state)
    }
}
