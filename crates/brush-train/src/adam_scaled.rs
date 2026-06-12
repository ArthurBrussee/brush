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
    record::Record,
    tensor::{Device, ElementConversion, Tensor},
};

/// Adam with per-parameter second-moment reduction (via [`AdamState::reduce_moment_2`]).
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

/// Per-parameter momentum state. When `reduce_moment_2` is set on the owning
/// [`AdamState`], `moment_2` has size 1 in trailing dims; `map_opt` callers
/// must stay shape-agnostic along those.
#[derive(Record, Clone)]
pub(crate) struct MomentumState<const D: usize> {
    pub moment_1: Tensor<D>,
    pub moment_2: Tensor<D>,
    pub time: usize,
}

impl<const D: usize> MomentumState<D> {
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(self, device: &Device) -> Self {
        Self {
            moment_1: self.moment_1.to_device(device),
            moment_2: self.moment_2.to_device(device),
            time: self.time,
        }
    }
}

/// Per-parameter optimizer state.
#[derive(Record, Clone)]
pub(crate) struct AdamState<const D: usize> {
    pub momentum: Option<MomentumState<D>>,
    /// Per-component learning rate scaling (e.g. different LR for means vs
    /// rotations vs scales within the transforms tensor).
    pub scaling: Option<Tensor<D>>,
    /// When true, the second moment is reduced to a scalar per row. Set by the
    /// caller when initializing state for parameters where per-element variance
    /// is not needed.
    pub reduce_moment_2: bool,
}

impl AdamScaledConfig {
    pub(crate) fn init<M: AutodiffModule>(&self) -> OptimizerAdaptor<AdamScaled, M> {
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

impl SimpleOptimizer for AdamScaled {
    type State<const D: usize> = AdamState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        mut grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
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

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.map(|m| m.to_device(device));
        state
    }
}

/// Reduce to a single mean per row by averaging across all trailing dims (1..D).
/// Result has size 1 in each trailing dim so it broadcasts back to the full shape.
fn mean_trailing_dims<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    debug_assert!(D > 1, "mean_trailing_dims requires D > 1");
    let shape = t.dims();
    let n = shape[0];
    let trailing_count: usize = shape[1..].iter().product();

    // Single flatten + sum avoids one kernel launch per trailing dim.
    let flat: Tensor<2> = t.flatten(1, D - 1);
    let reduced: Tensor<2> = flat.sum_dim(1) / trailing_count as f32;

    let mut target = [1usize; D];
    target[0] = n;
    reduced.reshape(target)
}

impl AdaptiveMomentum {
    fn transform<const D: usize>(
        &self,
        grad: &Tensor<D>,
        momentum_state: Option<MomentumState<D>>,
        reduce_moment_2: bool,
    ) -> (Tensor<D>, MomentumState<D>) {
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
        // moment_2_corrected broadcasts when it has reduced trailing dims.
        //
        // The clamp guards the sqrt: a second moment is non-negative by
        // construction, but a tiny negative residue can sneak in GPU-side
        // (refine's old scatter-add moment reset did exactly that, see
        // `scatter_zeroing_is_exact`), and sqrt(-ε) is NaN — which the
        // reduced-moment broadcast then smears across the whole row. A
        // negative state self-heals within a few steps via the
        // (1-β₂)·mean(g²) term, so clamping only at the read is enough.
        let grad = moment_1_corrected.div(
            moment_2_corrected
                .clamp_min(0.0)
                .sqrt()
                .add_scalar(self.epsilon),
        );
        (grad, state)
    }
}
