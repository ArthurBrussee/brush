use std::f64::consts::SQRT_2;

use burn::{
    config::Config,
    module::Module,
    nn::{
        Initializer, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::Backend,
    tensor::{Device, Tensor},
};

struct ConvReluConfig {
    conv: Conv2dConfig,
}

impl ConvReluConfig {
    /// Create a new instance of the residual block [config](BasicBlockConfig).
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        // conv3x3
        let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        Self { conv }
    }

    /// Initialize a new [basic residual block](BasicBlock) module.
    fn init<B: Backend>(&self, device: &Device<B>) -> ConvRelu<B> {
        // Conv initializer.
        // TODO: Make sure this is lazy.
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };

        ConvRelu {
            conv: self
                .conv
                .clone()
                .with_initializer(initializer.clone())
                .init(device),
            relu: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvRelu<B: Backend> {
    conv: Conv2d<B>,
    relu: Relu,
}

impl<B: Backend> ConvRelu<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        self.relu.forward(out)
    }
}

/// [Residual layer block](LayerBlock) configuration.
#[derive(Config)]
pub struct VggBlockConfig {
    num_blocks: usize,
    in_channels: usize,
    out_channels: usize,
}

impl VggBlockConfig {
    /// Initialize a new [LayerBlock](LayerBlock) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> VggBlock<B> {
        let convs = (0..self.num_blocks)
            .map(|b| {
                ConvReluConfig::new(
                    if b == 0 {
                        self.in_channels
                    } else {
                        self.out_channels
                    },
                    self.out_channels,
                    1,
                )
                .init(device)
            })
            .collect();

        VggBlock {
            convs,
            max_pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct VggBlock<B: Backend> {
    /// A bottleneck residual block.
    convs: Vec<ConvRelu<B>>,
    max_pool: MaxPool2d,
}

impl<B: Backend> VggBlock<B> {
    pub(crate) fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut cur = input;
        for conv in &self.convs {
            cur = conv.forward(cur);
        }
        cur = self.max_pool.forward(cur);
        cur
    }
}
