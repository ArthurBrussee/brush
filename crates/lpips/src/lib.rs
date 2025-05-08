#![recursion_limit = "256"]

use burn::nn::Initializer;
use burn::nn::PaddingConfig2d;
use burn::nn::Relu;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::tensor::Device;
use burn::{
    config::Config,
    module::Module,
    tensor::{Tensor, backend::Backend},
};
// use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use std::f64::consts::SQRT_2;

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
            conv: self.conv.clone().with_initializer(initializer).init(device),
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
struct VggBlockConfig {
    num_blocks: usize,
    in_channels: usize,
    out_channels: usize,
}

impl VggBlockConfig {
    /// Initialize a new [LayerBlock](LayerBlock) module.
    fn init<B: Backend>(&self, device: &Device<B>) -> VggBlock<B> {
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

        VggBlock { convs }
    }
}

#[derive(Module, Debug)]
struct VggBlock<B: Backend> {
    /// A bottleneck residual block.
    convs: Vec<ConvRelu<B>>,
}

impl<B: Backend> VggBlock<B> {
    pub(crate) fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut cur = input;
        for conv in &self.convs {
            cur = conv.forward(cur);
        }
        cur
    }
}

#[derive(Module, Debug)]
pub struct LpipsModel<B: Backend> {
    blocks: Vec<VggBlock<B>>,
    max_pool: MaxPool2d,
}

impl<B: Backend> LpipsModel<B> {
    pub fn forward(&self, patches: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut fold = patches;
        let mut res = vec![];
        for block in &self.blocks {
            fold = block.forward(fold);
            // Save intermediate state _before_ max pool as per the original lpips.
            res.push(fold.clone());
            fold = self.max_pool.forward(fold);
        }
        res
    }

    /// Calculate the lpips. Imgs are in NCHW order. Inputs should be 0-1 normalised.
    pub fn lpips(&self, imgs_a: Tensor<B, 4>, imgs_b: Tensor<B, 4>) -> Tensor<B, 1> {
        // Convert NHWC to NCHW
        let imgs_a = imgs_a.permute([0, 3, 1, 2]);
        let imgs_b = imgs_b.permute([0, 3, 1, 2]);

        // TODO: concatenating first might be faster.
        let imgs_a = self.forward(imgs_a * 2.0 - 1.0);
        let imgs_b = self.forward(imgs_b * 2.0 - 1.0);

        // Get mean of spatial values.
        let sim_values = imgs_a
            .into_iter()
            .zip(imgs_b)
            // TODO: 3rd dimension is learned 1x1 convs....
            .map(|(p1, p2)| {
                (p1 - p2)
                    .powf_scalar(2.0)
                    .mean_dim(1)
                    .mean_dim(2)
                    .mean_dim(3)
            })
            .collect();

        // Sum differences.
        let sim_values = Tensor::cat(sim_values, 0);
        sim_values.sum()
    }
}

#[derive(Config)]
pub struct LpipsModelConfig {}

impl LpipsModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LpipsModel<B> {
        // Could have different variations here but just doing VGG for now.
        let block1 = VggBlockConfig::new(2, 3, 64).init(device);
        let block2 = VggBlockConfig::new(2, 64, 128).init(device);
        let block3 = VggBlockConfig::new(3, 128, 256).init(device);
        let block4 = VggBlockConfig::new(3, 256, 512).init(device);
        let block5 = VggBlockConfig::new(3, 512, 512).init(device);

        LpipsModel {
            blocks: vec![block1, block2, block3, block4, block5],
            max_pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }
}

// pub fn load_vgg_lpips<B: Backend>(device: &B::Device) -> LpipsModel<B> {
//     let model = LpipsModelConfig::new().init::<B>(device);
//     let device = B::Device::default();

//     let mut load_args = LoadArgs::new("./vgg16_conv_float32.pth".into());

//     let key_mappings = [
//         // PyTorch Idx | Burn Block Idx | Burn Conv Idx in Block
//         ("0", 0, 0),
//         ("2", 0, 1),
//         ("5", 1, 0),
//         ("7", 1, 1),
//         ("10", 2, 0),
//         ("12", 2, 1),
//         ("14", 2, 2),
//         ("17", 3, 0),
//         ("19", 3, 1),
//         ("21", 3, 2),
//         ("24", 4, 0),
//         ("26", 4, 1),
//         ("28", 4, 2),
//     ];

//     for (pt_idx_str, burn_block_idx, burn_conv_idx) in &key_mappings {
//         load_args = load_args.with_key_remap(
//             &format!(r"^{pt_idx_str}\.(weight|bias)$"),
//             &format!(r"blocks.{burn_block_idx}.convs.{burn_conv_idx}.conv.$1"),
//         );
//     }

//     // TODO: Load linear logs.

//     let record: <LpipsModel<B> as Module<B>>::Record =
//         PyTorchFileRecorder::<FullPrecisionSettings>::default()
//             .load(load_args, &device)
//             .expect("Should decode state successfully");

//     println!("record {:?}", record.blocks.len());
//     model.load_record(record)
// }

// #[cfg(test)]
// mod tests {
//     use super::load_vgg_lpips;
//     use burn::backend::Wgpu;
//     use burn::backend::wgpu::WgpuDevice;
//     use burn::tensor::TensorData;
//     use burn::tensor::{Tensor, backend::Backend};
//     use image::ImageReader;

//     fn image_to_tensor<B: Backend>(device: &B::Device, img: &image::DynamicImage) -> Tensor<B, 4> {
//         // Convert to RGB float array
//         let rgb_img = img.to_rgb32f().into_vec();
//         let data = TensorData::new(rgb_img, [1, 64, 64, 3]);
//         Tensor::from_data(data, device)
//     }

//     #[test]
//     fn test_result() -> Result<(), Box<dyn std::error::Error>> {
//         let device = WgpuDevice::default();

//         // Load and preprocess the images
//         let image1 = ImageReader::open("./apple.jpg")?.decode()?;
//         let image2 = ImageReader::open("./pear.png")?.decode()?;

//         let tensor1 = image_to_tensor::<Wgpu>(&device, &image1);
//         let tensor2 = image_to_tensor::<Wgpu>(&device, &image2);

//         println!(
//             "Converted images to tensors with shape: {:?} and {:?}",
//             tensor1.shape(),
//             tensor2.shape()
//         );

//         // Load the LPIPS model
//         let model = load_vgg_lpips(&device);

//         // Calculate LPIPS similarity score between the two images
//         let similarity_score = model.lpips(tensor1, tensor2).into_scalar();

//         println!("LPIPS similarity score: {similarity_score}");

//         assert_eq!(similarity_score, 0.6731404);

//         Ok(())
//     }
// }
