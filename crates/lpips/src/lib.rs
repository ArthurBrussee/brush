#![recursion_limit = "256"]

use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::record::HalfPrecisionSettings;
use burn::record::NamedMpkGzFileRecorder;
use burn::record::Recorder;
use burn::tensor::Device;
use burn::tensor::activation::relu;
use burn::{
    config::Config,
    module::Module,
    tensor::{Tensor, backend::Backend},
};

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
                let in_channels = if b == 0 {
                    self.in_channels
                } else {
                    self.out_channels
                };

                // conv3x3
                let conv = Conv2dConfig::new([in_channels, self.out_channels], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .with_bias(true);
                conv.init(device)
            })
            .collect();

        VggBlock { convs }
    }
}

#[derive(Module, Debug)]
struct VggBlock<B: Backend> {
    /// A bottleneck residual block.
    convs: Vec<Conv2d<B>>,
}

impl<B: Backend> VggBlock<B> {
    pub(crate) fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut cur = input;
        for conv in &self.convs {
            cur = relu(conv.forward(cur));
        }
        cur
    }
}

#[derive(Module, Debug)]
pub struct LpipsModel<B: Backend> {
    blocks: Vec<VggBlock<B>>,
    heads: Vec<Conv2d<B>>,
    max_pool: MaxPool2d,
}

impl<B: Backend> LpipsModel<B> {
    pub fn forward(&self, patches: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let shift = Tensor::<B, 1>::from_floats([-0.030, -0.088, -0.188], &patches.device())
            .reshape([1, 3, 1, 1]);
        let scale = Tensor::<B, 1>::from_floats([0.458, 0.448, 0.450], &patches.device())
            .reshape([1, 3, 1, 1]);

        let mut fold = (patches - shift) / scale;

        let mut res = vec![];
        for (i, block) in self.blocks.iter().enumerate() {
            if i != 0 {
                fold = self.max_pool.forward(fold);
            }

            fold = block.forward(fold);

            // Save intermediate state as normalized vec per lpips.
            let norm_factor = fold.clone().powi_scalar(2).sum_dim(1).sqrt();
            let normed = fold.clone() / (norm_factor + 1e-10);
            res.push(normed);
        }
        res
    }

    /// Calculate the lpips. Imgs are in NCHW order. Inputs should be 0-1 normalised.
    pub fn lpips(&self, imgs_a: Tensor<B, 4>, imgs_b: Tensor<B, 4>) -> Tensor<B, 1> {
        // Convert NHWC to NCHW and to [-1, 1].
        let imgs_a = imgs_a.permute([0, 3, 1, 2]);
        let imgs_b = imgs_b.permute([0, 3, 1, 2]);

        // TODO: concatenating first might be faster.
        let imgs_a = self.forward(imgs_a * 2.0 - 1.0);
        let imgs_b = self.forward(imgs_b * 2.0 - 1.0);

        let device = imgs_a[0].device();

        imgs_a.into_iter().zip(imgs_b).zip(&self.heads).fold(
            Tensor::zeros([1], &device),
            |acc, ((p1, p2), head)| {
                let diff = (p1 - p2).powi_scalar(2);
                let class = head.forward(diff);
                // Add spatial mean.
                acc + class.mean_dim(2).mean_dim(3).reshape([1])
            },
        )
    }
}

#[derive(Config)]
pub struct LpipsModelConfig {}

impl LpipsModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LpipsModel<B> {
        // Could have different variations here but just doing VGG for now.
        let blocks = [
            (2, 3, 64),
            (2, 64, 128),
            (3, 128, 256),
            (3, 256, 512),
            (3, 512, 512),
        ]
        .iter()
        .map(|&(num_blocks, in_channels, out_channels)| {
            VggBlockConfig::new(num_blocks, in_channels, out_channels).init(device)
        })
        .collect();

        let heads = [64, 128, 256, 512, 512]
            .iter()
            .map(|&channels| {
                Conv2dConfig::new([channels, 1], [1, 1])
                    .with_stride([1, 1])
                    .with_bias(false)
                    .init(device)
            })
            .collect();

        LpipsModel {
            blocks,
            heads,
            max_pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }
}

pub fn load_vgg_lpips<B: Backend>(device: &B::Device) -> LpipsModel<B> {
    let model = LpipsModelConfig::new().init::<B>(device);
    model.load_record(
        NamedMpkGzFileRecorder::<HalfPrecisionSettings>::default()
            .load("./burn_mapped".into(), device)
            .expect("Should decode state successfully"),
    )
}

#[cfg(test)]
mod tests {
    use super::load_vgg_lpips;
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::TensorData;
    use burn::tensor::{Tensor, backend::Backend};
    use image::ImageReader;

    fn image_to_tensor<B: Backend>(device: &B::Device, img: &image::DynamicImage) -> Tensor<B, 4> {
        // Convert to RGB float array
        let rgb_img = img.to_rgb32f();
        let (w, h) = rgb_img.dimensions();
        let data = TensorData::new(rgb_img.into_vec(), [1, h as usize, w as usize, 3]);
        Tensor::from_data(data, device)
    }

    #[test]
    fn test_result() -> Result<(), Box<dyn std::error::Error>> {
        let device = WgpuDevice::default();

        // Load and preprocess the images
        let image1 = ImageReader::open("./apple.png")?.decode()?;
        let image2 = ImageReader::open("./pear.png")?.decode()?;

        let apple = image_to_tensor::<Wgpu>(&device, &image1);
        let pear = image_to_tensor::<Wgpu>(&device, &image2);

        let model = load_vgg_lpips(&device);

        // Calculate LPIPS similarity score between the two images
        let similarity_score = model.lpips(apple, pear).into_scalar();

        println!("LPIPS similarity score: {similarity_score}");

        assert!((similarity_score - 0.65710217).abs() < 1e-4);

        Ok(())
    }
}
