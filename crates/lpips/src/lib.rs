#![recursion_limit = "256"]

use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::record::HalfPrecisionSettings;
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

fn norm_vec<B: Backend>(vec: Tensor<B, 4>) -> Tensor<B, 4> {
    let norm_factor = vec.clone().powi_scalar(2).sum_dim(1).sqrt();
    vec / (norm_factor + 1e-10)
}

impl<B: Backend> LpipsModel<B> {
    /// Calculate the lpips. Imgs are in NCHW order. Inputs should be 0-1 normalised.
    pub fn lpips(&self, imgs_a: Tensor<B, 4>, imgs_b: Tensor<B, 4>) -> Tensor<B, 1> {
        let device = imgs_a.device();

        // Convert NHWC to NCHW and to [-1, 1].
        let imgs_a = imgs_a.permute([0, 3, 1, 2]) * 2.0 - 1.0;
        let imgs_b = imgs_b.permute([0, 3, 1, 2]) * 2.0 - 1.0;

        let shift =
            Tensor::<B, 1>::from_floats([-0.030, -0.088, -0.188], &device).reshape([1, 3, 1, 1]);
        let scale =
            Tensor::<B, 1>::from_floats([0.458, 0.448, 0.450], &device).reshape([1, 3, 1, 1]);

        let mut imgs_a = (imgs_a - shift.clone()) / scale.clone();
        let mut imgs_b = (imgs_b - shift) / scale;

        let mut loss = Tensor::<B, 1>::zeros([1], &device);
        for (i, (block, head)) in self.blocks.iter().zip(&self.heads).enumerate() {
            // TODO: concatenating first might be faster.
            if i != 0 {
                imgs_a = self.max_pool.forward(imgs_a);
                imgs_b = self.max_pool.forward(imgs_b);
            }

            // Process each part through the block
            imgs_a = block.forward(imgs_a);
            imgs_b = block.forward(imgs_b);

            let normed_a = norm_vec(imgs_a.clone());
            let normed_b = norm_vec(imgs_b.clone());

            let diff = (normed_a - normed_b).powi_scalar(2);
            let class = head.forward(diff);
            // Add spatial mean.
            loss = loss + class.mean_dim(2).mean_dim(3).reshape([1]);
        }
        loss
    }
}

impl<B: Backend> LpipsModel<B> {
    pub fn new(device: &B::Device) -> Self {
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

        Self {
            blocks,
            heads,
            max_pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
pub fn load_vgg_lpips<B: Backend>(device: &B::Device) -> LpipsModel<B> {
    use burn::record::BinBytesRecorder;

    let model = LpipsModel::<B>::new(device);
    // It's not great, but just about manageable.
    #[allow(clippy::large_include_file)]
    let bytes = include_bytes!("../burn_mapped.bin");

    model.load_record(
        BinBytesRecorder::<HalfPrecisionSettings, &[u8]>::default()
            .load(bytes, device)
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
