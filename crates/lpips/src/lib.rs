mod blocks;

use blocks::{VggBlock, VggBlockConfig};
use burn::record::Recorder;
use burn::{
    config::Config,
    module::Module,
    record::FullPrecisionSettings,
    tensor::{Tensor, backend::Backend},
};
use burn_import::pytorch::PyTorchFileRecorder;

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
        }
    }
}

#[derive(Module, Debug)]
pub struct LpipsModel<B: Backend> {
    blocks: Vec<VggBlock<B>>,
}

impl<B: Backend> LpipsModel<B> {
    pub fn forward(&self, patches: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut fold = patches;
        let mut res = vec![];
        for block in &self.blocks {
            fold = block.forward(fold);
            res.push(fold.clone());
        }
        res
    }

    /// Calculate the lpips. Imgs are in NCHW order. Inputs should be 0-1 normalised.
    pub fn lpips(&self, imgs_a: Tensor<B, 4>, imgs_b: Tensor<B, 4>) -> Tensor<B, 1> {
        // TODO: concatenating first might be faster.
        let patches_a = self.forward(imgs_a * 2.0 - 1.0);
        let patches_b = self.forward(imgs_b * 2.0 - 1.0);

        // Get mean of spatial values.
        let sim_values = patches_a
            .into_iter()
            .zip(patches_b)
            .map(|(p1, p2)| (p1 - p2).powi_scalar(2).mean())
            .collect();

        // Sum differences.
        let sim_values = Tensor::cat(sim_values, 0);
        sim_values.sum()
    }
}

pub fn load_vgg_lpips<B: Backend>(device: &B::Device) -> LpipsModel<B> {
    let model = LpipsModelConfig::new().init::<B>(device);
    let device = B::Device::default();
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load("./vgg_0.0.pth".into(), &device)
        .expect("Should decode state successfully");
    model.load_record(record)
}
