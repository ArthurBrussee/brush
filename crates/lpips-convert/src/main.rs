#![recursion_limit = "256"]

use burn::module::Module;
use burn::record::FullPrecisionSettings;
use burn::record::HalfPrecisionSettings;
use burn::record::Recorder;
use burn::tensor::backend::Backend;
use lpips::{LpipsModel, LpipsModelConfig};

fn convert_lpips<B: Backend>(device: &B::Device) {
    let model = LpipsModelConfig::new().init::<B>(device);
    let record: <LpipsModel<B> as Module<B>>::Record =
        burn_import::pytorch::PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(
                burn_import::pytorch::LoadArgs::new("./lpips_vgg_remapped.pth".into()),
                device,
            )
            .expect("Should decode state successfully");
    let model = model.load_record(record);
    let recorder = burn::record::BinFileRecorder::<HalfPrecisionSettings>::new();
    model
        .save_file("./burn_mapped", &recorder)
        .expect("Failed to convert model");
}

fn main() {
    println!("Converting LPIPS PyTorch model to Burn format...");
    convert_lpips::<burn::backend::Wgpu>(&burn::backend::wgpu::WgpuDevice::default());
    println!("Conversion completed successfully!");
}
