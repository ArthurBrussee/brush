#![recursion_limit = "256"]

#[cfg(not(target_family = "wasm"))]
mod convert {
    use burn::module::Module;
    use burn::record::HalfPrecisionSettings;
    use burn::tensor::backend::Backend;
    use burn_store::ModuleSnapshot;
    use lpips::LpipsModel;

    pub fn convert_lpips<B: Backend>(device: &B::Device) {
        let mut store = burn_store::pytorch::PytorchStore::from_file("./lpips_vgg_remapped.pth");
        let mut model = LpipsModel::<B>::new(device);
        model.load_from(&mut store).expect("Failed to load model");

        let recorder = burn::record::BinFileRecorder::<HalfPrecisionSettings>::new();
        model
            .save_file("./burn_mapped", &recorder)
            .expect("Failed to convert model");
    }
}

fn main() {
    #[cfg(not(target_family = "wasm"))]
    {
        println!("Converting LPIPS PyTorch model to Burn format...");
        convert::convert_lpips::<burn::backend::Wgpu>(&burn::backend::wgpu::WgpuDevice::default());
        println!("Conversion completed successfully!");
    }
}
