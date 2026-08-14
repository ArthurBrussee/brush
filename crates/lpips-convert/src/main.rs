#![recursion_limit = "256"]

#[cfg(not(target_family = "wasm"))]
mod convert {
    use burn::module::{Module, ModuleMapper, Param};
    use burn::tensor::{DType, Device, Tensor};
    use burn_store::ModuleSnapshot;
    use lpips::LpipsModel;

    /// Casts every float parameter to f16 so the packed weights stay half precision.
    struct CastF16;

    impl ModuleMapper for CastF16 {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.cast(DType::F16), mapper)
        }
    }

    pub fn convert_lpips(device: &Device) {
        let mut store = burn_store::pytorch::PytorchStore::from_file("./lpips_vgg_remapped.pth");
        let mut model = LpipsModel::new(device);
        model.load_from(&mut store).expect("Failed to load model");

        model
            .map(&mut CastF16)
            .into_record()
            .save("./burn_mapped.bpk")
            .expect("Failed to convert model");
    }
}

fn main() {
    #[cfg(not(target_family = "wasm"))]
    {
        println!("Converting LPIPS PyTorch model to Burn format...");
        convert::convert_lpips(&burn::backend::wgpu::WgpuDevice::default().into());
        println!("Conversion completed successfully!");
    }
}
