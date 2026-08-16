//! TEMP diagnostic: report what the GPU actually supports, so CI tells us why
//! native MSL was or wasn't selected. Delete once we know.
#[cfg(test)]
mod tests {
    use crate::MainBackendBase;
    use burn::backend::Backend;
    use burn::tensor::DType;
    use burn_cubecl::cubecl::Runtime;
    use burn_cubecl::cubecl::ir::{ElemType, UIntKind};
    use burn_wgpu::{AutoCompiler, WgpuDevice, WgpuRuntime};

    #[test]
    fn diag_msl_selection() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::<AutoCompiler>::client(&device);
        let props = client.properties();

        panic!(
            "MSL DIAG: adapter={:?} u8_usage={:?} supports_u8={} supports_f16={} plane={:?}",
            props.identity.name,
            props.type_usage(ElemType::UInt(UIntKind::U8)),
            <MainBackendBase as Backend>::supports_dtype(&device, DType::U8),
            <MainBackendBase as Backend>::supports_dtype(&device, DType::F16),
            props.features.plane,
        );
    }
}
