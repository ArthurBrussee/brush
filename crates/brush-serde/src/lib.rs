#![recursion_limit = "256"]

#[cfg(feature = "export")]
pub mod export;
#[cfg(feature = "import")]
pub mod import;
pub mod ply_gaussian;
pub mod quant;

// Re-export main functionality
#[cfg(feature = "export")]
pub use export::splat_to_ply;
#[cfg(feature = "import")]
pub use import::{
    ParseMetadata, SplatData, SplatMessage, load_splat_from_ply, stream_splat_from_ply,
};
pub use ply_gaussian::PlyGaussian;

// Re-export serde-ply types for compatibility
#[cfg(feature = "import")]
pub use serde_ply::DeserializeError;
#[cfg(feature = "export")]
pub use serde_ply::SerializeError;

#[cfg(test)]
#[allow(unused)]
mod test_utils {
    use brush_render::MainBackend;
    use brush_render::gaussian_splats::{SplatRenderMode, Splats};
    use brush_render::sh::sh_coeffs_for_degree;
    use burn::backend::wgpu::WgpuDevice;

    pub fn create_test_splats(sh_degree: u32) -> Splats<MainBackend> {
        create_test_splats_with_count(sh_degree, 1)
    }

    pub fn create_test_splats_with_count(sh_degree: u32, num_splats: usize) -> Splats<MainBackend> {
        let device = WgpuDevice::default();
        let coeffs_per_channel = sh_coeffs_for_degree(sh_degree) as usize;

        let mut means = Vec::new();
        let mut rotations = Vec::new();
        let mut log_scales = Vec::new();
        let mut sh_coeffs = Vec::new();
        let mut opacities = Vec::new();

        for i in 0..num_splats {
            let offset = i as f32;

            means.extend([offset, offset + 1.0, offset + 2.0]);
            rotations.extend([1.0, 0.0, 0.0, 0.0]);
            log_scales.extend([
                -0.1 + offset * 0.05,
                0.2 + offset * 0.05,
                -0.3 + offset * 0.05,
            ]);

            for _ in 0..3 {
                sh_coeffs.push(0.5 + offset * 0.1);
                for j in 1..coeffs_per_channel {
                    sh_coeffs.push(j as f32 * 0.1 + offset * 0.01);
                }
            }

            opacities.push(0.8 - offset * 0.1);
        }

        Splats::<MainBackend>::from_raw(
            means,
            rotations,
            log_scales,
            sh_coeffs,
            opacities,
            SplatRenderMode::Default,
            &device,
        )
        .with_sh_degree(sh_degree)
    }
}
