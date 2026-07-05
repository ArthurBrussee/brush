//! Rewrite `out_img` in each `test_cases/*.safetensors` from brush forward render.
//!
//! Run from repo root:
//! `cargo run -p brush-bench-test --bin regenerate_references`

use std::{collections::BTreeMap, fs, path::PathBuf};

use brush_bench_test::safetensor_utils::{safetensor_to_burn, splats_from_safetensors};
use brush_render::{
    TextureMode,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::{Splats, render_splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use safetensors::tensor::{Dtype, serialize_to_file};
use safetensors::{SafeTensors, tensor::TensorView};

const CASES: &[&str] = &["tiny_case", "basic_case", "mix_case"];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let cases_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_cases");

    for name in CASES {
        let path = cases_dir.join(format!("{name}.safetensors"));
        let data = fs::read(&path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let splats: Splats = splats_from_safetensors(&tensors, &device)?;
        let img_ref = safetensor_to_burn::<3>(&tensors.tensor("out_img")?, &device)?;
        let [h, w, _] = img_ref.dims();

        let fov = std::f64::consts::PI * 0.5;
        let focal = fov_to_focal(fov, w as u32, &Pinhole);
        let cam = Camera::new(
            glam::vec3(0.123, 0.456, -8.0),
            glam::Quat::IDENTITY,
            focal_to_fov(focal, w as u32, &Pinhole),
            focal_to_fov(focal, h as u32, &Pinhole),
            glam::vec2(0.5, 0.5),
            Pinhole,
        );

        let (rendered, _aux) = render_splats(
            splats,
            &cam,
            glam::uvec2(w as u32, h as u32),
            glam::Vec3::ZERO,
            None,
            TextureMode::Float,
        )
        .await;

        let out_img: Vec<f32> = rendered
            .into_data_async()
            .await?
            .into_vec()
            .expect("f32 out_img");
        let out_view = TensorView::new(Dtype::F32, vec![h, w, 4], bytemuck::cast_slice(&out_img))?;

        let mut out: BTreeMap<&str, TensorView<'_>> = BTreeMap::new();
        for key in ["means", "scales", "quats", "coeffs", "opacities"] {
            let t = tensors.tensor(key)?;
            out.insert(
                key,
                TensorView::new(t.dtype(), t.shape().to_vec(), t.data())?,
            );
        }
        out.insert("out_img", out_view);

        serialize_to_file(out, None, &path)?;
        println!("updated {path:?}");
    }

    Ok(())
}
