//! Mesh evaluation: render the extracted mesh at training-camera
//! viewpoints with an in-process wgpu renderer
//! ([`crate::mesh_render::MeshRenderer`]) and report PSNR vs the
//! re-rendered splat appearance at each view. Only the mesh-vs-splat
//! PSNR is reported — mesh-vs-photo mixes extractor error with splat-
//! vs-photo error and isn't useful for iterating on the extractor.
//!
//! Both the splat-render *and* the mesh-render use a pinhole camera so
//! the SBS panels are pixel-aligned. The mesh wgpu rasterizer is
//! intrinsically pinhole (no distortion model in the shader); forcing
//! the splat render to also be pinhole keeps the comparison honest and
//! the SBS frames aligned (otherwise distorted-camera scenes show a
//! visible "jump" between the two halves).
//!
//! The renderer is unlit (rendered RGB = barycentric vertex-color
//! interpolation), so PSNR reflects mesh fidelity rather than any
//! external shading model.

use std::path::{Path, PathBuf};

use anyhow::Context;
use brush_dataset::scene::SceneView;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{TextureMode, render_splats};
use burn::tensor::s;
use glam::{UVec2, Vec3};
use image::{ImageBuffer, Rgb};

pub struct ViewEval {
    pub view_idx: usize,
    /// PSNR of mesh-render vs the splat render (at a pinhole camera).
    pub psnr: f64,
    pub coverage: f64,
    pub rendered_path: PathBuf,
}

pub async fn eval_psnr(
    mesh: &brush_mesh::Mesh,
    train_views: &[SceneView],
    splats: &Splats,
    n: usize,
    out_dir: &Path,
) -> anyhow::Result<Vec<ViewEval>> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;
    let n = n.min(train_views.len());
    let renderer =
        crate::mesh_render::MeshRenderer::new().context("initializing wgpu mesh renderer")?;
    let mut results = Vec::with_capacity(n);

    for (i, view) in train_views.iter().take(n).enumerate() {
        let (w, h) = view
            .image
            .output_dimensions()
            .await
            .context("read view dims")?;
        let img_size = UVec2::new(w, h);

        // Force a pinhole camera for both renders so the SBS panels line
        // up pixel-exact regardless of the dataset's distortion model.
        let pin_cam = with_pinhole(&view.camera);

        // 2× supersample the mesh-render then bilinear-downsample to GT
        // resolution. Effective 4× MSAA — removes the per-face barycentric
        // edge stairstepping that's visible on bike spokes / bonsai
        // silhouettes. The splat-render is already continuous-density and
        // doesn't alias, so this closes some of the SBS pixel-mismatch.
        const SS: u32 = 2;
        let big = renderer.render(mesh, &pin_cam, img_size * SS);
        let rendered = image::imageops::resize(
            &big,
            img_size.x,
            img_size.y,
            image::imageops::FilterType::Triangle,
        );
        let rendered_path = out_dir.join(format!("view_{i:04}.png"));
        rendered
            .save(&rendered_path)
            .with_context(|| format!("saving mesh render to {}", rendered_path.display()))?;

        let splat_img = render_splats_to_rgb(splats, &pin_cam, img_size).await?;
        let splat_path = out_dir.join(format!("view_{i:04}_splat.png"));
        splat_img
            .save(&splat_path)
            .with_context(|| format!("saving splat render to {}", splat_path.display()))?;

        let psnr = crate::mesh_render::psnr(&rendered, &splat_img);
        let coverage = compute_coverage(&rendered);

        let sbs = side_by_side(&splat_img, &rendered);
        let sbs_path = out_dir.join(format!("view_{i:04}_sbs.png"));
        sbs.save(&sbs_path)
            .with_context(|| format!("saving SBS to {}", sbs_path.display()))?;

        log::info!(
            "view[{i:04}]: PSNR(mesh vs splat)={psnr:.2} coverage={:.1}%",
            coverage * 100.0
        );

        results.push(ViewEval {
            view_idx: i,
            psnr,
            coverage,
            rendered_path,
        });
    }

    if !results.is_empty() {
        let mean_psnr = results.iter().map(|r| r.psnr).sum::<f64>() / results.len() as f64;
        let mean_cov = results.iter().map(|r| r.coverage).sum::<f64>() / results.len() as f64;
        log::info!(
            "PSNR over {} views: mean_vs_splat={mean_psnr:.3} coverage={:.1}%",
            results.len(),
            mean_cov * 100.0
        );
    }

    Ok(results)
}

/// Build a copy of `cam` with the camera model swapped to `Pinhole` and
/// the existing intrinsics (focal + center) re-derived through it. For
/// pinhole-source cameras this is a no-op; for distorted cameras it
/// removes the distortion so the splat-render and our wgpu mesh-render
/// share the same projection.
fn with_pinhole(cam: &Camera) -> Camera {
    Camera {
        camera_model: CameraModel::Pinhole,
        ..*cam
    }
}

/// Render the splats at `camera` to an 8-bit RGB image.
async fn render_splats_to_rgb(
    splats: &Splats,
    camera: &Camera,
    img_size: UVec2,
) -> anyhow::Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let (img, _aux) = render_splats(
        splats.clone(),
        camera,
        img_size,
        Vec3::ZERO,
        None,
        TextureMode::Float,
        false,
    )
    .await;
    let rgb = img.slice(s![.., .., 0..3]);
    let [h, w, _] = [rgb.dims()[0], rgb.dims()[1], rgb.dims()[2]];
    let data = rgb
        .into_data_async()
        .await
        .map_err(|e| anyhow::anyhow!("splat render readback: {e:?}"))?
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("splat render f32 unpack: {e:?}"))?;
    let mut out = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            let r = (data[base].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (data[base + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (data[base + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            out.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    Ok(out)
}

/// Horizontal concat: splat render on the left, mesh render on the right.
fn side_by_side(
    left: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    right: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let w = left.width() + right.width();
    let h = left.height().max(right.height());
    let mut out = ImageBuffer::from_pixel(w, h, Rgb([0u8, 0, 0]));
    image::imageops::overlay(&mut out, left, 0, 0);
    image::imageops::overlay(&mut out, right, left.width() as i64, 0);
    out
}

/// Coverage = fraction of pixels with non-background (non-(0,0,0))
/// rendered colour. The wgpu renderer clears the colour target to
/// transparent black, so this measures actual triangle coverage.
fn compute_coverage(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> f64 {
    let mut hit = 0u64;
    let total = img.width() as u64 * img.height() as u64;
    for p in img.pixels() {
        if p[0] != 0 || p[1] != 0 || p[2] != 0 {
            hit += 1;
        }
    }
    hit as f64 / total as f64
}
