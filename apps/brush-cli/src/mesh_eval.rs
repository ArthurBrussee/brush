//! Mesh evaluation: render the extracted mesh at training-camera
//! viewpoints with an in-process wgpu renderer
//! ([`crate::mesh_render::MeshRenderer`]) and report PSNR + coverage vs
//! the GT image. The renderer is unlit (rendered RGB = barycentric
//! vertex-color interpolation), so PSNR reflects mesh fidelity rather
//! than any external shading model.

use std::path::{Path, PathBuf};

use anyhow::Context;
use brush_dataset::scene::SceneView;
use glam::UVec2;
use image::{ImageBuffer, Rgb};

pub struct ViewEval {
    pub view_idx: usize,
    pub psnr: f64,
    pub coverage: f64,
    pub rendered_path: PathBuf,
}

pub async fn eval_psnr(
    mesh: &brush_mesh::Mesh,
    train_views: &[SceneView],
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
            .context("read GT dims")?;
        let img_size = UVec2::new(w, h);
        let gt = view.image.load().await.context("load GT image")?.to_rgb8();

        let rendered = renderer.render(mesh, &view.camera, img_size);
        let rendered_path = out_dir.join(format!("view_{i:04}.png"));
        rendered
            .save(&rendered_path)
            .with_context(|| format!("saving rendered view to {}", rendered_path.display()))?;
        let gt_path = out_dir.join(format!("view_{i:04}_gt.png"));
        gt.save(&gt_path)
            .with_context(|| format!("saving GT to {}", gt_path.display()))?;
        let sbs_path = out_dir.join(format!("view_{i:04}_sbs.png"));
        side_by_side(&gt, &rendered)
            .save(&sbs_path)
            .with_context(|| format!("saving SBS to {}", sbs_path.display()))?;

        let psnr = crate::mesh_render::psnr(&rendered, &gt);
        let coverage = compute_coverage(&rendered);
        log::info!(
            "view[{i:04}]: PSNR={psnr:.2} coverage={:.1}%",
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
            "PSNR over {} views: mean={mean_psnr:.3} (coverage={:.1}%)",
            results.len(),
            mean_cov * 100.0
        );
    }

    Ok(results)
}

/// Horizontal concat: GT on the left, mesh render on the right.
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
