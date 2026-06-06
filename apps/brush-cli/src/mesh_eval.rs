//! Mesh evaluation: render the extracted mesh at training-camera
//! viewpoints with an in-process wgpu renderer
//! ([`crate::mesh_render::MeshRenderer`]) and report PSNR + coverage.
//!
//! Two comparison targets:
//! - **Photo GT** (always): the dataset's recorded image. Mixes
//!   *mesh-vs-splat* error with *splat-vs-photo* error.
//! - **Splat render** (when splats passed): re-render the splats at the
//!   same camera and compare to that. Isolates mesh quality from splat
//!   reconstruction quality, so any regression points squarely at the
//!   extractor.
//!
//! The renderer is unlit (rendered RGB = barycentric vertex-color
//! interpolation), so PSNR reflects mesh fidelity rather than any
//! external shading model.

use std::path::{Path, PathBuf};

use anyhow::Context;
use brush_dataset::scene::SceneView;
use brush_render::gaussian_splats::Splats;
use brush_render::{TextureMode, render_splats};
use burn::tensor::s;
use glam::{UVec2, Vec3};
use image::{ImageBuffer, Rgb};

pub struct ViewEval {
    pub view_idx: usize,
    /// PSNR of mesh-render vs the splat render at this view (when splats
    /// were provided). Falls back to photo PSNR otherwise.
    pub psnr: f64,
    /// PSNR of mesh-render vs the dataset photo (always computed).
    pub psnr_vs_photo: f64,
    pub coverage: f64,
    pub rendered_path: PathBuf,
}

pub async fn eval_psnr(
    mesh: &brush_mesh::Mesh,
    train_views: &[SceneView],
    splats: Option<&Splats>,
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

        let psnr_vs_photo = crate::mesh_render::psnr(&rendered, &gt);
        let coverage = compute_coverage(&rendered);

        // Splat-render comparison: re-render splats at this camera and use
        // as the per-pixel target. PSNR vs splat-render is the headline
        // metric — it isolates extractor error from splat reconstruction
        // error, which is the meaningful number when iterating on the
        // mesh pipeline alone.
        let (psnr, sbs) = if let Some(splats) = splats {
            let splat_img = render_splats_to_rgb(splats, &view.camera, img_size).await?;
            let splat_path = out_dir.join(format!("view_{i:04}_splat.png"));
            splat_img
                .save(&splat_path)
                .with_context(|| format!("saving splat render to {}", splat_path.display()))?;
            let psnr_vs_splat = crate::mesh_render::psnr(&rendered, &splat_img);
            let sbs = three_panel(&gt, &splat_img, &rendered);
            (psnr_vs_splat, sbs)
        } else {
            let sbs = side_by_side(&gt, &rendered);
            (psnr_vs_photo, sbs)
        };

        let sbs_path = out_dir.join(format!("view_{i:04}_sbs.png"));
        sbs.save(&sbs_path)
            .with_context(|| format!("saving SBS to {}", sbs_path.display()))?;

        log::info!(
            "view[{i:04}]: PSNR(mesh vs splat)={psnr:.2} PSNR(mesh vs photo)={psnr_vs_photo:.2} \
             coverage={:.1}%",
            coverage * 100.0
        );

        results.push(ViewEval {
            view_idx: i,
            psnr,
            psnr_vs_photo,
            coverage,
            rendered_path,
        });
    }

    if !results.is_empty() {
        let mean_psnr = results.iter().map(|r| r.psnr).sum::<f64>() / results.len() as f64;
        let mean_photo =
            results.iter().map(|r| r.psnr_vs_photo).sum::<f64>() / results.len() as f64;
        let mean_cov = results.iter().map(|r| r.coverage).sum::<f64>() / results.len() as f64;
        log::info!(
            "PSNR over {} views: mean_vs_splat={mean_psnr:.3} mean_vs_photo={mean_photo:.3} \
             coverage={:.1}%",
            results.len(),
            mean_cov * 100.0
        );
    }

    Ok(results)
}

/// Render the splats at `camera` to an 8-bit RGB image, applying the
/// same 8-bit roundtrip as the training eval pipeline (`*255`, round,
/// `/255`) so the mesh-render bytes and splat-render bytes are
/// quantised consistently before PSNR.
async fn render_splats_to_rgb(
    splats: &Splats,
    camera: &brush_render::camera::Camera,
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

/// 3-panel: photo | splat render | mesh render.
fn three_panel(
    a: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    b: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    c: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let w = a.width() + b.width() + c.width();
    let h = a.height().max(b.height()).max(c.height());
    let mut out = ImageBuffer::from_pixel(w, h, Rgb([0u8, 0, 0]));
    image::imageops::overlay(&mut out, a, 0, 0);
    image::imageops::overlay(&mut out, b, a.width() as i64, 0);
    image::imageops::overlay(&mut out, c, (a.width() + b.width()) as i64, 0);
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
