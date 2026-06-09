//! Mesh evaluation: render the extracted mesh at training-camera
//! viewpoints with an in-process wgpu renderer
//! ([`crate::mesh_render::MeshRenderer`]) and report PSNR vs the
//! re-rendered splat appearance at each view, plus a labeled grid with
//! one row per view and columns `GT | splat | mesh | depth`.
//!
//! All renders use a pinhole camera so the SBS panels are pixel-
//! aligned. The mesh wgpu rasterizer is intrinsically pinhole (no
//! distortion model in the shader); forcing the splat render to also be
//! pinhole keeps the comparison honest and the SBS frames aligned
//! (otherwise distorted-camera scenes show a visible "jump" between the
//! panels).
//!
//! The mesh colour renderer is unlit (rendered RGB = barycentric
//! vertex-color interpolation), so PSNR reflects mesh fidelity rather
//! than any external shading model. The depth panel is per-view-z-
//! normalised grayscale.

use std::path::{Path, PathBuf};

use anyhow::Context;
use brush_dataset::scene::SceneView;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{TextureMode, render_splats};
use burn::tensor::s;
use glam::{UVec2, Vec3};
use image::{ImageBuffer, Rgb, RgbImage};

pub struct ViewEval {
    pub view_idx: usize,
    /// PSNR of mesh-render vs the splat render (at a pinhole camera).
    pub psnr: f64,
    /// PSNR of mesh-render vs the GT image (at a pinhole camera).
    pub psnr_vs_gt: f64,
    pub coverage: f64,
    pub rendered_path: PathBuf,
}

pub async fn eval_psnr(
    mesh: &brush_mesh::Mesh,
    train_views: &[SceneView],
    splats: &Splats,
    n: usize,
    zoomed_out: bool,
    out_dir: &Path,
) -> anyhow::Result<Vec<ViewEval>> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;
    let n = n.min(train_views.len());
    let renderer =
        crate::mesh_render::MeshRenderer::new().context("initializing wgpu mesh renderer")?;
    let mut results = Vec::with_capacity(n);
    // One row per view: GT | splat | mesh | depth, assembled into a
    // single labeled grid after the loop.
    let mut rows: Vec<[RgbImage; 4]> = Vec::with_capacity(n);

    let selection = select_views(mesh, train_views, n, zoomed_out);
    for &i in &selection {
        let view = &train_views[i];
        let (w, h) = view
            .image
            .output_dimensions()
            .await
            .context("read view dims")?;
        let img_size = UVec2::new(w, h);

        let pin_cam = with_pinhole(&view.camera);

        // GT: load the dataset image, RGB at native resolution.
        let gt_img = load_gt_rgb(view).await?;

        // Colour render: 2× supersample then bilinear-downsample to GT
        // resolution. Effective 4× MSAA — removes per-face barycentric
        // edge stairstepping. Depth is rendered at native resolution and
        // *not* downsampled: downsampling blends valid foreground pixels
        // with the (0,0,0) background, fading the panel into mush.
        const SS: u32 = 2;
        let (big_color, _) = renderer.render_with_depth(mesh, &pin_cam, img_size * SS);
        let rendered = image::imageops::resize(
            &big_color,
            img_size.x,
            img_size.y,
            image::imageops::FilterType::Triangle,
        );
        let (_, depth_raw) = renderer.render_with_depth(mesh, &pin_cam, img_size);
        let depth_img = crate::mesh_render::depth_to_color(&depth_raw, img_size.x, img_size.y);

        let rendered_path = out_dir.join(format!("view_{i:04}.png"));
        rendered
            .save(&rendered_path)
            .with_context(|| format!("saving mesh render to {}", rendered_path.display()))?;

        let splat_img = render_splats_to_rgb(splats, &pin_cam, img_size).await?;
        let splat_path = out_dir.join(format!("view_{i:04}_splat.png"));
        splat_img
            .save(&splat_path)
            .with_context(|| format!("saving splat render to {}", splat_path.display()))?;

        let psnr_vs_splat = crate::mesh_render::psnr(&rendered, &splat_img);
        let psnr_vs_gt = crate::mesh_render::psnr(&rendered, &gt_img);
        let coverage = compute_coverage(&rendered);

        log::info!(
            "view[{i:04}]: PSNR(mesh vs splat)={psnr_vs_splat:.2} PSNR(mesh vs GT)={psnr_vs_gt:.2} \
             coverage={:.1}%",
            coverage * 100.0
        );

        rows.push([gt_img, splat_img, rendered, depth_img]);
        results.push(ViewEval {
            view_idx: i,
            psnr: psnr_vs_splat,
            psnr_vs_gt,
            coverage,
            rendered_path,
        });
    }

    if !rows.is_empty() {
        let grid = assemble_grid(&rows, ["GT", "SPLAT", "MESH", "DEPTH"]);
        let grid_path = out_dir.join("eval_grid.png");
        grid.save(&grid_path)
            .with_context(|| format!("saving grid to {}", grid_path.display()))?;
        log::info!("wrote {} ({} view rows)", grid_path.display(), rows.len());
    }

    if !results.is_empty() {
        let mean_psnr = results.iter().map(|r| r.psnr).sum::<f64>() / results.len() as f64;
        let mean_gt = results.iter().map(|r| r.psnr_vs_gt).sum::<f64>() / results.len() as f64;
        let mean_cov = results.iter().map(|r| r.coverage).sum::<f64>() / results.len() as f64;
        log::info!(
            "PSNR over {} views: mean_vs_splat={mean_psnr:.3} mean_vs_GT={mean_gt:.3} \
             coverage={:.1}%",
            results.len(),
            mean_cov * 100.0
        );
    }

    Ok(results)
}

/// Build a copy of `cam` with the camera model swapped to `Pinhole`.
/// For pinhole-source cameras this is a no-op; for distorted cameras it
/// removes the distortion so the splat-render and our wgpu mesh-render
/// share the same projection.
fn with_pinhole(cam: &Camera) -> Camera {
    Camera {
        camera_model: CameraModel::Pinhole,
        ..*cam
    }
}

/// Choose `n` view indices for the eval grid. Default: even spread across
/// the capture. `zoomed_out`: rank cameras by distance to the mesh
/// centroid, keep the farthest half, then spread those by frame order so
/// the picks are both pulled-back and varied.
fn select_views(
    mesh: &brush_mesh::Mesh,
    views: &[SceneView],
    n: usize,
    zoomed_out: bool,
) -> Vec<usize> {
    let len = views.len();
    let n = n.min(len);
    if n == 0 {
        return Vec::new();
    }
    if !zoomed_out {
        return (0..n).map(|k| k * len / n).collect();
    }
    let centroid = mesh.vertices.iter().fold(Vec3::ZERO, |acc, &v| acc + v)
        / mesh.vertices.len().max(1) as f32;
    let mut by_dist: Vec<usize> = (0..len).collect();
    by_dist.sort_by(|&a, &b| {
        let da = (views[a].camera.position - centroid).length();
        let db = (views[b].camera.position - centroid).length();
        db.partial_cmp(&da).expect("finite camera distances")
    });
    let mut pool: Vec<usize> = by_dist.into_iter().take((len / 2).max(n)).collect();
    pool.sort_unstable();
    (0..n).map(|k| pool[k * pool.len() / n]).collect()
}

async fn load_gt_rgb(view: &SceneView) -> anyhow::Result<RgbImage> {
    let dyn_img = view.image.load().await.context("load GT image bytes")?;
    Ok(dyn_img.into_rgb8())
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

/// Assemble the per-view rows into one grid: a labeled header band on
/// top, then one row per view with four columns `GT | splat | mesh |
/// depth`. Panels are placed top-left in a uniform cell sized to the
/// largest panel; smaller panels are black-padded.
fn assemble_grid(rows: &[[RgbImage; 4]], labels: [&str; 4]) -> RgbImage {
    let cell_w = rows.iter().flatten().map(|p| p.width()).max().unwrap_or(1);
    let cell_h = rows.iter().flatten().map(|p| p.height()).max().unwrap_or(1);

    let scale = (cell_w / 120).max(2);
    let header_h = GLYPH_H * scale + 4 * scale;
    let total_w = cell_w * 4;
    let total_h = header_h + cell_h * rows.len() as u32;
    let mut out = ImageBuffer::from_pixel(total_w, total_h, Rgb([0u8, 0, 0]));

    for (c, label) in labels.iter().enumerate() {
        draw_label_centered(&mut out, label, c as u32 * cell_w, cell_w, header_h, scale);
    }
    for (r, row) in rows.iter().enumerate() {
        let y = header_h + r as u32 * cell_h;
        for (c, panel) in row.iter().enumerate() {
            image::imageops::overlay(&mut out, panel, (c as u32 * cell_w) as i64, y as i64);
        }
    }
    out
}

const GLYPH_W: u32 = 5;
const GLYPH_H: u32 = 7;

/// Draw `text` centered horizontally within `[x0, x0 + band_w)` and
/// vertically within `[0, band_h)`, in white, scaled by `scale`.
fn draw_label_centered(
    out: &mut RgbImage,
    text: &str,
    x0: u32,
    band_w: u32,
    band_h: u32,
    scale: u32,
) {
    let advance = (GLYPH_W + 1) * scale;
    let text_w = text.chars().count() as u32 * advance - scale.min(advance);
    let start_x = x0 + band_w.saturating_sub(text_w) / 2;
    let start_y = band_h.saturating_sub(GLYPH_H * scale) / 2;
    for (i, ch) in text.chars().enumerate() {
        draw_glyph(out, ch, start_x + i as u32 * advance, start_y, scale);
    }
}

fn draw_glyph(out: &mut RgbImage, ch: char, x: u32, y: u32, scale: u32) {
    let Some(rows) = glyph(ch) else { return };
    for (gy, bits) in rows.iter().enumerate() {
        for gx in 0..GLYPH_W {
            if bits & (1 << (GLYPH_W - 1 - gx)) != 0 {
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = x + gx * scale + sx;
                        let py = y + gy as u32 * scale + sy;
                        if px < out.width() && py < out.height() {
                            out.put_pixel(px, py, Rgb([255, 255, 255]));
                        }
                    }
                }
            }
        }
    }
}

/// 5×7 uppercase bitmap glyphs for the column labels (low 5 bits per
/// row, MSB = leftmost pixel). Unknown chars render blank.
fn glyph(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        _ => return None,
    })
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
