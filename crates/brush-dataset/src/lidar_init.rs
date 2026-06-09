//! Splat initialization from per-view `LiDAR` depth.
//!
//! Back-projects each view's metric `LiDAR` depth into a world point cloud
//! (confidence-filtered), samples per-point color from the RGB image, and
//! voxel-downsamples in **metric** units. Because `LiDAR` is metric the voxel
//! size is a physical length (scene-scale invariant), and voxelizing collapses
//! the heavy cross-view overlap into one representative point per cell, giving
//! even coverage at a controllable density.

use std::collections::HashMap;

use brush_render::sh::rgb_to_sh;
use brush_serde::SplatData;
use glam::{UVec2, Vec3};

use crate::scene::SceneView;

/// Auto voxel size = cloud extent / this many cells along the longest axis,
/// when `voxel_size <= 0`. Keeps init density scene-scale invariant.
const AUTO_GRID: f32 = 128.0;

/// Build init `SplatData` from the views' `LiDAR` depth. `voxel_size` is in
/// metres; `<= 0` auto-derives it from the cloud extent (scene-relative).
/// `min_conf` is the `ARKit` confidence floor (0/1/2). Returns `None` when no
/// view has usable depth.
pub async fn lidar_init_splats(
    views: &[SceneView],
    voxel_size: f32,
    min_conf: u8,
) -> anyhow::Result<Option<SplatData>> {
    // Pass 1: back-project every confident depth pixel into a world point +
    // sampled color, tracking the cloud extent for auto voxel sizing. Runs on a
    // small pool of actor threads — each owns a chunk of views, loads their
    // depth/image and back-projects independently — then the per-chunk clouds
    // are merged.
    async fn project_chunk(
        views: Vec<SceneView>,
        min_conf: u8,
    ) -> anyhow::Result<(Vec<(Vec3, Vec3)>, Vec3, Vec3)> {
        let mut pts: Vec<(Vec3, Vec3)> = Vec::new();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for view in &views {
            let Some(depth_loader) = &view.depth else {
                continue;
            };
            let depth = depth_loader.load().await?;
            let (w, h) = (depth.width, depth.height);
            if w == 0 || h == 0 {
                continue;
            }

            let img = view.image.load().await?.to_rgb8();
            let (iw, ih) = (img.width() as f32, img.height() as f32);

            // Intrinsics at the depth map's native resolution (LiDAR is aligned
            // to the RGB camera's fov). Pinhole back-projection ignores lens
            // distortion, which is fine for an init.
            let pin = view
                .camera
                .build_pinhole_params(UVec2::new(w as u32, h as u32));
            let l2w = view.camera.local_to_world();

            for v in 0..h {
                for u in 0..w {
                    let i = v * w + u;
                    if depth.conf[i] < min_conf {
                        continue;
                    }
                    let z = depth.depth[i];
                    if !(z.is_finite() && z > 0.0) {
                        continue;
                    }
                    // z-depth back-projection: p_cam = ((u-cx)/fx,(v-cy)/fy,1)*z.
                    let xc = (u as f32 + 0.5 - pin.cx) / pin.fx * z;
                    let yc = (v as f32 + 0.5 - pin.cy) / pin.fy * z;
                    let p = l2w.transform_point3(Vec3::new(xc, yc, z));
                    if !p.is_finite() {
                        continue;
                    }

                    // Color from the matching image pixel (depth and RGB share
                    // the camera fov, different resolutions).
                    let ix = (((u as f32 + 0.5) / w as f32) * iw).clamp(0.0, iw - 1.0) as u32;
                    let iy = (((v as f32 + 0.5) / h as f32) * ih).clamp(0.0, ih - 1.0) as u32;
                    let px = img.get_pixel(ix, iy);
                    let col = Vec3::new(px[0] as f32, px[1] as f32, px[2] as f32) / 255.0;

                    lo = lo.min(p);
                    hi = hi.max(p);
                    pts.push((p, col));
                }
            }
        }
        Ok((pts, lo, hi))
    }

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(views.len().max(1));
    let chunk_size = views.len().div_ceil(workers).max(1);
    // Keep the actors alive until their handles resolve (dropping an actor
    // tears down its worker thread).
    let mut actors = Vec::new();
    let mut handles = Vec::new();
    for chunk in views.chunks(chunk_size) {
        let chunk = chunk.to_vec();
        let actor = brush_async::Actor::new("lidar-init");
        handles.push(actor.run(move || project_chunk(chunk, min_conf)));
        actors.push(actor);
    }

    let mut pts: Vec<(Vec3, Vec3)> = Vec::new();
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for handle in handles {
        let (p, l, h) = handle.await?;
        pts.extend(p);
        lo = lo.min(l);
        hi = hi.max(h);
    }
    drop(actors);

    if pts.is_empty() {
        return Ok(None);
    }

    // Scene-relative voxel size unless a metric override is given.
    let voxel = if voxel_size > 0.0 {
        voxel_size
    } else {
        ((hi - lo).max_element() / AUTO_GRID).max(1e-4)
    };
    let inv = 1.0 / voxel;

    // Pass 2: voxel-downsample (collapses cross-view overlap).
    // Per voxel: (sum position, sum color, count).
    let mut acc: HashMap<(i64, i64, i64), (Vec3, Vec3, u32)> = HashMap::new();
    for (p, col) in pts {
        let key = (
            (p.x * inv).floor() as i64,
            (p.y * inv).floor() as i64,
            (p.z * inv).floor() as i64,
        );
        let e = acc.entry(key).or_insert((Vec3::ZERO, Vec3::ZERO, 0));
        e.0 += p;
        e.1 += col;
        e.2 += 1;
    }

    let mut means = Vec::with_capacity(acc.len() * 3);
    let mut sh = Vec::with_capacity(acc.len() * 3);
    for (psum, csum, cnt) in acc.into_values() {
        let inv_n = 1.0 / cnt as f32;
        let p = psum * inv_n;
        let dc = rgb_to_sh(csum * inv_n);
        means.extend_from_slice(&[p.x, p.y, p.z]);
        sh.extend_from_slice(&[dc.x, dc.y, dc.z]);
    }

    Ok(Some(SplatData {
        means,
        rotations: None,
        log_scales: None,
        sh_coeffs: Some(sh),
        raw_opacities: None,
    }))
}
