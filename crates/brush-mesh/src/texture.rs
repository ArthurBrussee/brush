//! UV-atlased color texture: decouples color resolution from vertex density.
//!
//! The charter is deliberately simple: faces cluster into charts by BFS over
//! shared edges among faces with similar normals, each chart is planar-
//! projected along its dominant axis, and chart rectangles are shelf-packed
//! at uniform world-space texel density. Real parameterizers (xatlas, uvgen)
//! were tried and ran minutes on extraction-sized meshes; we don't need
//! their distortion guarantees because texels are baked at their true 3D
//! surface position, so chart distortion only modulates sample density.
//! Chart borders are gutter-dilated so bilinear filtering and mipmaps don't
//! bleed background.

use glam::Vec3;
use rustc_hash::FxHashMap;

use crate::Mesh;

/// Baked color atlas plus per-vertex UVs (aligned with the atlased mesh's
/// vertices; charting splits vertices along chart seams).
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub uvs: Vec<[f32; 2]>,
}

fn face_normal(mesh: &Mesh, f: usize) -> Vec3 {
    let [a, b, c] = mesh.faces[f];
    let (a, b, c) = (
        mesh.vertices[a as usize],
        mesh.vertices[b as usize],
        mesh.vertices[c as usize],
    );
    (b - a).cross(c - a).normalize_or_zero()
}

/// Chart + pack the mesh's UVs into an `atlas_size` square atlas. Returns
/// the re-indexed mesh (vertices duplicated per chart, colors carried over)
/// and per-vertex UVs in `[0, 1]`.
pub fn atlas_mesh(mesh: &Mesh, atlas_size: u32) -> Option<(Mesh, Vec<[f32; 2]>)> {
    let n_faces = mesh.faces.len();
    if n_faces == 0 {
        return None;
    }

    // Face adjacency over shared edges.
    let mut edge_owner: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut adj: Vec<[i64; 3]> = vec![[-1; 3]; n_faces];
    for (fi, f) in mesh.faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            match edge_owner.get(&key) {
                Some(&other) => {
                    adj[fi][k] = other as i64;
                    let of = &mesh.faces[other as usize];
                    for ok in 0..3 {
                        let (oa, ob) = (of[ok], of[(ok + 1) % 3]);
                        if (oa.min(ob), oa.max(ob)) == key {
                            adj[other as usize][ok] = fi as i64;
                        }
                    }
                }
                None => {
                    edge_owner.insert(key, fi as u32);
                }
            }
        }
    }

    // BFS face clustering: grow charts while normals stay within ~45 degrees
    // of the chart's seed normal, capped so packing rectangles stay sane.
    const NORMAL_COS: f32 = 0.7;
    const MAX_CHART_FACES: usize = 4096;
    let normals: Vec<Vec3> = (0..n_faces).map(|f| face_normal(mesh, f)).collect();
    let mut chart_of = vec![u32::MAX; n_faces];
    let mut charts: Vec<Vec<u32>> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for seed in 0..n_faces {
        if chart_of[seed] != u32::MAX {
            continue;
        }
        let chart_id = charts.len() as u32;
        let seed_n = normals[seed];
        let mut faces = vec![seed as u32];
        chart_of[seed] = chart_id;
        queue.clear();
        queue.push_back(seed);
        while let Some(f) = queue.pop_front() {
            if faces.len() >= MAX_CHART_FACES {
                break;
            }
            for &nb in &adj[f] {
                if nb < 0 {
                    continue;
                }
                let nb = nb as usize;
                if chart_of[nb] == u32::MAX && normals[nb].dot(seed_n) > NORMAL_COS {
                    chart_of[nb] = chart_id;
                    faces.push(nb as u32);
                    queue.push_back(nb);
                }
            }
        }
        charts.push(faces);
    }

    // Planar-project each chart along its dominant normal axis; collect the
    // chart's 2D bbox in world units.
    struct ChartUv {
        faces: Vec<u32>,
        // Per-face per-corner 2D coords, world units, bbox-relative.
        coords: Vec<[[f32; 2]; 3]>,
        size: [f32; 2],
    }
    let projected: Vec<ChartUv> = charts
        .into_iter()
        .map(|faces| {
            let mut mean_n = Vec3::ZERO;
            for &f in &faces {
                mean_n += normals[f as usize];
            }
            let an = mean_n.abs();
            let axis = if an.x >= an.y && an.x >= an.z {
                0
            } else if an.y >= an.z {
                1
            } else {
                2
            };
            let (u_ax, v_ax) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let mut coords = Vec::with_capacity(faces.len());
            let mut lo = [f32::INFINITY; 2];
            let mut hi = [f32::NEG_INFINITY; 2];
            for &f in &faces {
                let mut tri = [[0.0f32; 2]; 3];
                for (k, &vi) in mesh.faces[f as usize].iter().enumerate() {
                    let p = mesh.vertices[vi as usize];
                    let uv = [p[u_ax], p[v_ax]];
                    lo = [lo[0].min(uv[0]), lo[1].min(uv[1])];
                    hi = [hi[0].max(uv[0]), hi[1].max(uv[1])];
                    tri[k] = uv;
                }
                coords.push(tri);
            }
            for tri in &mut coords {
                for uv in tri {
                    uv[0] -= lo[0];
                    uv[1] -= lo[1];
                }
            }
            ChartUv {
                faces,
                coords,
                size: [hi[0] - lo[0], hi[1] - lo[1]],
            }
        })
        .collect();

    // Uniform texel density: total chart area maps to ~70% of the atlas.
    let total_area: f64 = projected
        .iter()
        .map(|c| (c.size[0] as f64) * (c.size[1] as f64))
        .sum();
    if total_area <= 0.0 {
        return None;
    }
    let px = atlas_size as f64;
    let texels_per_unit = (px * px * 0.7 / total_area).sqrt() as f32;
    const GUTTER: f32 = 2.0;

    // Shelf packing, tallest charts first.
    let mut order: Vec<usize> = (0..projected.len()).collect();
    order.sort_by(|&a, &b| {
        (projected[b].size[1])
            .partial_cmp(&projected[a].size[1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut origin: Vec<[f32; 2]> = vec![[0.0; 2]; projected.len()];
    let mut scale = texels_per_unit;
    'pack: loop {
        let mut cur_x = GUTTER;
        let mut cur_y = GUTTER;
        let mut shelf_h = 0.0f32;
        for &ci in &order {
            let w = projected[ci].size[0] * scale;
            let h = projected[ci].size[1] * scale;
            if cur_x + w + GUTTER > atlas_size as f32 {
                cur_y += shelf_h + GUTTER;
                cur_x = GUTTER;
                shelf_h = 0.0;
            }
            if cur_y + h + GUTTER > atlas_size as f32 || w + 2.0 * GUTTER > atlas_size as f32 {
                // Didn't fit: shrink density and repack.
                scale *= 0.95;
                continue 'pack;
            }
            origin[ci] = [cur_x, cur_y];
            cur_x += w + GUTTER;
            shelf_h = shelf_h.max(h);
        }
        break;
    }
    log::info!(
        "Atlas: {} charts, {:.0} texels/unit (started {:.0})",
        projected.len(),
        scale,
        texels_per_unit
    );

    // Emit the re-indexed mesh: vertices are duplicated per chart (a vertex
    // shared by two charts gets two UVs).
    let inv = 1.0 / atlas_size as f32;
    let has_colors = !mesh.vertex_colors.is_empty();
    let mut vertices = Vec::new();
    let mut vertex_colors = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut faces_out = Vec::with_capacity(n_faces);
    let mut vert_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    for (ci, chart) in projected.iter().enumerate() {
        for (fi, &f) in chart.faces.iter().enumerate() {
            let mut new_face = [0u32; 3];
            for k in 0..3 {
                let vi = mesh.faces[f as usize][k];
                let new_vi = *vert_map.entry((ci as u32, vi)).or_insert_with(|| {
                    vertices.push(mesh.vertices[vi as usize]);
                    if has_colors {
                        vertex_colors.push(mesh.vertex_colors[vi as usize]);
                    }
                    let c = chart.coords[fi][k];
                    uvs.push([
                        (origin[ci][0] + c[0] * scale) * inv,
                        (origin[ci][1] + c[1] * scale) * inv,
                    ]);
                    (vertices.len() - 1) as u32
                });
                new_face[k] = new_vi;
            }
            faces_out.push(new_face);
        }
    }
    Some((
        Mesh {
            vertices,
            vertex_colors,
            faces: faces_out,
        },
        uvs,
    ))
}

/// Rasterize `face`'s UV triangle at `atlas_px`: appends the world-space
/// surface position and atlas pixel index of every texel whose center lies
/// inside the triangle (clamped barycentrics give a half-texel skirt).
pub fn rasterize_face_texels(
    mesh: &Mesh,
    uvs: &[[f32; 2]],
    atlas_px: u32,
    face: usize,
    out_pos: &mut Vec<Vec3>,
    out_idx: &mut Vec<u32>,
) {
    let f = mesh.faces[face];
    let (ia, ib, ic) = (f[0] as usize, f[1] as usize, f[2] as usize);
    let res = atlas_px as f32;
    let (ua, ub, uc) = (uvs[ia], uvs[ib], uvs[ic]);
    let a = [ua[0] * res, ua[1] * res];
    let b = [ub[0] * res, ub[1] * res];
    let c = [uc[0] * res, uc[1] * res];

    let min_x = a[0].min(b[0]).min(c[0]).floor().max(0.0) as u32;
    let max_x = (a[0].max(b[0]).max(c[0]).ceil() as u32).min(atlas_px - 1);
    let min_y = a[1].min(b[1]).min(c[1]).floor().max(0.0) as u32;
    let max_y = (a[1].max(b[1]).max(c[1]).ceil() as u32).min(atlas_px - 1);

    let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    if det.abs() < 1e-12 {
        return;
    }
    let inv_det = 1.0 / det;
    let (pa, pb, pc) = (mesh.vertices[ia], mesh.vertices[ib], mesh.vertices[ic]);

    for ty in min_y..=max_y {
        for tx in min_x..=max_x {
            let px = tx as f32 + 0.5;
            let py = ty as f32 + 0.5;
            let mut wb = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) * inv_det;
            let mut wc = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) * inv_det;
            // Half-texel tolerance keeps edge texels owned by someone.
            let tol = 0.5 / res.max(1.0);
            if wb < -tol || wc < -tol || wb + wc > 1.0 + tol {
                continue;
            }
            wb = wb.clamp(0.0, 1.0);
            wc = wc.clamp(0.0, 1.0);
            let sum = wb + wc;
            if sum > 1.0 {
                wb /= sum;
                wc /= sum;
            }
            let wa = 1.0 - wb - wc;
            out_pos.push(pa * wa + pb * wb + pc * wc);
            out_idx.push(ty * atlas_px + tx);
        }
    }
}

/// Fill empty atlas texels from their nearest baked neighbours (`iters`
/// rings) so bilinear filtering and mip levels don't pull in background.
pub fn dilate(rgba: &mut [u8], w: u32, h: u32, iters: u32) {
    let (w, h) = (w as i64, h as i64);
    for _ in 0..iters {
        let snapshot = rgba.to_vec();
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                if snapshot[o + 3] != 0 {
                    continue;
                }
                'n: for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let no = ((ny * w + nx) * 4) as usize;
                    if snapshot[no + 3] != 0 {
                        rgba[o..o + 4].copy_from_slice(&snapshot[no..no + 4]);
                        break 'n;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_and_rasterize_unit_quad() {
        let mesh = Mesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            vertex_colors: vec![[255, 0, 0]; 4],
            faces: vec![[0, 1, 2], [1, 3, 2]],
        };
        let (atlased, uvs) = atlas_mesh(&mesh, 128).expect("atlas");
        assert_eq!(atlased.vertices.len(), uvs.len(), "uv per vertex");
        assert_eq!(
            atlased.vertices.len(),
            atlased.vertex_colors.len(),
            "colors carried to split vertices"
        );
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        for f in 0..atlased.faces.len() {
            rasterize_face_texels(&atlased, &uvs, 64, f, &mut pos, &mut idx);
        }
        assert!(pos.len() > 500, "quad covers a decent part of a 64px atlas");
        assert!(idx.iter().all(|&i| i < 64 * 64), "indices in atlas");
        for p in &pos {
            assert!(p.z.abs() < 1e-5, "samples on the quad plane");
        }
    }

    #[test]
    fn dilation_fills_borders() {
        let mut rgba = vec![0u8; 4 * 4 * 4];
        rgba[(5 * 4)..(5 * 4 + 4)].copy_from_slice(&[10, 20, 30, 255]);
        dilate(&mut rgba, 4, 4, 1);
        // The 4-neighbours of texel (1,1) got filled.
        assert_eq!(&rgba[4..8], &[10, 20, 30, 255]);
    }
}
