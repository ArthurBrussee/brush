//! GOF's `filter_mesh` step. The decision is per-*crossing* (= per mesh
//! vertex), not per-face: a crossing is kept iff the original Delaunay
//! edge it spans has length ≤ the sum of its two endpoint Gaussian
//! scales. The caller computes that flag from the original Delaunay
//! `(pts, scales)` (see [`crate::extract`]) — by the time we get the
//! `Mesh`, vertex positions have been refined by the binary search and
//! the Delaunay edge length is no longer recoverable.
//!
//! Once the per-crossing mask is set, any face touching a dropped
//! crossing is dropped, matching GOF's `face_mask = mask[faces].all(axis=1)`.

use crate::Mesh;

/// Apply a per-vertex keep mask: any face whose any vertex is `false`
/// in `keep` is dropped, and the surviving vertices are compacted with
/// face indices remapped.
pub fn filter_mesh_with_keep(mesh: &Mesh, keep: &[bool]) -> Mesh {
    assert_eq!(keep.len(), mesh.vertices.len(), "keep mask is per-vertex");

    let mut faces = Vec::with_capacity(mesh.faces.len());
    for f in &mesh.faces {
        if keep[f[0] as usize] && keep[f[1] as usize] && keep[f[2] as usize] {
            faces.push(*f);
        }
    }

    let has_colors = !mesh.vertex_colors.is_empty();
    let mut remap = vec![u32::MAX; mesh.vertices.len()];
    let mut vertices = Vec::new();
    let mut vertex_colors = Vec::new();
    for (i, &k) in keep.iter().enumerate() {
        if k {
            remap[i] = vertices.len() as u32;
            vertices.push(mesh.vertices[i]);
            if has_colors {
                vertex_colors.push(mesh.vertex_colors[i]);
            }
        }
    }
    for f in &mut faces {
        for v in f {
            *v = remap[*v as usize];
        }
    }
    Mesh {
        vertices,
        vertex_colors,
        faces,
    }
}

/// Drop connected components with fewer than `min_faces` faces. Isolated
/// iso-crossing blobs (the "speckled halo" around real geometry) are tens of
/// faces; legitimate disconnected objects are orders of magnitude larger.
/// Mirrors GOF/2DGS mesh post-processing (cluster removal).
pub fn filter_small_components(mesh: &Mesh, min_faces: usize) -> Mesh {
    if min_faces <= 1 || mesh.faces.is_empty() {
        return mesh.clone();
    }

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }

    let n = mesh.vertices.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for f in &mesh.faces {
        let a = find(&mut parent, f[0]);
        let b = find(&mut parent, f[1]);
        parent[b as usize] = a;
        let c = find(&mut parent, f[2]);
        parent[c as usize] = a;
    }

    let mut comp_faces = vec![0u32; n];
    for f in &mesh.faces {
        comp_faces[find(&mut parent, f[0]) as usize] += 1;
    }
    let keep: Vec<bool> = (0..n as u32)
        .map(|v| comp_faces[find(&mut parent, v) as usize] as usize >= min_faces)
        .collect();

    let dropped = keep.iter().filter(|&&k| !k).count();
    if dropped > 0 {
        log::info!("Component filter: dropping {dropped} verts in components < {min_faces} faces");
    }
    filter_mesh_with_keep(mesh, &keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_components_are_dropped() {
        // A two-face quad plus one isolated triangle far away.
        let mesh = Mesh {
            vertices: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
                glam::Vec3::new(1.0, 1.0, 0.0),
                glam::Vec3::new(100.0, 0.0, 0.0),
                glam::Vec3::new(101.0, 0.0, 0.0),
                glam::Vec3::new(100.0, 1.0, 0.0),
            ],
            vertex_colors: Vec::new(),
            faces: vec![[0, 1, 2], [1, 3, 2], [4, 5, 6]],
        };
        let out = filter_small_components(&mesh, 2);
        assert_eq!(out.faces.len(), 2);
        assert_eq!(out.vertices.len(), 4);
        // min_faces 1 keeps everything.
        let all = filter_small_components(&mesh, 1);
        assert_eq!(all.faces.len(), 3);
    }

    #[test]
    fn drops_marked_vertices() {
        let mesh = Mesh {
            vertices: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(0.1, 0.0, 0.0),
                glam::Vec3::new(0.0, 0.1, 0.0),
                glam::Vec3::new(100.0, 0.0, 0.0),
            ],
            vertex_colors: Vec::new(),
            faces: vec![[0, 1, 2], [0, 1, 3]],
        };
        let keep = vec![true, true, true, false];
        let out = filter_mesh_with_keep(&mesh, &keep);
        assert_eq!(out.faces.len(), 1);
        assert_eq!(out.vertices.len(), 3);
    }
}
