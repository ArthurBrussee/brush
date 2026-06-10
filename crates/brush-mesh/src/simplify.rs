//! Quadric mesh simplification via meshoptimizer: the geometry-rate dial.
//! Collapses the marching-tets output to a target face count before UV
//! charting and texture baking; the color atlas then carries the detail.

use meshopt::{SimplifyOptions, VertexDataAdapter};

use crate::Mesh;

/// Simplify to roughly `target_faces` (quadric edge collapse, then unused
/// vertices compacted). Vertex colors follow the surviving vertices.
pub fn simplify_mesh(mesh: &Mesh, target_faces: usize) -> Mesh {
    if mesh.faces.len() <= target_faces || mesh.faces.is_empty() {
        return mesh.clone();
    }
    let indices: Vec<u32> = mesh.faces.iter().flatten().copied().collect();
    let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    let bytes: &[u8] = bytemuck_cast(&positions);
    let adapter =
        VertexDataAdapter::new(bytes, 12, 0).expect("vertex adapter over tightly packed f32x3");
    let mut err = 0.0f32;
    let new_indices = meshopt::simplify(
        &indices,
        &adapter,
        target_faces * 3,
        // Generous error bound: the target count is the real constraint.
        0.05,
        SimplifyOptions::None,
        Some(&mut err),
    );
    log::info!(
        "Simplified {} -> {} faces (relative error {err:.4})",
        mesh.faces.len(),
        new_indices.len() / 3
    );

    // Compact: keep only referenced vertices.
    let mut remap = vec![u32::MAX; mesh.vertices.len()];
    let mut vertices = Vec::new();
    let mut vertex_colors = Vec::new();
    let has_colors = !mesh.vertex_colors.is_empty();
    let mut faces = Vec::with_capacity(new_indices.len() / 3);
    for tri in new_indices.chunks_exact(3) {
        let mut f = [0u32; 3];
        for (slot, &old) in f.iter_mut().zip(tri) {
            let old_u = old as usize;
            if remap[old_u] == u32::MAX {
                remap[old_u] = vertices.len() as u32;
                vertices.push(mesh.vertices[old_u]);
                if has_colors {
                    vertex_colors.push(mesh.vertex_colors[old_u]);
                }
            }
            *slot = remap[old_u];
        }
        faces.push(f);
    }
    Mesh {
        vertices,
        vertex_colors,
        faces,
    }
}

fn bytemuck_cast(positions: &[[f32; 3]]) -> &[u8] {
    // SAFETY: [f32; 3] is tightly packed with no padding, so reinterpreting
    // the slice as bytes is layout-safe; alignment of u8 is 1.
    unsafe {
        std::slice::from_raw_parts(
            positions.as_ptr().cast::<u8>(),
            std::mem::size_of_val(positions),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn simplifies_dense_grid() {
        // A dense planar grid collapses heavily at tiny error.
        let n = 50usize;
        let mut vertices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                vertices.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let mut faces = Vec::new();
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let i = (y * n + x) as u32;
                let nn = n as u32;
                faces.push([i, i + 1, i + nn]);
                faces.push([i + 1, i + nn + 1, i + nn]);
            }
        }
        let mesh = Mesh {
            vertices,
            vertex_colors: Vec::new(),
            faces,
        };
        let out = simplify_mesh(&mesh, 200);
        assert!(
            out.faces.len() <= 250,
            "reached target-ish: {}",
            out.faces.len()
        );
        assert!(out.faces.len() > 10, "did not collapse to nothing");
        assert!(
            out.faces
                .iter()
                .all(|f| f.iter().all(|&v| (v as usize) < out.vertices.len())),
            "faces reference compacted vertices"
        );
    }
}
