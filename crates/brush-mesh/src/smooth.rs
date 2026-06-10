//! Taubin λ|μ mesh smoothing.
//!
//! Marching-tets surfaces inherit high-frequency noise from the alpha field
//! (fuzzy splat shells) and the irregular Delaunay triangle shapes. Plain
//! Laplacian smoothing removes it but shrinks the mesh; Taubin alternates a
//! positive (λ) and a slightly larger negative (μ) Laplacian step, acting as
//! a low-pass filter that preserves volume (Taubin '95, λ = 0.5, μ = −0.53).
//!
//! Neighbour weights are uniform with face multiplicity (an interior edge is
//! shared by two faces and counts twice): no dedup pass needed, and the
//! weighting slightly favours well-connected neighbours.

use crate::Mesh;
use glam::Vec3;
use rayon::prelude::*;

const LAMBDA: f32 = 0.5;
const MU: f32 = -0.53;

/// In-place Taubin smoothing of `mesh.vertices`; `iters` is the number of
/// λ+μ pairs. Vertex colours/scales and connectivity are untouched.
pub fn taubin_smooth(mesh: &mut Mesh, iters: u32) {
    let nv = mesh.vertices.len();
    if iters == 0 || nv == 0 || mesh.faces.is_empty() {
        return;
    }

    // CSR adjacency over directed face edges (with multiplicity).
    let mut deg = vec![0u32; nv];
    for f in &mesh.faces {
        for &v in f {
            deg[v as usize] += 2;
        }
    }
    let mut offsets = vec![0u32; nv + 1];
    for i in 0..nv {
        offsets[i + 1] = offsets[i] + deg[i];
    }
    let mut adj = vec![0u32; offsets[nv] as usize];
    let mut cursor = offsets[..nv].to_vec();
    for f in &mesh.faces {
        for k in 0..3 {
            let a = f[k] as usize;
            let b = f[(k + 1) % 3];
            let c = f[(k + 2) % 3];
            adj[cursor[a] as usize] = b;
            adj[cursor[a] as usize + 1] = c;
            cursor[a] += 2;
        }
    }

    let mut pos = std::mem::take(&mut mesh.vertices);
    let mut next = vec![Vec3::ZERO; nv];
    for _ in 0..iters {
        for factor in [LAMBDA, MU] {
            next.par_iter_mut().enumerate().for_each(|(i, out)| {
                let (lo, hi) = (offsets[i] as usize, offsets[i + 1] as usize);
                let p = pos[i];
                *out = if lo == hi {
                    p
                } else {
                    let mut sum = Vec3::ZERO;
                    for &j in &adj[lo..hi] {
                        sum += pos[j as usize];
                    }
                    let mean = sum / (hi - lo) as f32;
                    p + (mean - p) * factor
                };
            });
            std::mem::swap(&mut pos, &mut next);
        }
    }
    mesh.vertices = pos;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An octahedron with one vertex pulled far out: smoothing must move the
    /// spike towards its neighbours without collapsing the shape.
    #[test]
    fn smooths_spike_without_collapse() {
        let mut mesh = Mesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 5.0), // spike (regular would be z = 1)
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 0.0, -1.0),
            ],
            vertex_colors: Vec::new(),
            faces: vec![
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 2, 1],
                [5, 3, 2],
                [5, 4, 3],
                [5, 1, 4],
            ],
        };
        let before_spike = mesh.vertices[0].z;
        taubin_smooth(&mut mesh, 10);
        let after_spike = mesh.vertices[0].z;
        assert!(after_spike < before_spike * 0.5, "spike not attenuated");
        // Volume preservation (roughly): the equator ring must not collapse.
        let r = mesh.vertices[1].length();
        assert!(r > 0.5, "mesh collapsed: equator radius {r}");
        assert!(mesh.vertices.iter().all(|v| v.is_finite()));
    }
}
