//! Marching tetrahedra. Produces one [`Crossing`] per tet edge that straddles
//! the iso-value (`sdf < 0` vs `sdf > 0`), plus the triangle list. The 16-row
//! triangle table and 6-row edge table mirror the Kaolin implementation that
//! GOF's `utils/tetmesh.py` adapted from.
//!
//! Edges are deduplicated by sorted-pair vertex id so a single crossing point
//! is shared between all tets that touch the edge. Downstream
//! [`crate::refine`] refines each crossing on the GPU.

use hashbrown::HashMap;

/// Six edges of a tet, as pairs of vertex local indices `(0..4, 0..4)`.
const EDGE_VERTS: [(u8, u8); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Per-pattern triangle table. Indices into the 6 edges. `-1` is sentinel.
/// Patterns are keyed by `bit_i = (sdf[v_i] > 0) << i`.
#[rustfmt::skip]
const TRIANGLE_TABLE: [[i8; 6]; 16] = [
    [-1, -1, -1, -1, -1, -1],
    [ 1,  0,  2, -1, -1, -1],
    [ 4,  0,  3, -1, -1, -1],
    [ 1,  4,  2,  1,  3,  4],
    [ 3,  1,  5, -1, -1, -1],
    [ 2,  3,  0,  2,  5,  3],
    [ 1,  4,  0,  1,  5,  4],
    [ 4,  2,  5, -1, -1, -1],
    [ 4,  5,  2, -1, -1, -1],
    [ 4,  1,  0,  4,  5,  1],
    [ 3,  2,  0,  3,  5,  2],
    [ 1,  3,  5, -1, -1, -1],
    [ 4,  1,  2,  4,  3,  1],
    [ 3,  0,  4, -1, -1, -1],
    [ 2,  0,  1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
];

const NUM_TRIS: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0];

/// A single edge of the input tet mesh that crosses the iso-surface. `(a, b)`
/// are vertex indices into the seed-point set; the crossing point will be
/// found by [`crate::refine`] somewhere on the segment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Crossing {
    pub a: u32,
    pub b: u32,
}

pub struct MarchingTetResult {
    /// Unique crossing edges, in deterministic order. The output mesh's
    /// vertex `i` corresponds to `crossings[i]`.
    pub crossings: Vec<Crossing>,
    /// Triangles, with indices into `crossings`.
    pub faces: Vec<[u32; 3]>,
}

/// Run marching tets. `sdf[i]` is the signed value at vertex `i`; the
/// iso-surface is `sdf = 0`.
pub fn marching_tets(tets: &[[u32; 4]], sdf: &[f32]) -> MarchingTetResult {
    let mut crossing_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut crossings: Vec<Crossing> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    for tet in tets {
        let occ = [
            sdf[tet[0] as usize] > 0.0,
            sdf[tet[1] as usize] > 0.0,
            sdf[tet[2] as usize] > 0.0,
            sdf[tet[3] as usize] > 0.0,
        ];
        let pattern = (occ[0] as usize)
            | ((occ[1] as usize) << 1)
            | ((occ[2] as usize) << 2)
            | ((occ[3] as usize) << 3);
        let n = NUM_TRIS[pattern] as usize;
        if n == 0 {
            continue;
        }

        // Resolve / intern the six tet edges; for edges that don't actually
        // cross the surface we never index into them, so it's OK to leave
        // them as a sentinel. We compute all six upfront so the triangle
        // table indices map straight through.
        let mut edge_idx = [u32::MAX; 6];
        for (k, &(ea, eb)) in EDGE_VERTS.iter().enumerate() {
            // Only edges with one vertex inside and one outside are needed.
            if occ[ea as usize] == occ[eb as usize] {
                continue;
            }
            let va = tet[ea as usize];
            let vb = tet[eb as usize];
            let key = if va < vb { (va, vb) } else { (vb, va) };
            let idx = if let Some(&i) = crossing_map.get(&key) {
                i
            } else {
                let i = crossings.len() as u32;
                crossings.push(Crossing { a: key.0, b: key.1 });
                crossing_map.insert(key, i);
                i
            };
            edge_idx[k] = idx;
        }

        let row = &TRIANGLE_TABLE[pattern];
        for t in 0..n {
            let i0 = row[t * 3] as usize;
            let i1 = row[t * 3 + 1] as usize;
            let i2 = row[t * 3 + 2] as usize;
            faces.push([edge_idx[i0], edge_idx[i1], edge_idx[i2]]);
        }
    }

    MarchingTetResult { crossings, faces }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tet with v0 inside (sdf < 0) and v1..v3 outside → one triangle on
    /// edges (0,1), (0,2), (0,3).
    #[test]
    fn single_inside_vertex_makes_one_tri() {
        let tets = vec![[0u32, 1, 2, 3]];
        let sdf = vec![-1.0, 1.0, 1.0, 1.0];
        let out = marching_tets(&tets, &sdf);
        assert_eq!(out.crossings.len(), 3);
        assert_eq!(out.faces.len(), 1);
    }

    /// Two adjacent tets sharing a face that contains the crossing edge: the
    /// crossing vertex must be deduplicated so the two triangles share an
    /// edge in the output mesh.
    #[test]
    fn shared_crossings_are_deduplicated() {
        let tets = vec![[0u32, 1, 2, 3], [0u32, 1, 2, 4]];
        let sdf = vec![-1.0, 1.0, 1.0, 1.0, 1.0];
        let out = marching_tets(&tets, &sdf);
        // The two tets share edges (0,1), (0,2). Each contributes one tri
        // (3 edges), but the shared two are merged → 3 + 3 − 2 = 4 unique
        // crossings.
        assert_eq!(out.crossings.len(), 4);
        assert_eq!(out.faces.len(), 2);
    }

    /// All-inside or all-outside tets must produce no triangles.
    #[test]
    fn uniform_signs_emit_nothing() {
        let tets = vec![[0u32, 1, 2, 3], [0u32, 1, 2, 3]];
        let sdf_all_in = vec![-1.0, -2.0, -3.0, -4.0];
        let sdf_all_out = vec![1.0, 2.0, 3.0, 4.0];
        assert!(marching_tets(&tets, &sdf_all_in).faces.is_empty());
        assert!(marching_tets(&tets, &sdf_all_out).faces.is_empty());
    }
}
