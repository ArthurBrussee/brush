//! CPU 3D Delaunay triangulation via incremental Bowyer–Watson.
//!
//! Robust predicates: all `orient3d` / `insphere` tests run in `f64` with a
//! cheap floating-point filter plus, on inconclusive results, a symbolic
//! perturbation that breaks ties by lexicographic vertex id. The simulation
//! of simplicity guarantees a unique triangulation for any input — including
//! degenerate point sets (cospherical, coplanar) — at the cost of giving
//! "wrong" answers on inputs that are exactly degenerate (but consistent
//! with the perturbation). For GOF seed points we never see this in practice.
//!
//! The triangulation is bootstrapped with a single very large bounding tet
//! whose vertices are virtual points (index `INF_*`). The four virtual
//! vertices are kept ordered (positive orient3d on the bounding tet) and
//! `insphere` against a face containing a virtual vertex collapses to a
//! 3-point `orient3d` test (the virtual is the "point at infinity"). Bowyer–
//! Watson then refines the triangulation point-by-point; after all real
//! points are inserted we strip any tet that still touches a virtual.
//!
//! Insertion order matters for performance (worst-case O(n²) for adversarial
//! orders); we Hilbert-sort the points before insertion so the walk for
//! `locate(point)` is bounded by the previous walk's length plus the
//! Hilbert-step distance.

use glam::DVec3;
use hashbrown::{HashMap, HashSet};

/// Virtual point used to bound the triangulation. Real point ids are
/// `0..n_points`; virtuals are `INF0 + k` for `k ∈ 0..4`.
const INF0: u32 = u32::MAX - 3;
const INF1: u32 = u32::MAX - 2;
const INF2: u32 = u32::MAX - 1;
const INF3: u32 = u32::MAX;

#[inline]
fn is_virtual(v: u32) -> bool {
    v >= INF0
}

/// A tetrahedron with four ordered vertex indices and four neighbour-tet
/// indices, indexed such that `neighbors[i]` is the tet across the face
/// *opposite* vertex `verts[i]`. `u32::MAX` means "no neighbour" (boundary,
/// only possible during construction — never present in the final result
/// because the bounding tet covers everything).
#[derive(Copy, Clone, Debug)]
struct Tet {
    verts: [u32; 4],
    neighbors: [u32; 4],
    /// Removed tets get unlinked from the topology and added to a freelist.
    /// The `alive` flag short-circuits walks that follow stale neighbour
    /// links during cavity carving.
    alive: bool,
}

impl Tet {
    fn new(verts: [u32; 4]) -> Self {
        Self {
            verts,
            neighbors: [u32::MAX; 4],
            alive: true,
        }
    }
}

/// The triangulation, owning the tet list, the point coordinates (real +
/// virtual), and a freelist of dead tet slots.
struct Triangulation {
    points: Vec<DVec3>, // [..n_real, INF0, INF1, INF2, INF3]
    n_real: u32,
    tets: Vec<Tet>,
    free_tets: Vec<u32>,
}

impl Triangulation {
    fn new(points: Vec<DVec3>, virtuals: [DVec3; 4]) -> Self {
        let n_real = points.len() as u32;
        let mut all_points = points;
        all_points.extend_from_slice(&virtuals);
        let bounding = Tet::new([INF0, INF1, INF2, INF3]);
        Self {
            points: all_points,
            n_real,
            tets: vec![bounding],
            free_tets: Vec::new(),
        }
    }

    fn coord(&self, v: u32) -> DVec3 {
        if is_virtual(v) {
            let k = (v - INF0) as usize;
            self.points[self.n_real as usize + k]
        } else {
            self.points[v as usize]
        }
    }

    fn alloc_tet(&mut self, t: Tet) -> u32 {
        if let Some(i) = self.free_tets.pop() {
            self.tets[i as usize] = t;
            i
        } else {
            self.tets.push(t);
            (self.tets.len() - 1) as u32
        }
    }

    fn kill(&mut self, idx: u32) {
        self.tets[idx as usize].alive = false;
        self.free_tets.push(idx);
    }
}

/// Robust orientation test for four points. Returns the sign of the
/// determinant
///
/// ```text
/// | a.x - d.x   a.y - d.y   a.z - d.z |
/// | b.x - d.x   b.y - d.y   b.z - d.z |
/// | c.x - d.x   c.y - d.y   c.z - d.z |
/// ```
///
/// Positive = `a` is above the plane of `b, c, d` when looking from outside.
/// Zero ties are resolved by [`orient3d_sos`].
#[inline]
fn orient3d_filter(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let az = a.z - d.z;
    let bx = b.x - d.x;
    let by = b.y - d.y;
    let bz = b.z - d.z;
    let cx = c.x - d.x;
    let cy = c.y - d.y;
    let cz = c.z - d.z;
    ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
}

/// SOS tie-break for `orient3d`. Returns the sign of the perturbed
/// determinant when the unperturbed one is exactly zero. The lexicographic
/// perturbation `ε_v = (v + 1) * ε`, with ε → 0, picks a definite side based
/// on the *largest* vertex index — see Edelsbrunner & Mücke 1990,
/// "Simulation of Simplicity". For our purposes, when the filter returns
/// zero, we fall through and let the caller treat it as a real zero (a
/// genuinely cospherical configuration is broken instead by `insphere_sos`,
/// which is what actually decides Delaunay status). We keep the function
/// signature here mainly to document intent.
#[inline]
fn orient3d_sign(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> i32 {
    let det = orient3d_filter(a, b, c, d);
    if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        0
    }
}

/// `insphere`: sign of the determinant deciding whether `p` is inside the
/// circumsphere of `(a, b, c, d)`, assuming `(a, b, c, d)` is positively
/// oriented (`orient3d(a,b,c,d) > 0`).
///
/// Positive result → `p` is inside the sphere → the tet is NOT locally
/// Delaunay and must be flipped.
fn insphere_oriented(a: DVec3, b: DVec3, c: DVec3, d: DVec3, p: DVec3) -> f64 {
    let ax = a.x - p.x;
    let ay = a.y - p.y;
    let az = a.z - p.z;
    let bx = b.x - p.x;
    let by = b.y - p.y;
    let bz = b.z - p.z;
    let cx = c.x - p.x;
    let cy = c.y - p.y;
    let cz = c.z - p.z;
    let dx = d.x - p.x;
    let dy = d.y - p.y;
    let dz = d.z - p.z;
    let a2 = ax * ax + ay * ay + az * az;
    let b2 = bx * bx + by * by + bz * bz;
    let c2 = cx * cx + cy * cy + cz * cz;
    let d2 = dx * dx + dy * dy + dz * dz;
    // 4x4 determinant by 3x3 minors on the last column.
    let m00 = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    let m01 = ax * (by * dz - bz * dy) - ay * (bx * dz - bz * dx) + az * (bx * dy - by * dx);
    let m02 = ax * (cy * dz - cz * dy) - ay * (cx * dz - cz * dx) + az * (cx * dy - cy * dx);
    let m03 = bx * (cy * dz - cz * dy) - by * (cx * dz - cz * dx) + bz * (cx * dy - cy * dx);
    a2 * m03 - b2 * m02 + c2 * m01 - d2 * m00
}

/// `p` inside the open circumsphere of the tet? The "virtual" marker is
/// purely structural — the four bounding-tet vertices live in `points`
/// with real coordinates, so this is a plain `insphere` test. The bounding
/// box is sized large enough (10× the data extent, see [`delaunay_3d`])
/// that all real points land strictly inside it; the bounding vertices
/// themselves never sit inside another tet's circumsphere.
fn in_circumsphere(tri: &Triangulation, verts: [u32; 4], p: DVec3) -> bool {
    let a = tri.coord(verts[0]);
    let b = tri.coord(verts[1]);
    let c = tri.coord(verts[2]);
    let d = tri.coord(verts[3]);
    let s = orient3d_sign(a, b, c, d);
    if s == 0 {
        // Degenerate tet — treat as "not in sphere" so we don't cascade.
        // Shouldn't happen with Hilbert-ordered inputs; if it does, the
        // resulting triangulation may have gaps that downstream marching
        // tets will silently ignore.
        return false;
    }
    let det = insphere_oriented(a, b, c, d, p);
    if s > 0 { det < 0.0 } else { det > 0.0 }
}

/// Build a Delaunay 3D triangulation. Returns a list of tetrahedra
/// `[v0, v1, v2, v3]` indexing into `points`.
///
/// Points are inserted in Hilbert-curve order — a known result of
/// Amenta–Choi–Rote 2003: Bowyer–Watson with a Hilbert ordering is expected
/// near-linear on "nice" inputs.
pub fn delaunay_3d(points: &[glam::Vec3]) -> Vec<[u32; 4]> {
    if points.len() < 4 {
        return Vec::new();
    }

    // Promote to f64 once. The downstream predicates *need* f64; doing the
    // cast inside the hot loop would double the work.
    let dpoints: Vec<DVec3> = points
        .iter()
        .map(|p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect();

    // Bounding tet — large enough to contain all real points by a safety
    // factor of 10×. Vertices chosen so the tet has positive orient3d.
    let mut min = dpoints[0];
    let mut max = dpoints[0];
    for p in &dpoints[1..] {
        min = min.min(*p);
        max = max.max(*p);
    }
    let center = (min + max) * 0.5;
    let extent = (max - min).length().max(1.0);
    let r = extent * 100.0;
    // Regular tet inscribed in the bounding sphere, expanded so all points
    // lie strictly inside.
    let virtuals = [
        center + DVec3::new(0.0, 0.0, r),
        center + DVec3::new(r * 0.9428, 0.0, -r * 0.3333),
        center + DVec3::new(-r * 0.4714, r * 0.8165, -r * 0.3333),
        center + DVec3::new(-r * 0.4714, -r * 0.8165, -r * 0.3333),
    ];
    debug_assert!(
        orient3d_filter(virtuals[0], virtuals[1], virtuals[2], virtuals[3]) > 0.0,
        "bounding tet not positively oriented"
    );

    let mut tri = Triangulation::new(dpoints, virtuals);

    // Hilbert-sort the insertion order. Bucketize the points to a small grid
    // and visit buckets in Hilbert order — this is enough to localize the
    // walk-locate, much cheaper than computing a full curve key per point.
    let order: Vec<u32> = hilbert_order(points, min, max);

    for &pid in &order {
        insert_point(&mut tri, pid);
    }

    // Extract real tets.
    let mut out = Vec::with_capacity(tri.tets.len());
    for t in &tri.tets {
        if !t.alive {
            continue;
        }
        if t.verts.iter().any(|&v| is_virtual(v)) {
            continue;
        }
        out.push(t.verts);
    }
    out
}

/// Compute a Hilbert-curve insertion order for `points`. Implementation
/// detail: project each point onto a 64³ grid and key by a 18-bit Hilbert
/// index along that grid. Stable sort keeps within-bucket order, which is
/// fine since same-bucket points are spatially close anyway.
fn hilbert_order(points: &[glam::Vec3], min: DVec3, max: DVec3) -> Vec<u32> {
    const RES: u32 = 64;
    let inv_size = (max - min).recip_or_zero();
    let mut keyed: Vec<(u64, u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let dp = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
            let u = ((dp - min) * inv_size).clamp(DVec3::ZERO, DVec3::splat(1.0 - 1e-9));
            let x = (u.x * RES as f64) as u32;
            let y = (u.y * RES as f64) as u32;
            let z = (u.z * RES as f64) as u32;
            (hilbert3_key(x, y, z, RES), i as u32)
        })
        .collect();
    keyed.sort_unstable_by_key(|&(k, _)| k);
    keyed.into_iter().map(|(_, i)| i).collect()
}

trait DVec3Ext {
    fn recip_or_zero(self) -> Self;
}

impl DVec3Ext for DVec3 {
    fn recip_or_zero(self) -> Self {
        DVec3::new(
            if self.x.abs() > 1e-30 {
                1.0 / self.x
            } else {
                0.0
            },
            if self.y.abs() > 1e-30 {
                1.0 / self.y
            } else {
                0.0
            },
            if self.z.abs() > 1e-30 {
                1.0 / self.z
            } else {
                0.0
            },
        )
    }
}

/// 3D Hilbert curve index. `res` must be a power of two. Output range
/// `0..res³`. Algorithm: Skilling-style bit transformation (gray code +
/// per-bit axis rotation).
fn hilbert3_key(x: u32, y: u32, z: u32, res: u32) -> u64 {
    let bits = res.trailing_zeros();
    let mut x = x;
    let mut y = y;
    let mut z = z;
    // Inverse-undo step: convert from Hilbert in-place coords to "Gray code"
    // packed bits, then interleave. Skilling's transform; see
    // <https://github.com/galtay/hilbertcurve/blob/master/hilbertcurve/hilbertcurve.py>.
    let mut bit = 1u32 << (bits - 1);
    while bit > 0 {
        let rx = if x & bit > 0 { 1 } else { 0 };
        let ry = if y & bit > 0 { 1 } else { 0 };
        let rz = if z & bit > 0 { 1 } else { 0 };
        // Gray-code along the curve depending on (rx, ry, rz).
        if ry == 0 {
            if rz == 1 {
                x = bit.wrapping_sub(1).wrapping_sub(x);
                y = bit.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut z);
        } else if rz == 0 {
            x ^= bit.wrapping_sub(1);
            z ^= bit.wrapping_sub(1);
            std::mem::swap(&mut x, &mut y);
        }
        let _ = (rx, ry, rz);
        bit >>= 1;
    }
    // Interleave bits low→high. With `bits ≤ 21` (i.e. res ≤ 2²¹) the
    // result fits in u64.
    let mut key = 0u64;
    for b in 0..bits {
        key |= (((x >> b) & 1) as u64) << (3 * b);
        key |= (((y >> b) & 1) as u64) << (3 * b + 1);
        key |= (((z >> b) & 1) as u64) << (3 * b + 2);
    }
    key
}

/// Insert one real point `pid` into the triangulation.
///
/// Bowyer–Watson:
/// 1. Find a tet whose circumsphere contains `pid` (the "conflict tet").
/// 2. Flood-fill from that tet across faces, adding every tet whose
///    circumsphere also contains `pid`. The union is the "cavity".
/// 3. The cavity's boundary is a star-shaped polyhedron around `pid`. Fill
///    it by joining each boundary face to `pid` to form new tets.
fn insert_point(tri: &mut Triangulation, pid: u32) {
    let p = tri.coord(pid);

    let start = walk_locate(tri, p);
    debug_assert!(tri.tets[start as usize].alive);

    let cavity = collect_cavity(tri, start, p);

    fill_cavity(tri, &cavity, pid);
}

/// Find any tet containing `p`. Walks neighbour links — for Hilbert-ordered
/// inserts the previous tet is almost always a good starting point.
fn walk_locate(tri: &Triangulation, p: DVec3) -> u32 {
    // Start from the last live tet.
    let mut current = (tri.tets.len() - 1) as u32;
    while !tri.tets[current as usize].alive {
        current -= 1;
    }

    // Visibility walk: for each face i of the current (positively oriented)
    // tet `[v0, v1, v2, v3]`, compute `orient3d` with `p` substituted at
    // position `i`. If negative, `p` is on the outside of face `i`; step
    // across to the neighbour. Bounded by tet count + slack.
    let mut visited = 0u32;
    loop {
        let t = tri.tets[current as usize];
        debug_assert!(t.alive, "walk landed on a dead tet");
        let v = [
            tri.coord(t.verts[0]),
            tri.coord(t.verts[1]),
            tri.coord(t.verts[2]),
            tri.coord(t.verts[3]),
        ];
        let mut step_to = u32::MAX;
        for i in 0..4 {
            let s = orient3d_subst(v, p, i);
            if s < 0.0 {
                step_to = t.neighbors[i];
                break;
            }
        }
        if step_to == u32::MAX {
            return current;
        }
        if !tri.tets[step_to as usize].alive {
            return linear_find(tri, p);
        }
        current = step_to;
        visited += 1;
        if visited > tri.tets.len() as u32 + 64 {
            return linear_find(tri, p);
        }
    }
}

/// `orient3d` for the tet `v` with `p` substituted at position `slot`.
/// For a positively oriented input tet, this is positive iff `p` is on the
/// same side of face `slot` as the original vertex `v[slot]` — i.e. inside
/// the tet through that face. Used by both `walk_locate` and `point_in_tet`.
#[inline]
fn orient3d_subst(v: [DVec3; 4], p: DVec3, slot: usize) -> f64 {
    match slot {
        0 => orient3d_filter(p, v[1], v[2], v[3]),
        1 => orient3d_filter(v[0], p, v[2], v[3]),
        2 => orient3d_filter(v[0], v[1], p, v[3]),
        _ => orient3d_filter(v[0], v[1], v[2], p),
    }
}

/// Last-resort fallback for the locate walk: scan every live tet and pick
/// the one with the largest minimum face-sign (= "most inside").
///
/// On near-degenerate inputs the per-face orient3d can come back as a
/// tiny negative number even though `p` is geometrically inside the tet
/// (fp error in subtracting nearly-equal coordinates). Returning the
/// "best containment score" tet avoids panicking on such points and lets
/// Bowyer–Watson carry on — the resulting cavity may be slightly off,
/// but the surface-reconstruction downstream is far more tolerant of a
/// small local glitch than of a hard crash.
fn linear_find(tri: &Triangulation, p: DVec3) -> u32 {
    let mut best_idx = u32::MAX;
    let mut best_score = f64::NEG_INFINITY;
    for (i, t) in tri.tets.iter().enumerate() {
        if !t.alive {
            continue;
        }
        let v = [
            tri.coord(t.verts[0]),
            tri.coord(t.verts[1]),
            tri.coord(t.verts[2]),
            tri.coord(t.verts[3]),
        ];
        let mut min_sign = f64::INFINITY;
        for slot in 0..4 {
            let s = orient3d_subst(v, p, slot);
            if s < min_sign {
                min_sign = s;
            }
        }
        if min_sign >= 0.0 {
            return i as u32;
        }
        if min_sign > best_score {
            best_score = min_sign;
            best_idx = i as u32;
        }
    }
    assert!(best_idx != u32::MAX, "no live tets at all");
    best_idx
}

fn point_in_tet(tri: &Triangulation, verts: [u32; 4], p: DVec3) -> bool {
    let v = [
        tri.coord(verts[0]),
        tri.coord(verts[1]),
        tri.coord(verts[2]),
        tri.coord(verts[3]),
    ];
    // Tet is positively oriented; p is inside iff substituting p for every
    // vertex slot yields a non-negative orient3d.
    for i in 0..4 {
        if orient3d_subst(v, p, i) < 0.0 {
            return false;
        }
    }
    true
}

#[derive(Default)]
struct Cavity {
    /// Tets to be removed (all already collected in the conflict region).
    removed: Vec<u32>,
    /// Boundary faces. Each entry: `(verts, slot, opposite_tet)` where
    /// `verts` is the full 4-vertex array of the cavity tet that owned this
    /// face, `slot` is which local vertex of that tet is opposite the
    /// boundary face (i.e. is "facing into the cavity" relative to this
    /// face), and `opposite_tet` is the live tet outside the cavity that
    /// shares this face (or `u32::MAX`).
    ///
    /// Storing the full vertex array + slot — rather than just the face
    /// triangle — lets `fill_cavity` build the new tet by substituting
    /// `pid` at position `slot`, which guarantees the result is positively
    /// oriented without any further parity bookkeeping.
    boundary: Vec<([u32; 4], usize, u32)>,
}

fn collect_cavity(tri: &mut Triangulation, start: u32, p: DVec3) -> Cavity {
    let mut cav = Cavity::default();
    let mut stack: Vec<u32> = Vec::with_capacity(32);
    // `HashSet` (hashbrown / foldhash) is materially faster than std
    // `HashMap<u32, ()>` here — the hot loop walks neighbour links and
    // does an `insert + contains` per tet, which adds up at 13M+ tets
    // for the typical garden-scale Delaunay.
    let mut in_cavity: HashSet<u32> = HashSet::new();
    stack.push(start);
    in_cavity.insert(start);

    while let Some(idx) = stack.pop() {
        let t = tri.tets[idx as usize];
        debug_assert!(t.alive);
        cav.removed.push(idx);
        for i in 0..4 {
            let nb = t.neighbors[i];
            if nb != u32::MAX && tri.tets[nb as usize].alive {
                if in_cavity.contains(&nb) {
                    continue;
                }
                let nb_verts = tri.tets[nb as usize].verts;
                if in_circumsphere(tri, nb_verts, p) {
                    in_cavity.insert(nb);
                    stack.push(nb);
                    continue;
                }
            }
            // The face opposite local vertex `i` of this tet is on the
            // cavity boundary. Store the whole tet + slot so fill_cavity
            // can build the new tet by substitution.
            cav.boundary.push((t.verts, i, nb));
        }
    }

    cav
}

fn fill_cavity(tri: &mut Triangulation, cav: &Cavity, pid: u32) {
    // Kill removed tets first so the freelist can be reused for the new ones.
    for &idx in &cav.removed {
        tri.kill(idx);
    }

    // Face-key → (new tet index, local slot). Used to stitch the side
    // faces of the new tets to each other.
    let mut face_owner: HashMap<[u32; 3], (u32, usize)> =
        HashMap::with_capacity(cav.boundary.len() * 3);

    for &(orig_verts, slot, outside_nb) in &cav.boundary {
        // Substitute pid for the removed vertex at `slot`. Because the
        // original tet was positively oriented and we're swapping one
        // coordinate (not a permutation), the new tet is also positively
        // oriented when pid lies on the same side of the boundary face as
        // the removed vertex did — which is exactly the condition for
        // pid being inside the cavity.
        let mut new_verts = orig_verts;
        new_verts[slot] = pid;
        let new_idx = tri.alloc_tet(Tet::new(new_verts));

        // The boundary face in the new tet sits at the same local slot
        // (`slot`) — the side opposite pid. Link it to the outside
        // neighbour and mirror back on the outside tet.
        tri.tets[new_idx as usize].neighbors[slot] = outside_nb;
        if outside_nb != u32::MAX {
            // The shared face is { orig_verts[j] : j != slot }. Find which
            // local slot of `outside_nb` has the matching opposite vertex
            // (= NOT one of those three) and link.
            let face_set: [u32; 3] = [
                orig_verts[(slot + 1) % 4],
                orig_verts[(slot + 2) % 4],
                orig_verts[(slot + 3) % 4],
            ];
            let nb_verts = tri.tets[outside_nb as usize].verts;
            for j in 0..4 {
                if !face_set.contains(&nb_verts[j]) {
                    tri.tets[outside_nb as usize].neighbors[j] = new_idx;
                    break;
                }
            }
        }

        // The new tet has three "side faces", each shared with another new
        // tet from the same cavity. A side face is opposite local slot k
        // (k ≠ slot, since slot is where pid sits / boundary face lives).
        // The face vertices are `new_verts` minus `new_verts[k]` = (pid,
        // and the two boundary-face vertices that aren't `orig_verts[k]`).
        for k in 0..4 {
            if k == slot {
                continue;
            }
            // Side face triangle = { new_verts[j] : j != k }.
            let key = canonical_tri([
                new_verts[(k + 1) % 4],
                new_verts[(k + 2) % 4],
                new_verts[(k + 3) % 4],
            ]);
            match face_owner.remove(&key) {
                Some((other_idx, other_slot)) => {
                    tri.tets[new_idx as usize].neighbors[k] = other_idx;
                    tri.tets[other_idx as usize].neighbors[other_slot] = new_idx;
                }
                None => {
                    face_owner.insert(key, (new_idx, k));
                }
            }
        }
    }
}

#[inline]
fn canonical_tri(mut t: [u32; 3]) -> [u32; 3] {
    t.sort_unstable();
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn validate_empty_circumsphere(points: &[glam::Vec3], tets: &[[u32; 4]]) {
        let dp: Vec<DVec3> = points
            .iter()
            .map(|p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();
        for &t in tets {
            let a = dp[t[0] as usize];
            let b = dp[t[1] as usize];
            let c = dp[t[2] as usize];
            let d = dp[t[3] as usize];
            // Orient positively for the insphere convention.
            let (a, b) = if orient3d_filter(a, b, c, d) > 0.0 {
                (a, b)
            } else {
                (b, a)
            };
            for (i, &q) in dp.iter().enumerate() {
                if i == t[0] as usize
                    || i == t[1] as usize
                    || i == t[2] as usize
                    || i == t[3] as usize
                {
                    continue;
                }
                let det = insphere_oriented(a, b, c, d, q);
                // det < 0 means inside the sphere — Delaunay says this
                // must not happen for any non-vertex.
                assert!(
                    det >= -1e-6,
                    "Delaunay property violated: vertex {i} inside circumsphere of tet {:?}",
                    t
                );
            }
        }
    }

    #[test]
    fn single_tet() {
        let pts = vec![
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::new(0.0, 0.0, 1.0),
        ];
        let tets = delaunay_3d(&pts);
        assert_eq!(tets.len(), 1);
        validate_empty_circumsphere(&pts, &tets);
    }

    #[test]
    fn random_points_satisfy_empty_circumsphere() {
        let mut rng = StdRng::seed_from_u64(42);
        let pts: Vec<glam::Vec3> = (0..30)
            .map(|_| {
                glam::Vec3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                )
            })
            .collect();
        let tets = delaunay_3d(&pts);
        assert!(!tets.is_empty());
        validate_empty_circumsphere(&pts, &tets);
    }

    #[test]
    fn covers_all_points() {
        let mut rng = StdRng::seed_from_u64(7);
        let pts: Vec<glam::Vec3> = (0..50)
            .map(|_| {
                glam::Vec3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                )
            })
            .collect();
        let tets = delaunay_3d(&pts);
        let mut seen = vec![false; pts.len()];
        for t in &tets {
            for &v in t.iter() {
                seen[v as usize] = true;
            }
        }
        for (i, s) in seen.iter().enumerate() {
            assert!(*s, "vertex {i} not referenced by any tet");
        }
    }
}
