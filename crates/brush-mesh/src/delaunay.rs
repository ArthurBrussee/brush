//! Lock-grid parallel 3D Delaunay (Bowyer–Watson).
//!
//! Soundness: every per-tet field that can be touched by more than one thread
//! is atomic (`verts`/`neighbors`/`alive`), so there are no torn reads and no
//! `unsafe` — the only shared owner is `&Arena`, which is `Sync` because all
//! its interior mutability goes through atomics. The lock grid provides
//! *logical* exclusion (no two threads carve overlapping cavities at once),
//! not memory safety.
//!
//! Strategy: insert in Hilbert order so cavities stay tiny (the property that
//! makes incremental Bowyer–Watson fast). Bootstrap a small seed triangulation
//! serially, then insert in parallel levels at geometrically increasing
//! density, each with a lock grid sized to the density the level ends at.
//! Each insertion locks the 3×3×3 grid-cell block around its point; on a lock
//! conflict or a cavity that escapes the locked region the point defers to a
//! retry pass (the mesh is finer by then) and finally to a serial tail.
//!
//! Predicates (`orient3d`, `insphere`) run in f64. The triangulation is
//! bootstrapped from a single very large bounding tet whose vertices are
//! virtual points (index `INF0..`); tets still touching a virtual are
//! stripped from the output.

use glam::DVec3;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Virtual point used to bound the triangulation. Real point ids are
/// `0..n_points`; virtuals are `INF0 + k` for `k in 0..4`.
const INF0: u32 = u32::MAX - 3;

#[inline]
fn is_virtual(v: u32) -> bool {
    v >= INF0
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

/// `orient3d` for the tet `v` with `p` substituted at position `slot`.
/// For a positively oriented input tet, this is positive iff `p` is on the
/// same side of face `slot` as the original vertex `v[slot]` — i.e. inside
/// the tet through that face. Used by `walk_locate`.
#[inline]
fn orient3d_subst(v: &[DVec3; 4], p: DVec3, slot: usize) -> f64 {
    match slot {
        0 => orient3d_filter(p, v[1], v[2], v[3]),
        1 => orient3d_filter(v[0], p, v[2], v[3]),
        2 => orient3d_filter(v[0], v[1], p, v[3]),
        _ => orient3d_filter(v[0], v[1], v[2], p),
    }
}

/// Compute a Hilbert-curve insertion order for `points`. We quantise each
/// axis at the full 21-bit resolution (2²¹ = ~2M values per axis,
/// finer than f32 can distinguish), so each point gets a unique key and
/// there's no bucket-tied tie-breaking — within-bucket order was the
/// hidden source of long `walk_locate` hops as N grew. The 3-axis Hilbert
/// key fits in 63 bits.
fn hilbert_order(points: &[glam::Vec3], min: DVec3, max: DVec3) -> Vec<u32> {
    use rayon::prelude::*;
    const RES: u32 = 1 << 21;
    let inv_size = (max - min).recip_or_zero();
    // Pre-compute keys; sort an indices vec by key. Avoids the
    // Vec<(u64,u32)> → Vec<u32> reshape after sort.
    let keys: Vec<u64> = points
        .par_iter()
        .map(|p| {
            let dp = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
            let u = ((dp - min) * inv_size).clamp(DVec3::ZERO, DVec3::splat(1.0 - 1e-9));
            let x = (u.x * RES as f64) as u32;
            let y = (u.y * RES as f64) as u32;
            let z = (u.z * RES as f64) as u32;
            hilbert3_key(x, y, z, RES)
        })
        .collect();
    let mut order: Vec<u32> = (0..points.len() as u32).collect();
    // Tie-break equal keys by index so the order (and thus the
    // triangulation walk behaviour) is deterministic across thread counts.
    order.par_sort_unstable_by_key(|&i| (keys[i as usize], i));
    order
}

trait DVec3Ext {
    fn recip_or_zero(self) -> Self;
}

impl DVec3Ext for DVec3 {
    fn recip_or_zero(self) -> Self {
        Self::new(
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
        // Skilling's transform only branches on ry/rz; rx is implicit in
        // the bit-pattern of x and doesn't need its own variable.
        let ry = i32::from(y & bit > 0);
        let rz = i32::from(z & bit > 0);
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

/// Atomic tetrahedron. `verts` is written once at creation and then only read,
/// but lives in atomics anyway so a concurrent reader following a freshly
/// linked neighbour never observes a torn value.
struct Tet {
    verts: [AtomicU32; 4],
    neighbors: [AtomicU32; 4],
    alive: AtomicBool,
}

impl Tet {
    fn dead() -> Self {
        Self {
            verts: [const { AtomicU32::new(0) }; 4],
            neighbors: [const { AtomicU32::new(u32::MAX) }; 4],
            alive: AtomicBool::new(false),
        }
    }
    #[inline]
    fn verts(&self) -> [u32; 4] {
        [
            self.verts[0].load(Ordering::Relaxed),
            self.verts[1].load(Ordering::Relaxed),
            self.verts[2].load(Ordering::Relaxed),
            self.verts[3].load(Ordering::Relaxed),
        ]
    }
    #[inline]
    fn neighbor(&self, i: usize) -> u32 {
        self.neighbors[i].load(Ordering::Relaxed)
    }
    #[inline]
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }
    fn set(&self, verts: [u32; 4], neighbors: [u32; 4]) {
        for i in 0..4 {
            self.verts[i].store(verts[i], Ordering::Relaxed);
            self.neighbors[i].store(neighbors[i], Ordering::Relaxed);
        }
        self.alive.store(true, Ordering::Release);
    }
}

/// Fixed-capacity tet arena with a bump allocator (no freelist — dead tets are
/// compacted out at the end). `Sync` via all-atomic interior mutability.
struct Arena {
    points: Vec<DVec3>,
    n_real: u32,
    tets: Box<[Tet]>,
    next: AtomicU32,
}

impl Arena {
    fn new(points: Vec<DVec3>, n_real: u32, virtuals: [DVec3; 4], cap: usize) -> Self {
        let mut all = points;
        all.extend_from_slice(&virtuals);
        // Parallel init: at 32 tets/point this is hundreds of MB of atomics,
        // which a serial fill leaves memory-bandwidth-starved on one core.
        let tets: Box<[Tet]> = (0..cap)
            .into_par_iter()
            .map(|_| Tet::dead())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            points: all,
            n_real,
            tets,
            next: AtomicU32::new(0),
        }
    }
    #[inline]
    fn coord(&self, v: u32) -> DVec3 {
        if is_virtual(v) {
            self.points[self.n_real as usize + (v - INF0) as usize]
        } else {
            self.points[v as usize]
        }
    }
    #[inline]
    fn tet(&self, i: u32) -> &Tet {
        &self.tets[i as usize]
    }
    /// Bump-allocate a tet; returns its index.
    fn alloc(&self, verts: [u32; 4], neighbors: [u32; 4]) -> u32 {
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        self.tets[i as usize].set(verts, neighbors);
        i
    }
    fn len(&self) -> u32 {
        self.next.load(Ordering::Relaxed)
    }
}

/// A grid of spin-locks over space. Locking a set of cells gives exclusive
/// rights to mutate the tets owned by those cells (a tet is owned by the cell
/// of its first real vertex).
struct LockGrid {
    locks: Box<[AtomicBool]>,
    min: DVec3,
    inv: DVec3,
    gdim: u32,
}

impl LockGrid {
    fn new(min: DVec3, max: DVec3, gdim: u32) -> Self {
        let inv = {
            let d = max - min;
            DVec3::new(
                if d.x.abs() > 1e-30 { 1.0 / d.x } else { 0.0 },
                if d.y.abs() > 1e-30 { 1.0 / d.y } else { 0.0 },
                if d.z.abs() > 1e-30 { 1.0 / d.z } else { 0.0 },
            )
        };
        let n = (gdim as usize).pow(3);
        Self {
            locks: (0..n).map(|_| AtomicBool::new(false)).collect(),
            min,
            inv,
            gdim,
        }
    }
    #[inline]
    fn cell_of(&self, q: DVec3) -> (u32, u32, u32) {
        let u = ((q - self.min) * self.inv).clamp(DVec3::ZERO, DVec3::splat(1.0 - 1e-9));
        (
            (u.x * self.gdim as f64) as u32,
            (u.y * self.gdim as f64) as u32,
            (u.z * self.gdim as f64) as u32,
        )
    }
    #[inline]
    fn idx(&self, c: (u32, u32, u32)) -> usize {
        ((c.0 * self.gdim + c.1) * self.gdim + c.2) as usize
    }
    #[inline]
    fn try_lock(&self, cell: usize) -> bool {
        self.locks[cell]
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
    #[inline]
    fn unlock(&self, cell: usize) {
        self.locks[cell].store(false, Ordering::Release);
    }
}

#[inline]
fn in_circumsphere(arena: &Arena, verts: [u32; 4], p: DVec3) -> bool {
    let a = arena.coord(verts[0]);
    let b = arena.coord(verts[1]);
    let c = arena.coord(verts[2]);
    let d = arena.coord(verts[3]);
    insphere_oriented(a, b, c, d, p) < 0.0
}

/// Per-thread scratch reused across insertions.
#[derive(Default)]
struct Scratch {
    visited: rustc_hash::FxHashSet<u32>,
    stack: Vec<u32>,
    removed: Vec<u32>,
    boundary: Vec<([u32; 4], usize, u32)>,
    new_tets: Vec<u32>,
    face_owner: FxHashMap<[u32; 3], (u32, usize)>,
}
/// The set of grid cells a thread holds; used both to bound the cavity walk
/// and to decide whether a tet may be touched.
struct Locked<'a> {
    grid: &'a LockGrid,
    center: (u32, u32, u32),
}
impl Locked<'_> {
    /// Is `cell` within the locked 3×3×3 block around `center`?
    #[inline]
    fn contains_cell(&self, c: (u32, u32, u32)) -> bool {
        let d = |a: u32, b: u32| a.max(b) - a.min(b);
        d(c.0, self.center.0) <= 1 && d(c.1, self.center.1) <= 1 && d(c.2, self.center.2) <= 1
    }
    /// Is this tet owned by a locked cell? Ownership = cell of the tet's
    /// centroid (over real vertices). Centroid keeps a small cavity tet's owner
    /// cell near the query point, so cavities rarely "escape" spuriously the way
    /// a single-vertex rule does. Reads immutable `verts`, so always safe, and
    /// it's a deterministic single cell per tet (keeps ownership exclusive).
    #[inline]
    fn owns(&self, arena: &Arena, verts: [u32; 4]) -> bool {
        let mut sum = DVec3::ZERO;
        let mut cnt = 0u32;
        for &v in &verts {
            if !is_virtual(v) {
                sum += arena.coord(v);
                cnt += 1;
            }
        }
        if cnt == 0 {
            return false; // all-virtual hull tet — never locked
        }
        self.contains_cell(self.grid.cell_of(sum / cnt as f64))
    }
}

/// Visibility-walk point location from `start`. With `bound`, returns `None`
/// if the walk needs to leave the locked region (defer). `visited` guards
/// against cycles on a transiently-inconsistent read.
fn walk_locate(
    arena: &Arena,
    start: u32,
    p: DVec3,
    bound: Option<&Locked>,
    scratch_seen: &mut rustc_hash::FxHashSet<u32>,
) -> Option<u32> {
    scratch_seen.clear();
    let mut current = start;
    loop {
        if !arena.tet(current).is_alive() || !scratch_seen.insert(current) {
            return None;
        }
        let verts = arena.tet(current).verts();
        let v = [
            arena.coord(verts[0]),
            arena.coord(verts[1]),
            arena.coord(verts[2]),
            arena.coord(verts[3]),
        ];
        let mut step = None;
        for i in 0..4 {
            if orient3d_subst(&v, p, i) < 0.0 {
                step = Some(arena.tet(current).neighbor(i));
                break;
            }
        }
        let next = match step {
            None => return Some(current),  // no face failed → p is inside `current`
            Some(INF_NONE) => return None, // walked off the hull (shouldn't happen)
            Some(nb) => nb,
        };
        // If stepping out of the locked region, defer.
        if let Some(b) = bound {
            let nv = arena.tet(next).verts();
            if !b.owns(arena, nv) {
                return None;
            }
        }
        current = next;
    }
}

const INF_NONE: u32 = u32::MAX;

/// Collect the Bowyer–Watson cavity of `p` starting at the located tet `start`,
/// recording removed tets and boundary faces in `scratch`. With `bound`, any
/// cavity or boundary tet outside the locked region aborts to `None` (defer).
fn collect_cavity(
    arena: &Arena,
    start: u32,
    p: DVec3,
    bound: Option<&Locked>,
    scratch: &mut Scratch,
) -> Option<()> {
    scratch.visited.clear();
    scratch.stack.clear();
    scratch.removed.clear();
    scratch.boundary.clear();
    scratch.stack.push(start);
    scratch.visited.insert(start);
    while let Some(idx) = scratch.stack.pop() {
        let tet = arena.tet(idx);
        if !tet.is_alive() {
            return None;
        }
        scratch.removed.push(idx);
        let verts = tet.verts();
        for i in 0..4 {
            let nb = tet.neighbor(i);
            if nb != INF_NONE {
                let nb_tet = arena.tet(nb);
                let nb_verts = nb_tet.verts();
                // Any neighbour we might cross into or relink must be owned by
                // the locked region; otherwise defer.
                if let Some(b) = bound
                    && !b.owns(arena, nb_verts)
                {
                    return None;
                }
                if nb_tet.is_alive() {
                    if scratch.visited.contains(&nb) {
                        continue;
                    }
                    if in_circumsphere(arena, nb_verts, p) {
                        scratch.visited.insert(nb);
                        scratch.stack.push(nb);
                        continue;
                    }
                }
            }
            scratch.boundary.push((verts, i, nb));
        }
    }
    Some(())
}

/// Retriangulate the collected cavity by connecting `pid` to every boundary
/// face. Kills removed tets, allocates the new ones, stitches links. Returns a
/// new tet incident to `pid`.
fn fill_cavity(arena: &Arena, pid: u32, scratch: &mut Scratch) -> u32 {
    for &idx in &scratch.removed {
        arena.tet(idx).alive.store(false, Ordering::Relaxed);
    }
    scratch.new_tets.clear();
    scratch.face_owner.clear();
    let mut last = INF_NONE;
    for &(orig_verts, slot, outside_nb) in &scratch.boundary {
        let mut nv = orig_verts;
        nv[slot] = pid;
        let new_idx = arena.alloc(nv, [INF_NONE; 4]);
        last = new_idx;
        scratch.new_tets.push(new_idx);

        arena.tet(new_idx).neighbors[slot].store(outside_nb, Ordering::Relaxed);
        if outside_nb != INF_NONE {
            let face = [
                orig_verts[(slot + 1) % 4],
                orig_verts[(slot + 2) % 4],
                orig_verts[(slot + 3) % 4],
            ];
            let ov = arena.tet(outside_nb).verts();
            for j in 0..4 {
                if !face.contains(&ov[j]) {
                    arena.tet(outside_nb).neighbors[j].store(new_idx, Ordering::Relaxed);
                    break;
                }
            }
        }
        for k in 0..4 {
            if k == slot {
                continue;
            }
            let mut key = [nv[(k + 1) % 4], nv[(k + 2) % 4], nv[(k + 3) % 4]];
            key.sort_unstable();
            match scratch.face_owner.remove(&key) {
                Some((other, other_slot)) => {
                    arena.tet(new_idx).neighbors[k].store(other, Ordering::Relaxed);
                    arena.tet(other).neighbors[other_slot].store(new_idx, Ordering::Relaxed);
                }
                None => {
                    scratch.face_owner.insert(key, (new_idx, k));
                }
            }
        }
    }
    last
}

/// Last-resort point location: scan alive tets for the one containing `p`.
/// Mirrors the serial `linear_find` fallback for near-degenerate inputs where
/// the visibility walk oscillates.
fn linear_locate(arena: &Arena, p: DVec3) -> u32 {
    let mut best = 0u32;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..arena.len() {
        let tet = arena.tet(i);
        if !tet.is_alive() {
            continue;
        }
        let verts = tet.verts();
        let v = [
            arena.coord(verts[0]),
            arena.coord(verts[1]),
            arena.coord(verts[2]),
            arena.coord(verts[3]),
        ];
        let mut min_sign = f64::INFINITY;
        for slot in 0..4 {
            min_sign = min_sign.min(orient3d_subst(&v, p, slot));
        }
        if min_sign >= 0.0 {
            return i;
        }
        if min_sign > best_score {
            best_score = min_sign;
            best = i;
        }
    }
    best
}

/// Unlocked serial insertion (bootstrap + deferred cleanup). Walks from `hint`,
/// falling back to a linear scan if the walk fails (degenerate seeds).
fn serial_insert(arena: &Arena, pid: u32, hint: u32, scratch: &mut Scratch) -> u32 {
    let p = arena.coord(pid);
    let start = walk_locate(arena, hint, p, None, &mut scratch.visited)
        .unwrap_or_else(|| linear_locate(arena, p));
    if collect_cavity(arena, start, p, None, scratch).is_none() {
        let start = linear_locate(arena, p);
        collect_cavity(arena, start, p, None, scratch).expect("serial collect after relocate");
    }
    fill_cavity(arena, pid, scratch)
}

enum Ins {
    Committed,
    Deferred,
}

/// Attempt a locked parallel insertion of `pid`. Locks the 3×3×3 cell block
/// around `pid`; on any lock conflict, walk escape, or cavity escape, releases
/// and returns `Deferred` (handled later serially).
fn try_insert_locked(
    arena: &Arena,
    grid: &LockGrid,
    cell_seed: &[AtomicU32],
    pid: u32,
    scratch: &mut Scratch,
) -> Ins {
    let p = arena.coord(pid);
    let c = grid.cell_of(p);
    let g = grid.gdim;
    let clamp = |x: i64| x.clamp(0, g as i64 - 1) as u32;
    let mut cells: Vec<usize> = Vec::with_capacity(27);
    for dx in -1..=1i64 {
        for dy in -1..=1i64 {
            for dz in -1..=1i64 {
                let cc = (
                    clamp(c.0 as i64 + dx),
                    clamp(c.1 as i64 + dy),
                    clamp(c.2 as i64 + dz),
                );
                cells.push(grid.idx(cc));
            }
        }
    }
    cells.sort_unstable();
    cells.dedup();

    // Try-lock all (sorted order); release everything on the first failure.
    let mut held = 0usize;
    let mut ok = true;
    for (k, &cell) in cells.iter().enumerate() {
        if grid.try_lock(cell) {
            held = k + 1;
        } else {
            ok = false;
            break;
        }
    }
    let unlock_all = |held: usize| {
        for &cell in &cells[..held] {
            grid.unlock(cell);
        }
    };
    if !ok {
        unlock_all(held);
        return Ins::Deferred;
    }

    let bound = Locked { grid, center: c };
    let result = (|| {
        // Walk start: the centre cell's seed, or (when that seed died
        // without a replacement tet landing in the same cell) any alive
        // seed from the locked block. Without the fallback such points
        // defer on every pass and pile onto the serial tail.
        let mut start = cell_seed[grid.idx(c)].load(Ordering::Relaxed);
        if start == INF_NONE || !arena.tet(start).is_alive() {
            start = INF_NONE;
            for &cell in &cells {
                let cand = cell_seed[cell].load(Ordering::Relaxed);
                if cand != INF_NONE && arena.tet(cand).is_alive() {
                    start = cand;
                    break;
                }
            }
        }
        if start == INF_NONE {
            return None;
        }
        let loc = walk_locate(arena, start, p, Some(&bound), &mut scratch.visited)?;
        collect_cavity(arena, loc, p, Some(&bound), scratch)?;
        fill_cavity(arena, pid, scratch);
        // Refresh seeds for the locked cells touched by the new tets.
        for i in 0..scratch.new_tets.len() {
            let nt = scratch.new_tets[i];
            let nv = arena.tet(nt).verts();
            if let Some(v) = nv.iter().copied().find(|&v| !is_virtual(v)) {
                let cc = grid.cell_of(arena.coord(v));
                if bound.contains_cell(cc) {
                    cell_seed[grid.idx(cc)].store(nt, Ordering::Relaxed);
                }
            }
        }
        Some(())
    })();
    unlock_all(held);
    match result {
        Some(()) => Ins::Committed,
        None => Ins::Deferred,
    }
}

/// Build a per-cell seed-tet table for `grid` by locating every cell centre
/// in the current triangulation. Read-only on the arena (no insertion may run
/// concurrently), so the x-slabs locate in parallel; within a slab the walk
/// hint chains cell-to-cell so each locate is a short hop.
fn seed_cells(
    arena: &Arena,
    grid: &LockGrid,
    min: DVec3,
    max: DVec3,
    hint: u32,
) -> Box<[AtomicU32]> {
    let gdim = grid.gdim;
    let cell_seed: Box<[AtomicU32]> = (0..(gdim as usize).pow(3))
        .map(|_| AtomicU32::new(INF_NONE))
        .collect();
    (0..gdim).into_par_iter().for_each(|cx| {
        let mut seen = rustc_hash::FxHashSet::default();
        let mut h = hint;
        for cy in 0..gdim {
            for cz in 0..gdim {
                let q = min
                    + DVec3::new(
                        (cx as f64 + 0.5) / gdim as f64,
                        (cy as f64 + 0.5) / gdim as f64,
                        (cz as f64 + 0.5) / gdim as f64,
                    ) * (max - min);
                let t = walk_locate(arena, h, q, None, &mut seen)
                    .unwrap_or_else(|| linear_locate(arena, q));
                h = t;
                cell_seed[grid.idx((cx, cy, cz))].store(t, Ordering::Relaxed);
            }
        }
    });
    cell_seed
}

/// Newest alive tet: a good serial-walk hint after parallel work may have
/// killed whatever hint we were holding.
fn newest_alive(arena: &Arena) -> u32 {
    let mut h = arena.len() - 1;
    while !arena.tet(h).is_alive() {
        h -= 1;
    }
    h
}

/// One parallel insertion level: multi-pass locked insertion of the
/// Hilbert-ordered `remaining`, retrying deferrals into the ever-finer mesh,
/// then a serial tail. Returns `(passes, serial_tail_len)` for logging.
fn insert_level(
    arena: &Arena,
    grid: &LockGrid,
    cell_seed: &[AtomicU32],
    mut remaining: Vec<u32>,
    scratch: &mut Scratch,
) -> (u32, usize) {
    let n_threads = rayon::current_num_threads().max(1);
    // Parallel insertion in repeated passes. Each pass inserts the remaining
    // points in contiguous Hilbert chunks (spatially separated threads → low
    // contention); points whose cavity escaped the lock or lost a lock race
    // are retried next pass into a now-finer mesh (smaller cavities). The
    // residual shrinks geometrically; a small tail finishes serially.
    let serial_tail = 4096usize;
    let mut passes = 0u32;
    while remaining.len() > serial_tail && passes < 12 {
        let before = remaining.len();
        let chunk = remaining.len().div_ceil(n_threads).max(1);
        let mut next: Vec<u32> = remaining
            .par_chunks(chunk)
            .flat_map_iter(|pids| {
                let mut scr = Scratch::default();
                let mut deferred = Vec::new();
                for &pid in pids {
                    if matches!(
                        try_insert_locked(arena, grid, cell_seed, pid, &mut scr),
                        Ins::Deferred
                    ) {
                        deferred.push(pid);
                    }
                }
                deferred
            })
            .collect();
        next.sort_unstable(); // restore Hilbert order for next pass's chunks
        remaining = next;
        passes += 1;
        // Keep retrying while passes still meaningfully shrink the residual:
        // a parallel re-attempt is cheaper than a serial insert, so we only bail
        // to the serial tail once a pass clears < 10% of what remained.
        if remaining.len() * 10 > before * 9 {
            break;
        }
    }

    // Serial cleanup of the residual tail (Hilbert order, chained hint).
    let n_deferred = remaining.len();
    remaining.sort_unstable();
    let mut h = newest_alive(arena);
    for pid in remaining {
        h = serial_insert(arena, pid, h, scratch);
    }
    (passes, n_deferred)
}

/// Lock-grid parallel 3D Delaunay. Returns tets as `[v0,v1,v2,v3]` indexing
/// into `points` (caller order).
pub fn delaunay_3d(points: &[glam::Vec3]) -> Vec<[u32; 4]> {
    let n = points.len();
    if n < 4 {
        return Vec::new();
    }
    let dpoints: Vec<DVec3> = points
        .iter()
        .map(|p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect();
    let mut min = dpoints[0];
    let mut max = dpoints[0];
    for p in &dpoints[1..] {
        min = min.min(*p);
        max = max.max(*p);
    }
    let center = (min + max) * 0.5;
    let r = (max - min).length().max(1.0) * 100.0;
    let virtuals = [
        center + DVec3::new(0.0, 0.0, r),
        center + DVec3::new(r * 0.9428, 0.0, -r * 0.3333),
        center + DVec3::new(-r * 0.4714, r * 0.8165, -r * 0.3333),
        center + DVec3::new(-r * 0.4714, -r * 0.8165, -r * 0.3333),
    ];

    let order = hilbert_order(points, min, max);
    let reordered: Vec<DVec3> = order.iter().map(|&i| dpoints[i as usize]).collect();

    let t_arena = std::time::Instant::now();
    let cap = n.saturating_mul(32) + 64;
    let arena = Arena::new(reordered, n as u32, virtuals, cap);
    let arena_secs = t_arena.elapsed().as_secs_f64();
    // Bounding tet (index 0). Virtual vertices use the `INF0..` sentinels so
    // `is_virtual` strips hull tets and `coord` maps them to the appended
    // virtual points.
    arena.alloc([INF0, INF0 + 1, INF0 + 2, INF0 + 3], [INF_NONE; 4]);

    // Hierarchical insertion: a small serial bootstrap, then parallel levels
    // at geometrically increasing density (every level multiplies the point
    // count by 8). Each level uses *uniform* strided subsamples of the
    // Hilbert order (a prefix would densify one region only) and a lock grid
    // sized to the density the level ends at, so cavities span ≲ a cell and
    // the 3×3×3 locked block contains them. This keeps the serial fraction
    // at n/SERIAL_STRIDE instead of the n/8 a single-level bootstrap needs.
    const SERIAL_STRIDE: usize = 512;
    const LEVEL_STRIDES: [usize; 3] = [64, 8, 1];

    let t_boot = std::time::Instant::now();
    let mut scratch = Scratch::default();
    let mut hint = 0u32;
    let mut bootstrap = 0usize;
    for pid in (0..n as u32).step_by(SERIAL_STRIDE) {
        hint = serial_insert(&arena, pid, hint, &mut scratch);
        bootstrap += 1;
    }
    let boot_secs = t_boot.elapsed().as_secs_f64();

    let mut inserted = bootstrap;
    let mut prev_stride = SERIAL_STRIDE;
    let mut level_logs: Vec<String> = Vec::new();
    for stride in LEVEL_STRIDES {
        let pids: Vec<u32> = (0..n as u32)
            .filter(|&p| {
                (p as usize).is_multiple_of(stride) && !(p as usize).is_multiple_of(prev_stride)
            })
            .collect();
        prev_stride = stride;
        if pids.is_empty() {
            continue;
        }
        let t_level = std::time::Instant::now();
        let level_points = pids.len();
        inserted += level_points;
        // ~16 points/cell at the density this level ends at. Smaller cells
        // spread the per-insertion 27-cell locks out so threads contend
        // less; cavity escapes are mopped up by the multi-pass retry rather
        // than by enlarging the locked block.
        let gdim = (((inserted as f64) / 16.0).cbrt().ceil() as u32).max(1);
        let grid = LockGrid::new(min, max, gdim);
        let t_seed = std::time::Instant::now();
        let cell_seed = seed_cells(&arena, &grid, min, max, newest_alive(&arena));
        let seed_secs = t_seed.elapsed().as_secs_f64();
        let (passes, n_deferred) = insert_level(&arena, &grid, &cell_seed, pids, &mut scratch);
        level_logs.push(format!(
            "1/{stride}: {level_points}pts gdim={gdim} seed={seed_secs:.2}s {passes}p tail={n_deferred} {:.2}s",
            t_level.elapsed().as_secs_f64(),
        ));
    }

    log::info!(
        "lockgrid: threads={} arena={} | arena_init={arena_secs:.2}s boot={boot_secs:.2}s({bootstrap}) | {}",
        rayon::current_num_threads(),
        arena.len(),
        level_logs.join(" | "),
    );

    // Compact: alive, all-real tets, remapped to caller indices.
    let len = arena.len();
    (0..len)
        .into_par_iter()
        .filter_map(|i| {
            let tet = arena.tet(i);
            if !tet.is_alive() {
                return None;
            }
            let verts = tet.verts();
            if verts.iter().any(|&v| is_virtual(v)) {
                return None;
            }
            Some([
                order[verts[0] as usize],
                order[verts[1] as usize],
                order[verts[2] as usize],
                order[verts[3] as usize],
            ])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Self-consistency: every non-vertex point must lie outside each tet's
    /// circumsphere (the defining Delaunay property). `queries` limits the
    /// number of points checked against all tets (None = all points).
    fn validate(points: &[glam::Vec3], tets: &[[u32; 4]], queries: Option<usize>) {
        let dp: Vec<DVec3> = points
            .iter()
            .map(|p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();
        let stride = queries.map_or(1, |q| (points.len() / q).max(1));
        for &t in tets {
            let (a, b, c, d) = (
                dp[t[0] as usize],
                dp[t[1] as usize],
                dp[t[2] as usize],
                dp[t[3] as usize],
            );
            let (a, b) = if orient3d_filter(a, b, c, d) > 0.0 {
                (a, b)
            } else {
                (b, a)
            };
            for (i, &q) in dp.iter().enumerate().step_by(stride) {
                if t.contains(&(i as u32)) {
                    continue;
                }
                assert!(
                    insphere_oriented(a, b, c, d, q) >= -1e-6,
                    "Delaunay property violated: point {i} inside tet {t:?}"
                );
            }
        }
    }

    fn random_points(n: usize, seed: u64) -> Vec<glam::Vec3> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                glam::Vec3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                )
            })
            .collect()
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
        validate(&pts, &tets, None);
    }

    #[test]
    fn random_points_satisfy_empty_circumsphere() {
        let pts = random_points(300, 123);
        let tets = delaunay_3d(&pts);
        assert!(!tets.is_empty());
        validate(&pts, &tets, None);
        let mut seen = vec![false; pts.len()];
        for t in &tets {
            for &v in t {
                seen[v as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "every input point is referenced");
    }

    /// At 20k points the hierarchical path engages its parallel levels (the
    /// final level is above the serial-tail threshold). Validates coverage
    /// plus the empty-circumsphere property on a sample of query points.
    #[test]
    fn parallel_levels_satisfy_empty_circumsphere() {
        let pts = random_points(20_000, 7);
        let tets = delaunay_3d(&pts);
        let mut seen = vec![false; pts.len()];
        for t in &tets {
            for &v in t {
                seen[v as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "every input point is referenced");
        validate(&pts, &tets, Some(32));
    }

    /// Manual perf check: `cargo test -p brush-mesh --release -- --ignored
    /// --nocapture bench_lockgrid`.
    #[test]
    #[ignore = "manual perf benchmark"]
    fn bench_lockgrid_2m() {
        let pts = random_points(2_000_000, 99);
        let t = std::time::Instant::now();
        let tets = delaunay_3d(&pts);
        println!(
            "lockgrid 2M: {:.2}s ({} tets)",
            t.elapsed().as_secs_f64(),
            tets.len()
        );
    }

    /// Lockgrid with a stdout logger so the per-phase breakdown prints.
    #[test]
    #[ignore = "manual perf benchmark"]
    fn bench_lockgrid_phases_2m() {
        struct StdoutLog;
        impl log::Log for StdoutLog {
            fn enabled(&self, _: &log::Metadata) -> bool {
                true
            }
            fn log(&self, record: &log::Record) {
                println!("{}", record.args());
            }
            fn flush(&self) {}
        }
        let _ = log::set_logger(Box::leak(Box::new(StdoutLog)));
        log::set_max_level(log::LevelFilter::Info);
        let pts = random_points(2_000_000, 99);
        let t = std::time::Instant::now();
        let tets = delaunay_3d(&pts);
        println!(
            "lockgrid total {:.2}s ({} tets)",
            t.elapsed().as_secs_f64(),
            tets.len()
        );
    }
}
