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
//! makes incremental Bowyer–Watson fast). Bootstrap a seed triangulation
//! serially, then split the remaining Hilbert-ordered points into contiguous
//! per-thread chunks (spatially separated → low lock contention). Each
//! insertion locks the grid cells its cavity touches; on a lock conflict or a
//! cavity that escapes the locked region it defers the point to a final serial
//! pass. `verts` is immutable after creation, so the point-location walk can
//! read geometry freely; only the mutable links/alive need the lock.

use crate::delaunay::{
    INF0, hilbert_order, insphere_oriented as insphere, is_virtual, orient3d_subst,
};
use glam::DVec3;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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
        let tets: Box<[Tet]> = (0..cap).map(|_| Tet::dead()).collect();
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

// Geometric predicates (`orient3d`, `orient3d_subst`, `insphere`) and the
// `is_virtual`/`INF0`/`hilbert_order` helpers are shared with the serial
// `delaunay` module via the `use` above — no duplication.

#[inline]
fn in_circumsphere(arena: &Arena, verts: [u32; 4], p: DVec3) -> bool {
    let a = arena.coord(verts[0]);
    let b = arena.coord(verts[1]);
    let c = arena.coord(verts[2]);
    let d = arena.coord(verts[3]);
    insphere(a, b, c, d, p) < 0.0
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
impl Scratch {
    fn clear(&mut self) {
        self.visited.clear();
        self.stack.clear();
        self.removed.clear();
        self.boundary.clear();
        self.new_tets.clear();
        self.face_owner.clear();
    }
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
    Committed(u32),
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
        let start = cell_seed[grid.idx(c)].load(Ordering::Relaxed);
        if start == INF_NONE || !arena.tet(start).is_alive() {
            return None;
        }
        let loc = walk_locate(arena, start, p, Some(&bound), &mut scratch.visited)?;
        collect_cavity(arena, loc, p, Some(&bound), scratch)?;
        let new_tet = fill_cavity(arena, pid, scratch);
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
        Some(new_tet)
    })();
    unlock_all(held);
    match result {
        Some(t) => Ins::Committed(t),
        None => Ins::Deferred,
    }
}

/// Lock-grid parallel 3D Delaunay. Returns tets as `[v0,v1,v2,v3]` indexing
/// into `points` (caller order).
pub fn delaunay_3d_lockgrid(points: &[glam::Vec3]) -> Vec<[u32; 4]> {
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

    // Bootstrap: insert a serial prefix so every grid cell has a seed tet and
    // cavities are small once we go parallel.
    // Bootstrap with a *uniform* strided subsample (every `stride`-th
    // Hilbert point), not a Hilbert prefix — a prefix would densify only one
    // spatial region and leave the rest coarse, blowing up the parallel
    // cavities. A strided sample coarsely covers the whole volume.
    let t_boot = std::time::Instant::now();
    const STRIDE: usize = 8;
    let boot_pids: Vec<u32> = (0..n as u32).step_by(STRIDE).collect();
    let bootstrap = boot_pids.len();
    let mut scratch = Scratch::default();
    let mut hint = 0u32;
    for &pid in &boot_pids {
        hint = serial_insert(&arena, pid, hint, &mut scratch);
    }
    let boot_secs = t_boot.elapsed().as_secs_f64();

    let t_seed = std::time::Instant::now();
    // ~16 points/cell. Smaller cells spread the per-insertion 27-cell locks
    // out so threads contend less; cavity escapes are mopped up by the
    // multi-pass retry below rather than by enlarging the locked block.
    let gdim = (((n as f64) / 16.0).cbrt().ceil() as u32).max(1);
    let grid = LockGrid::new(min, max, gdim);
    let cell_seed: Box<[AtomicU32]> = (0..(gdim as usize).pow(3))
        .map(|_| AtomicU32::new(INF_NONE))
        .collect();
    // Seed every cell by locating its centre against the bootstrap tri. Start
    // from the bootstrap's last (alive) tet — tet 0 is the now-dead bounding
    // tet — and fall back to a linear scan so every cell gets a live seed.
    {
        let mut h = hint;
        for cx in 0..gdim {
            for cy in 0..gdim {
                for cz in 0..gdim {
                    let q = min
                        + DVec3::new(
                            (cx as f64 + 0.5) / gdim as f64,
                            (cy as f64 + 0.5) / gdim as f64,
                            (cz as f64 + 0.5) / gdim as f64,
                        ) * (max - min);
                    let t = walk_locate(&arena, h, q, None, &mut scratch.visited)
                        .unwrap_or_else(|| linear_locate(&arena, q));
                    h = t;
                    cell_seed[grid.idx((cx, cy, cz))].store(t, Ordering::Relaxed);
                }
            }
        }
    }

    let seed_secs = t_seed.elapsed().as_secs_f64();

    // Parallel insertion in repeated passes. Each pass inserts the remaining
    // points in contiguous Hilbert chunks (spatially separated threads → low
    // contention); points whose cavity escaped the lock or lost a lock race are
    // retried next pass into a now-finer mesh (smaller cavities). The residual
    // shrinks geometrically; a small tail finishes serially.
    let t_par = std::time::Instant::now();
    let n_threads = rayon::current_num_threads().max(1);
    let mut remaining: Vec<u32> = (0..n as u32)
        .filter(|&p| (p as usize) % STRIDE != 0)
        .collect();
    let first_pass = remaining.len();
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
                    if let Ins::Deferred =
                        try_insert_locked(&arena, &grid, &cell_seed, pid, &mut scr)
                    {
                        deferred.push(pid);
                    }
                }
                deferred
            })
            .collect();
        next.sort_unstable(); // restore Hilbert order for next pass's chunks
        remaining = next;
        passes += 1;
        // Keep retrying while passes still meaningfully shrink the residual — a
        // parallel re-attempt is cheaper than a serial insert, so we only bail
        // to the serial tail once a pass clears < 10% of what remained.
        if remaining.len() * 10 > before * 9 {
            break;
        }
    }
    let par_secs = t_par.elapsed().as_secs_f64();

    // Serial cleanup of the residual tail (Hilbert order, chained hint).
    let t_def = std::time::Instant::now();
    let n_deferred = remaining.len();
    remaining.sort_unstable();
    let mut h = 0u32;
    while !arena.tet(h).is_alive() {
        h += 1;
    }
    for pid in remaining {
        h = serial_insert(&arena, pid, h, &mut scratch);
    }

    log::info!(
        "lockgrid: threads={} arena={} | arena_init={:.2}s boot={:.2}s({}) seed={:.2}s par={:.2}s({} passes over {}) tail={:.2}s({})",
        n_threads,
        arena.len(),
        arena_secs,
        boot_secs,
        bootstrap,
        seed_secs,
        par_secs,
        passes,
        first_pass,
        t_def.elapsed().as_secs_f64(),
        n_deferred,
    );

    // Compact: alive, all-real tets, remapped to caller indices.
    let mut out = Vec::new();
    for i in 0..arena.len() {
        let tet = arena.tet(i);
        if !tet.is_alive() {
            continue;
        }
        let verts = tet.verts();
        if verts.iter().any(|&v| is_virtual(v)) {
            continue;
        }
        out.push([
            order[verts[0] as usize],
            order[verts[1] as usize],
            order[verts[2] as usize],
            order[verts[3] as usize],
        ]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delaunay::orient3d_filter as orient3d;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Every non-vertex point must lie outside each tet's circumsphere.
    fn validate(points: &[glam::Vec3], tets: &[[u32; 4]]) {
        let dp: Vec<DVec3> = points
            .iter()
            .map(|p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();
        for &t in tets {
            let (a, b, c, d) = (
                dp[t[0] as usize],
                dp[t[1] as usize],
                dp[t[2] as usize],
                dp[t[3] as usize],
            );
            let (a, b) = if orient3d(a, b, c, d) > 0.0 {
                (a, b)
            } else {
                (b, a)
            };
            for (i, &q) in dp.iter().enumerate() {
                if t.contains(&(i as u32)) {
                    continue;
                }
                assert!(
                    insphere(a, b, c, d, q) >= -1e-6,
                    "Delaunay property violated: point {i} inside tet {t:?}"
                );
            }
        }
    }

    #[test]
    fn lockgrid_random_is_delaunay() {
        let mut rng = StdRng::seed_from_u64(123);
        let pts: Vec<glam::Vec3> = (0..300)
            .map(|_| {
                glam::Vec3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                )
            })
            .collect();
        let tets = delaunay_3d_lockgrid(&pts);
        assert!(!tets.is_empty());
        validate(&pts, &tets);
        let mut seen = vec![false; pts.len()];
        for t in &tets {
            for &v in t {
                seen[v as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "every input point is referenced");
    }
}
