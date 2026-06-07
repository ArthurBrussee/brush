//! Greedy view-subset selection for mesh extraction.
//!
//! With large datasets (1k+ views), running the integrate pass over every
//! view is wasteful — most views are spatially redundant. This module
//! picks a subset that fills per-splat *directional coverage slots* up
//! to the requested fraction.
//!
//! ## Slots
//!
//! Each seed splat owns `K` saturating scalar slots in `[0, 1]`. Each
//! slot has a fixed reference direction on the unit sphere (Fibonacci
//! sphere — `K = 24` directions, ~42° between nearest neighbours).
//! When view `v` sees splat `s` from unit ray `view_dir = (cam − s)/d`,
//! every slot `k` gets:
//!
//! ```text
//!     contrib_v,s,k = max(0, dot(slot_dir_k, view_dir))² / d²
//! ```
//!
//! Squared cosine is sharper than linear — each view only contributes
//! meaningfully to the 2-3 slots it's most aligned with, not the whole
//! hemisphere. Slots saturate at 1.
//!
//! ## Coverage target
//!
//! The all-views saturated mass — the sum of slot values when *every*
//! view is selected (with slots clamped at 1) — is the natural
//! ceiling. `coverage ∈ [0, 1]` is interpreted as a fraction of that:
//! `1.0` means "fill until you reach all-views saturation", `0.5`
//! means "halfway there". This is intentionally non-linear in view
//! count: the greedy's first picks fill mass fast (each contributes
//! to many empty slots), the tail-end picks add only fractions of
//! slots. So a heavily-redundant 10k-view dataset drops to far below
//! 10% kept at `coverage = 0.5`, while a 292-view orbit (already
//! near-optimal) only loses ~25% of views at the same target. The
//! point of the param is to express "how much quality do I want" in
//! a dataset-independent way.
//!
//! ## Why this captures angular diversity
//!
//! A single view fills the 2-3 slots aligned with its viewing
//! direction — but contributes ~zero to slots on the opposite side of
//! the sphere or perpendicular to the view ray. To saturate all 24
//! slots of a splat, you need views spread around the sphere. Distance
//! `1/d²` down-weights distant views smoothly, so close angularly-
//! diverse views are preferred over far ones.
//!
//! ## Algorithm
//!
//! 1. Per (view, seed) precompute the in-frustum entry `(view_dir,
//!    1/d²)`. Done in parallel.
//! 2. Compute per-(seed, slot) all-views saturated value. Sum =
//!    target denominator.
//! 3. Greedy: while filled mass < `coverage × target`, pick the view
//!    maximising `Σ_(s,k) (min(slot+contrib, 1) − slot)`; update
//!    slots. Gain scan is parallel; updates are serial.

use brush_render::camera::Camera;
use glam::{Mat4, UVec2, Vec3, Vec4Swizzles};
use rayon::prelude::*;

/// `K` slot directions per splat, evenly spread on the unit sphere via
/// the Fibonacci-spiral construction. `K = 24` gives ~42° between
/// adjacent slots, fine enough that opposite views fill disjoint slots
/// but coarse enough that the greedy can fill all of them in a few
/// dozen well-chosen picks.
const K: usize = 24;

const SLOT_DIRS: [Vec3; K] = fibonacci_sphere();

const fn fibonacci_sphere() -> [Vec3; K] {
    let mut out = [Vec3::ZERO; K];
    // Golden angle in radians; the Vec3 array has to be const so we
    // bake in a precomputed table of (x, y, z) for each slot rather
    // than running f32 ops in a const fn (no support yet).
    //
    // Generated with:
    //   phi = (1 + sqrt(5)) / 2
    //   for i in 0..K:
    //     y = 1 - 2*(i + 0.5)/K
    //     r = sqrt(1 - y*y)
    //     theta = 2*pi*i/phi
    //     x, z = r*cos(theta), r*sin(theta)
    #[allow(clippy::excessive_precision)]
    let table = [
        [0.427_137_3, 0.958_333_3, 0.209_619_3],
        [-0.854_162_2, 0.875, 0.277_750_2],
        [0.717_493_5, 0.791_666_6, -0.610_247_4],
        [0.169_444_1, 0.708_333_3, 0.684_997_1],
        [-0.830_180_2, 0.625, -0.300_709_2],
        [0.957_172_7, 0.541_666_6, -0.207_107_1],
        [-0.512_538_1, 0.458_333_3, 0.725_981_3],
        [-0.305_407_5, 0.375, -0.875_303_3],
        [0.887_487_5, 0.291_666_6, 0.354_901_0],
        [-0.819_982_9, 0.208_333_3, 0.533_104_9],
        [0.217_176_3, 0.125, -0.968_411_6],
        [0.590_382_9, 0.041_666_6, 0.806_082_0],
        [-0.979_561_1, -0.041_666_7, -0.196_763_4],
        [0.718_071_3, -0.125, -0.684_660_4],
        [-0.051_321_0, -0.208_333_3, 0.976_700_1],
        [-0.669_351_6, -0.291_666_6, -0.683_112_9],
        [0.958_194_0, -0.375, 0.053_731_8],
        [-0.751_622_6, -0.458_333_3, 0.474_031_0],
        [0.131_298_8, -0.541_666_6, -0.830_316_0],
        [0.478_351_5, -0.625, 0.616_677_8],
        [-0.699_878_2, -0.708_333_3, -0.090_147_3],
        [0.544_150_4, -0.791_666_6, -0.276_404_1],
        [-0.137_671_9, -0.875, 0.464_222_0],
        [-0.270_681_7, -0.958_333_3, -0.090_981_1],
    ];
    let mut i = 0;
    while i < K {
        out[i] = Vec3::new(table[i][0], table[i][1], table[i][2]);
        i += 1;
    }
    out
}

/// Per-view visibility entry: which seed, the unit ray from seed to
/// camera, and `1/d²`.
#[derive(Copy, Clone)]
struct VisEntry {
    seed_idx: u32,
    view_dir: Vec3,
    inv_d2: f32,
}

/// Pick views by greedy slot-fill until the running per-splat
/// directional mass reaches `coverage × all-views mass`. The first
/// pick is the single most-useful view; each subsequent pick adds
/// the most marginal gain given the running slot state. Output
/// indices are sorted ascending.
///
/// The mapping from `coverage` to view count is intentionally
/// non-linear and dataset-dependent: heavily-redundant captures drop
/// far more views than near-optimal ones at the same target.
///
/// - `coverage = 1.0` → no subsetting (returns all views verbatim).
/// - `coverage = 0.8` → typically a few percent off all-views mass;
///   the default for the CLI.
/// - `coverage = 0.0` → just the single best view.
pub fn select_views_by_coverage(
    seed_points: &[Vec3],
    views: &[(Camera, UVec2)],
    coverage: f32,
) -> Vec<usize> {
    if views.is_empty() {
        return Vec::new();
    }
    if coverage >= 1.0 {
        // Default — no subsetting. Skip the greedy entirely.
        return (0..views.len()).collect();
    }
    let coverage = coverage.clamp(0.0, 1.0);
    let n_seed = seed_points.len();
    if n_seed == 0 {
        return (0..views.len()).collect();
    }

    // 1. Per-view visibility table, built in parallel (one view per
    // task; each task allocates its own list of in-frustum entries).
    let per_view: Vec<Vec<VisEntry>> = views
        .par_iter()
        .map(|(cam, sz)| {
            let pinhole = cam.build_pinhole_params(*sz);
            let w2c = Mat4::from(cam.world_to_local());
            let cam_pos = cam.position;
            let w = sz.x as f32;
            let h = sz.y as f32;
            let mut entries: Vec<VisEntry> = Vec::with_capacity(n_seed / 4);
            for (i, p) in seed_points.iter().enumerate() {
                let p_cam = (w2c * p.extend(1.0)).xyz();
                if p_cam.z <= 0.1 {
                    continue;
                }
                let px = p_cam.x / p_cam.z * pinhole.fx + pinhole.cx;
                let py = p_cam.y / p_cam.z * pinhole.fy + pinhole.cy;
                if px < 0.0 || px >= w || py < 0.0 || py >= h {
                    continue;
                }
                let delta = cam_pos - *p;
                let d2 = delta.length_squared().max(1e-6);
                let view_dir = delta * d2.sqrt().recip();
                entries.push(VisEntry {
                    seed_idx: i as u32,
                    view_dir,
                    inv_d2: 1.0 / d2,
                });
            }
            entries
        })
        .collect();

    // 2. All-views saturated mass per (seed, slot) — the upper bound
    // the coverage target is expressed against.
    let mut all_views_slot: Vec<f32> = vec![0.0; n_seed * K];
    for entries in &per_view {
        for e in entries {
            let s_off = e.seed_idx as usize * K;
            for k in 0..K {
                let cos = SLOT_DIRS[k].dot(e.view_dir).max(0.0);
                let c = cos * cos * e.inv_d2;
                let v = &mut all_views_slot[s_off + k];
                *v = (*v + c).min(1.0);
            }
        }
    }
    let total_mass: f32 = all_views_slot.iter().sum();
    if total_mass <= 0.0 {
        return Vec::new();
    }
    let target_mass = coverage * total_mass;

    // 3. Greedy. Score gain in parallel across views per iteration;
    // pick the best, update slots, repeat until filled mass hits the
    // target (or remaining gain underflows to zero).
    let mut slots: Vec<f32> = vec![0.0; n_seed * K];
    let mut taken = vec![false; views.len()];
    let mut selected: Vec<usize> = Vec::new();
    let mut filled_mass: f32 = 0.0;

    while filled_mass < target_mass {
        let best = per_view
            .par_iter()
            .enumerate()
            .filter(|(vi, _)| !taken[*vi])
            .map(|(vi, entries)| {
                let mut gain = 0.0f32;
                for e in entries {
                    let s_off = e.seed_idx as usize * K;
                    for k in 0..K {
                        let cos = SLOT_DIRS[k].dot(e.view_dir).max(0.0);
                        let c = cos * cos * e.inv_d2;
                        let cur = slots[s_off + k];
                        let new = (cur + c).min(1.0);
                        gain += new - cur;
                    }
                }
                (vi, gain)
            })
            .reduce(
                || (usize::MAX, 0.0f32),
                |a, b| if b.1 > a.1 { b } else { a },
            );
        let (best_vi, best_gain) = best;
        if best_vi == usize::MAX || best_gain <= 1e-6 {
            break;
        }
        for e in &per_view[best_vi] {
            let s_off = e.seed_idx as usize * K;
            for k in 0..K {
                let cos = SLOT_DIRS[k].dot(e.view_dir).max(0.0);
                let c = cos * cos * e.inv_d2;
                let cur = slots[s_off + k];
                let new = (cur + c).min(1.0);
                filled_mass += new - cur;
                slots[s_off + k] = new;
            }
        }
        taken[best_vi] = true;
        selected.push(best_vi);
    }

    selected.sort_unstable();
    selected
}
