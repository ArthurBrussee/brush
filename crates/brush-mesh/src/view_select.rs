//! Greedy view-subset selection for mesh extraction.
//!
//! With large datasets (1k+ views), running the integrate pass over every
//! view is wasteful — most views are spatially redundant. This module
//! picks a subset that covers the scene's seed points well, where the
//! caller controls how aggressive the trimming is via a `[0, 1]`
//! coverage parameter.
//!
//! ## Per-direction coverage
//!
//! Naive "seed is in this view's frustum" is not enough: in a 360
//! dataset of a single object, every view sees every seed point from
//! some direction, yet a *single* view only captures the front-side
//! information. Mesh extraction wants the seed-point's surface to be
//! seen from a *diverse range of angles*.
//!
//! So we partition viewing directions into 26 bins (3³ − 1 — each
//! axis classified as negative / near-zero / positive). The element
//! to cover is `(seed_idx, dir_bin)`, not just `seed_idx`. A 360 spin
//! around an object naturally needs ~26× more (seed, bin) pairs filled
//! than a single-angle capture, so it scales to many views; a single
//! viewpoint caps at one pair per seed.
//!
//! ## Distance discount
//!
//! A view that sees a seed from far away is less useful than a close
//! one (lower angular resolution, more occlusion ambiguity). Each
//! (view, seed) pair gets a weight `1 / distance²`, normalised so the
//! closest view of each seed gets weight 1. The greedy maximises sum
//! of weights of newly-covered (seed, bin) pairs — so far views still
//! contribute, just less.
//!
//! ## Algorithm
//!
//! Weighted greedy max-coverage:
//! 1. For each view × seed in-frustum pair: record `(bin, weight)`.
//! 2. Loop: pick the view whose currently-uncovered (seed, bin) pairs
//!    sum to the most weight; mark them covered. Stop when we've
//!    captured `coverage * max_reachable_weight`.
//!
//! Cost: `O(n_views * n_seed)` for the visibility table plus
//! `O(n_selected * n_views * n_seed)` for the greedy loop. On bonsai
//! (292 views × 10k seeds, ~200 picked) this is a couple of seconds
//! on CPU.

use brush_render::camera::Camera;
use glam::{Mat4, UVec2, Vec3, Vec4Swizzles};

/// Number of viewing-direction bins, 3³ − 1 sign-classified buckets.
/// The all-zero bucket (axes all near zero) is unreachable for unit
/// vectors, so 26 are actually populated.
const N_BINS: usize = 27;

/// Classify a non-zero direction into one of [`N_BINS`] buckets by
/// sign of each axis (with a small dead zone around zero so a
/// near-axis-aligned ray bins consistently).
fn dir_bin(d: Vec3) -> usize {
    const DEAD: f32 = 0.3;
    let bx = if d.x < -DEAD {
        0
    } else if d.x > DEAD {
        2
    } else {
        1
    };
    let by = if d.y < -DEAD {
        0
    } else if d.y > DEAD {
        2
    } else {
        1
    };
    let bz = if d.z < -DEAD {
        0
    } else if d.z > DEAD {
        2
    } else {
        1
    };
    bx * 9 + by * 3 + bz
}

/// Pick a view subset that covers the seed points at the requested level.
///
/// `coverage` is in `[0, 1]`:
/// - `1.0` (or higher) means "cover everything any view can cover" —
///   typically returns ~70-80% of inputs (drops only redundant views).
/// - `0.5` cuts the subset to ~half-coverage of seeds, dropping more.
/// - `0.0` returns the single best view.
///
/// The output indices are sorted ascending so the caller can preserve
/// the original ordering when picking out the kept views.
pub fn select_views_by_coverage(
    seed_points: &[Vec3],
    views: &[(Camera, UVec2)],
    coverage: f32,
) -> Vec<usize> {
    if views.is_empty() {
        return Vec::new();
    }
    if coverage >= 1.0 && views.len() <= 1 {
        return (0..views.len()).collect();
    }
    let coverage = coverage.clamp(0.0, 1.0);
    let n_seed = seed_points.len();
    if n_seed == 0 {
        return (0..views.len()).collect();
    }

    // Per-(view, seed) record: which bin (or `u8::MAX` for "not in
    // frustum") and the inverse-square distance weight. Storing dense
    // keeps the greedy loop's gain calculation a flat seed-major scan
    // per view.
    let mut bin_per_seed: Vec<Vec<u8>> = Vec::with_capacity(views.len());
    let mut wt_per_seed: Vec<Vec<f32>> = Vec::with_capacity(views.len());
    for (cam, sz) in views {
        let pinhole = cam.build_pinhole_params(*sz);
        let w2c = Mat4::from(cam.world_to_local());
        let cam_pos = cam.position;
        let w = sz.x as f32;
        let h = sz.y as f32;
        let mut bins = vec![u8::MAX; n_seed];
        let mut wts = vec![0.0f32; n_seed];
        for (i, p) in seed_points.iter().enumerate() {
            let p_cam = (w2c * p.extend(1.0)).xyz();
            if !(p_cam.z > 0.1) {
                continue;
            }
            let px = p_cam.x / p_cam.z * pinhole.fx + pinhole.cx;
            let py = p_cam.y / p_cam.z * pinhole.fy + pinhole.cy;
            if px < 0.0 || px >= w || py < 0.0 || py >= h {
                continue;
            }
            let delta = cam_pos - *p;
            bins[i] = dir_bin(delta) as u8;
            let d2 = delta.length_squared().max(1e-6);
            wts[i] = 1.0 / d2;
        }
        bin_per_seed.push(bins);
        wt_per_seed.push(wts);
    }

    // Normalise each seed's weights so the closest view of that seed
    // gets weight 1. Without normalisation, scenes with widely varying
    // depths bias toward whichever seed is closest to *some* camera,
    // which can starve the rest of the scene.
    for seed_i in 0..n_seed {
        let mut max_w = 0.0f32;
        for vi in 0..views.len() {
            if bin_per_seed[vi][seed_i] != u8::MAX && wt_per_seed[vi][seed_i] > max_w {
                max_w = wt_per_seed[vi][seed_i];
            }
        }
        if max_w > 0.0 {
            let inv = 1.0 / max_w;
            for vi in 0..views.len() {
                wt_per_seed[vi][seed_i] *= inv;
            }
        }
    }

    // For each (seed, bin) pair: track the max weight any view
    // contributes if we were to pick it. `max_reachable_weight` is the
    // sum of these maxes — the cap our coverage target multiplies.
    let mut max_w_per_pair: Vec<f32> = vec![0.0f32; n_seed * N_BINS];
    for vi in 0..views.len() {
        for s in 0..n_seed {
            let bin = bin_per_seed[vi][s];
            if bin != u8::MAX {
                let pair_idx = s * N_BINS + bin as usize;
                let w = wt_per_seed[vi][s];
                if w > max_w_per_pair[pair_idx] {
                    max_w_per_pair[pair_idx] = w;
                }
            }
        }
    }
    let max_reachable: f32 = max_w_per_pair.iter().sum();
    let target = max_reachable * coverage;

    // Greedy: each iteration picks the view whose currently-uncovered
    // (seed, bin) pairs sum to the highest weight, and marks them
    // covered. We track per-pair "achieved weight" — once a pair is
    // covered by the first view that hits it (so the weight is fixed
    // to that view's contribution), later views can't improve it.
    let mut achieved: Vec<f32> = vec![0.0f32; n_seed * N_BINS];
    let mut taken = vec![false; views.len()];
    let mut selected = Vec::new();
    let mut current: f32 = 0.0;

    while current < target {
        let mut best_vi = usize::MAX;
        let mut best_gain = 0.0f32;
        for vi in 0..views.len() {
            if taken[vi] {
                continue;
            }
            let mut gain = 0.0f32;
            for s in 0..n_seed {
                let bin = bin_per_seed[vi][s];
                if bin == u8::MAX {
                    continue;
                }
                let pair_idx = s * N_BINS + bin as usize;
                let w = wt_per_seed[vi][s];
                if w > achieved[pair_idx] {
                    gain += w - achieved[pair_idx];
                }
            }
            if gain > best_gain {
                best_gain = gain;
                best_vi = vi;
            }
        }
        if best_vi == usize::MAX || best_gain <= 1e-6 {
            break;
        }
        // Commit: update achieved with the picked view's contributions.
        for s in 0..n_seed {
            let bin = bin_per_seed[best_vi][s];
            if bin == u8::MAX {
                continue;
            }
            let pair_idx = s * N_BINS + bin as usize;
            let w = wt_per_seed[best_vi][s];
            if w > achieved[pair_idx] {
                achieved[pair_idx] = w;
            }
        }
        current += best_gain;
        taken[best_vi] = true;
        selected.push(best_vi);
    }

    selected.sort_unstable();
    selected
}
