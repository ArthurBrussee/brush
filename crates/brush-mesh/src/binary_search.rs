//! Edge-wise binary search for the iso-surface crossing.
//!
//! Exposed as a state machine driven externally so the caller can plug in
//! its async opacity-evaluator without us needing `AsyncFnMut` plumbing.
//! Usage:
//!
//! ```ignore
//! let mut state = RefineState::new(crossings, &vertex_positions, &vertex_sdf);
//! for _ in 0..N_STEPS {
//!     let midpoints = state.midpoints();
//!     let mid_sdf = evaluate_async(&midpoints).await;
//!     state.step(&mid_sdf);
//! }
//! let refined = state.finish();
//! ```
//!
//! Crossings whose mid-SDF magnitude drops below [`CONVERGE_EPS`] get locked
//! at their current midpoint and dropped from the active set. Subsequent
//! [`RefineState::midpoints`] calls only return the remaining-active set,
//! so the GPU walk (the dominant per-step cost) shrinks as the bisection
//! progresses. [`RefineState::finish`] still returns one position per
//! original crossing — locked-in midpoints come along for the ride.

use glam::Vec3;

use crate::marching_tet::Crossing;

pub const N_STEPS: usize = 8;

/// A crossing whose mid-SDF lies within this tolerance of zero is locked
/// at its current midpoint and skipped on subsequent steps. SDF here is
/// `alpha - iso`, so `1e-2` means the midpoint is within 1% of the
/// iso-surface in alpha space — well below the precision marching tets
/// can exploit downstream, and tight enough not to move any vertex
/// position by a visible amount.
pub const CONVERGE_EPS: f32 = 5e-2;

pub struct RefineState {
    left_pos: Vec<Vec3>,
    right_pos: Vec<Vec3>,
    left_sdf: Vec<f32>,
    right_sdf: Vec<f32>,
    /// Original-crossing indices still being bisected. Shrinks each
    /// step as crossings converge.
    active: Vec<u32>,
}

impl RefineState {
    pub fn new(crossings: &[Crossing], vertex_positions: &[Vec3], vertex_sdf: &[f32]) -> Self {
        let left_pos = crossings
            .iter()
            .map(|c| vertex_positions[c.a as usize])
            .collect();
        let right_pos = crossings
            .iter()
            .map(|c| vertex_positions[c.b as usize])
            .collect();
        let left_sdf = crossings.iter().map(|c| vertex_sdf[c.a as usize]).collect();
        let right_sdf = crossings.iter().map(|c| vertex_sdf[c.b as usize]).collect();
        let active = (0..crossings.len() as u32).collect();
        Self {
            left_pos,
            right_pos,
            left_sdf,
            right_sdf,
            active,
        }
    }

    /// Midpoints of the currently-active crossings, in order. The caller
    /// passes the SDFs evaluated at these midpoints back to [`step`].
    pub fn midpoints(&self) -> Vec<Vec3> {
        self.active
            .iter()
            .map(|&i| (self.left_pos[i as usize] + self.right_pos[i as usize]) * 0.5)
            .collect()
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    pub fn step(&mut self, mid_sdf: &[f32]) {
        assert_eq!(mid_sdf.len(), self.active.len());
        let mut new_active = Vec::with_capacity(self.active.len());
        for (j, &i) in self.active.iter().enumerate() {
            let idx = i as usize;
            let mid_pos = (self.left_pos[idx] + self.right_pos[idx]) * 0.5;
            let sdf = mid_sdf[j];
            if sdf.abs() < CONVERGE_EPS {
                // Converged — lock both endpoints at the midpoint so
                // `finish` returns this position, and drop from active.
                self.left_pos[idx] = mid_pos;
                self.right_pos[idx] = mid_pos;
            } else if (sdf < 0.0 && self.left_sdf[idx] < 0.0)
                || (sdf > 0.0 && self.left_sdf[idx] > 0.0)
            {
                self.left_pos[idx] = mid_pos;
                self.left_sdf[idx] = sdf;
                new_active.push(i);
            } else {
                self.right_pos[idx] = mid_pos;
                self.right_sdf[idx] = sdf;
                new_active.push(i);
            }
        }
        self.active = new_active;
    }

    pub fn finish(self) -> Vec<Vec3> {
        (0..self.left_pos.len())
            .map(|i| (self.left_pos[i] + self.right_pos[i]) * 0.5)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Converges to the analytic zero of a linear SDF: f(x) = x.
    /// Uses 8 steps (independent of the production `N_STEPS`) to keep
    /// the convergence-tolerance assertion meaningful.
    #[test]
    fn linear_sdf_converges() {
        let positions = vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let sdf = vec![-1.0, 1.0];
        let crossings = vec![Crossing { a: 0, b: 1 }];
        let mut state = RefineState::new(&crossings, &positions, &sdf);
        for _ in 0..8 {
            let mids = state.midpoints();
            let mid_sdf: Vec<f32> = mids.iter().map(|p| p.x).collect();
            state.step(&mid_sdf);
        }
        let refined = state.finish();
        assert_eq!(refined.len(), 1);
        assert!(refined[0].x.abs() < 0.01);
    }

    /// Offset zero so the first midpoint is *not* exactly at the
    /// crossing — exercises the actual bisection branch (left/right
    /// shrink) as well as the convergence test that fires once we get
    /// close enough.
    #[test]
    fn offset_linear_sdf_converges() {
        // Zero at x = 0.3. Bracket [-1, 1].
        let positions = vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let sdf = vec![-1.3, 0.7];
        let crossings = vec![Crossing { a: 0, b: 1 }];
        let mut state = RefineState::new(&crossings, &positions, &sdf);
        for _ in 0..16 {
            let mids = state.midpoints();
            if mids.is_empty() {
                break;
            }
            let mid_sdf: Vec<f32> = mids.iter().map(|p| p.x - 0.3).collect();
            state.step(&mid_sdf);
        }
        let refined = state.finish();
        assert_eq!(refined.len(), 1);
        // Position precision is bounded by `CONVERGE_EPS` (since the
        // SDF for f(x)=x equals the position offset from the zero).
        assert!(
            (refined[0].x - 0.3).abs() < CONVERGE_EPS,
            "expected x ≈ 0.3 within CONVERGE_EPS, got {}",
            refined[0].x
        );
    }
}
