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

use glam::Vec3;

use crate::marching_tet::Crossing;

pub const N_STEPS: usize = 4;

pub struct RefineState {
    left_pos: Vec<Vec3>,
    right_pos: Vec<Vec3>,
    left_sdf: Vec<f32>,
    right_sdf: Vec<f32>,
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
        Self {
            left_pos,
            right_pos,
            left_sdf,
            right_sdf,
        }
    }

    pub fn midpoints(&self) -> Vec<Vec3> {
        (0..self.left_pos.len())
            .map(|i| (self.left_pos[i] + self.right_pos[i]) * 0.5)
            .collect()
    }

    pub fn step(&mut self, mid_sdf: &[f32]) {
        let mids = self.midpoints();
        for i in 0..self.left_pos.len() {
            if (mid_sdf[i] < 0.0 && self.left_sdf[i] < 0.0)
                || (mid_sdf[i] > 0.0 && self.left_sdf[i] > 0.0)
            {
                self.left_pos[i] = mids[i];
                self.left_sdf[i] = mid_sdf[i];
            } else {
                self.right_pos[i] = mids[i];
                self.right_sdf[i] = mid_sdf[i];
            }
        }
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
    #[test]
    fn linear_sdf_converges() {
        let positions = vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let sdf = vec![-1.0, 1.0];
        let crossings = vec![Crossing { a: 0, b: 1 }];
        let mut state = RefineState::new(&crossings, &positions, &sdf);
        for _ in 0..N_STEPS {
            let mids = state.midpoints();
            let mid_sdf: Vec<f32> = mids.iter().map(|p| p.x).collect();
            state.step(&mid_sdf);
        }
        let refined = state.finish();
        assert_eq!(refined.len(), 1);
        assert!(refined[0].x.abs() < 0.01);
    }
}
