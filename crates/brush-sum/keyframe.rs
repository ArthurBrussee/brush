// =============================================================================
// crates/brush-sfm/src/keyframe.rs
// =============================================================================
// PIPELINE STAGE: Telemetry Fusion → [Keyframes]
//
// Implements EXACTLY the condition box shown in the diagram:
//
//   if distance > 2 m
//   OR yaw change > 8°
//   OR pitch change > 5°
//   OR time since last keyframe > 1 s
//       → keyframe
//
// The keyframe list feeds TWO downstream nodes:
//   1. [Pose Priors]  — GPS ENU positions and rotation priors
//   2. [Match pruning] — image pair candidates
//   (the diagram shows both paths emerging from [Keyframes])
// =============================================================================

use crate::telemetry::TelemetryRecord;

// ─────────────────────────────────────────────────────────────────────────────
// THRESHOLDS — from the diagram's condition box (exact values)
// ─────────────────────────────────────────────────────────────────────────────

/// Selection thresholds matching the diagram exactly.
#[derive(Debug, Clone)]
pub struct KeyframeThresholds {
    /// distance > 2 m  (ENU ground distance)
    pub min_dist_m:         f64,
    /// yaw change > 8°
    pub min_yaw_delta_deg:  f64,
    /// pitch change > 5°
    pub min_pitch_delta_deg: f64,
    /// time since last keyframe > 1 s
    pub max_time_gap_s:     f64,
    /// Quality gate: reject frames with speed above this (blur avoidance)
    pub max_speed_ms:       f64,
}

impl Default for KeyframeThresholds {
    fn default() -> Self {
        Self {
            min_dist_m:          2.0,   // diagram: distance > 2 m
            min_yaw_delta_deg:   8.0,   // diagram: yaw change > 8°
            min_pitch_delta_deg: 5.0,   // diagram: pitch change > 5°
            max_time_gap_s:      1.0,   // diagram: time since last keyframe > 1 s
            max_speed_ms:        8.0,   // quality gate (not in diagram, practical add)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SELECTION REASON  (for debugging / logging)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SelectionReason {
    FirstFrame,
    Distance  { dist_m: f64 },
    YawChange { delta_deg: f64 },
    PitchChange { delta_deg: f64 },
    TimeGap   { elapsed_s: f64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// KEYFRAME  — one selected frame + its pose prior
// ─────────────────────────────────────────────────────────────────────────────

/// A selected keyframe. Carries both the telemetry record (for pose priors)
/// and the frame index into the video (for image extraction).
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Index into the original TelemetryRecord Vec
    pub record_idx:  usize,
    /// Frame index in the source MP4
    pub frame_idx:   usize,
    /// Presentation timestamp in seconds
    pub timestamp_s: f64,
    /// Why this frame was selected (diagnostics)
    pub reason:      SelectionReason,
    /// GPS ENU position [E, N, U] metres — None if no GPS
    pub enu_m:       Option<[f64; 3]>,
    /// Rotation prior matrix (Mode C only) — None otherwise
    pub r_prior:     Option<[[f64; 3]; 3]>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SELECTOR
// ─────────────────────────────────────────────────────────────────────────────

pub struct KeyframeSelector {
    thresholds: KeyframeThresholds,
}

impl KeyframeSelector {
    pub fn new(thresholds: KeyframeThresholds) -> Self {
        Self { thresholds }
    }

    /// Select keyframes from fused telemetry records.
    ///
    /// Applies the four conditions from the diagram in order:
    ///   distance > 2 m  →  selected
    ///   yaw > 8°        →  selected
    ///   pitch > 5°      →  selected
    ///   time > 1 s      →  selected
    ///   none match      →  skipped
    ///
    /// Also applies implicit quality gate: skip high-speed/blurry frames.
    pub fn select(&self, records: &[TelemetryRecord]) -> Vec<Keyframe> {
        if records.is_empty() { return vec![]; }

        let mut keyframes: Vec<Keyframe> = Vec::new();

        // Always include the first record
        keyframes.push(self.make_keyframe(0, records, SelectionReason::FirstFrame));

        let mut last_yaw   = records[0].yaw_deg;
        let mut last_pitch = records[0].pitch_deg;
        let mut last_ts    = records[0].timestamp_s;
        let mut last_enu   = records[0].enu_m.unwrap_or([0.0, 0.0, 0.0]);

        for (i, rec) in records.iter().enumerate().skip(1) {
            // Quality gate: skip blurry/shaky frames before any selection check
            if rec.speed_ms > self.thresholds.max_speed_ms {
                continue;
            }

            // Compute deltas
            let cur_enu  = rec.enu_m.unwrap_or([0.0, 0.0, 0.0]);
            let dist_m   = ground_dist(&cur_enu, &last_enu);
            let yaw_d    = angular_diff(rec.yaw_deg,   last_yaw).abs();
            let pitch_d  = angular_diff(rec.pitch_deg, last_pitch).abs();
            let elapsed  = rec.timestamp_s - last_ts;

            // Apply the four conditions from the diagram
            let reason: Option<SelectionReason> = if dist_m > self.thresholds.min_dist_m {
                Some(SelectionReason::Distance { dist_m })
            } else if yaw_d > self.thresholds.min_yaw_delta_deg {
                Some(SelectionReason::YawChange { delta_deg: yaw_d })
            } else if pitch_d > self.thresholds.min_pitch_delta_deg {
                Some(SelectionReason::PitchChange { delta_deg: pitch_d })
            } else if elapsed > self.thresholds.max_time_gap_s {
                Some(SelectionReason::TimeGap { elapsed_s: elapsed })
            } else {
                None
            };

            if let Some(r) = reason {
                keyframes.push(self.make_keyframe(i, records, r));
                last_yaw   = rec.yaw_deg;
                last_pitch = rec.pitch_deg;
                last_ts    = rec.timestamp_s;
                last_enu   = cur_enu;
            }
        }

        tracing::info!(
            "KeyframeSelector: {} keyframes from {} records ({:.1}% reduction)",
            keyframes.len(), records.len(),
            (1.0 - keyframes.len() as f64 / records.len() as f64) * 100.0
        );
        keyframes
    }

    fn make_keyframe(
        &self, idx: usize, records: &[TelemetryRecord], reason: SelectionReason
    ) -> Keyframe {
        let rec = &records[idx];
        Keyframe {
            record_idx:  idx,
            frame_idx:   rec.frame_idx,
            timestamp_s: rec.timestamp_s,
            reason,
            enu_m:       rec.enu_m,
            r_prior:     rec.r_prior,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// POSE PRIORS  (diagram: Keyframes → Pose Priors AND loop-back from Keyframes)
//
// The diagram shows Pose Priors receiving input from both Keyframes
// AND from the loop-back arrow at the top. That loop-back carries the
// GPS ENU translation that primes RANSAC with a near-correct pose.
// ─────────────────────────────────────────────────────────────────────────────

/// Pose prior for a keyframe pair — used to constrain RANSAC.
#[derive(Debug, Clone)]
pub struct PosePrior {
    pub frame_a: usize,           // keyframe index in the Keyframe Vec
    pub frame_b: usize,
    /// Relative ENU translation (B.enu - A.enu), unit vector
    pub t_direction: [f64; 3],
    /// Scale: Euclidean distance between the two GPS fixes in metres
    pub baseline_m: f64,
    /// Rotation prior for frame B (from yaw + gimbal pitch), if available
    pub r_prior_b: Option<[[f64; 3]; 3]>,
}

/// Build pose priors for all consecutive keyframe pairs.
/// Feeds both [Match pruning] and [Incremental SfM] in the diagram.
pub fn build_pose_priors(keyframes: &[Keyframe]) -> Vec<PosePrior> {
    let mut priors = Vec::new();
    for i in 0..keyframes.len().saturating_sub(1) {
        let a = &keyframes[i];
        let b = &keyframes[i + 1];
        if let (Some(ea), Some(eb)) = (a.enu_m, b.enu_m) {
            let dx = eb[0] - ea[0];
            let dy = eb[1] - ea[1];
            let dz = eb[2] - ea[2];
            let norm = (dx*dx + dy*dy + dz*dz).sqrt().max(1e-9);
            priors.push(PosePrior {
                frame_a:     i,
                frame_b:     i + 1,
                t_direction: [dx/norm, dy/norm, dz/norm],
                baseline_m:  norm,
                r_prior_b:   b.r_prior,
            });
        }
    }
    priors
}

// ─────────────────────────────────────────────────────────────────────────────
// MATCH PRUNING  (diagram: Pose Priors → Match pruning)
//
// Uses GPS epipolar direction to filter correspondences before RANSAC.
// Eliminates matches that are geometrically inconsistent with the known
// translation direction, reducing RANSAC iterations from Mode A (2000)
// to Mode C (100).
// ─────────────────────────────────────────────────────────────────────────────

/// Prune a list of 2D correspondences using the GPS epipolar direction.
/// Returns a boolean mask (true = keep).
///
/// The GPS direction gives us the approximate epipole in both images.
/// Matches whose optical flow is roughly orthogonal to the epipolar
/// direction are outliers — reject them.
pub fn gps_epipolar_prune(
    pts1:       &[[f32; 2]],
    pts2:       &[[f32; 2]],
    fx: f64, fy: f64, cx: f64, cy: f64,
    prior:      &PosePrior,
) -> Vec<bool> {
    let t = prior.t_direction;
    let t_norm = (t[0]*t[0] + t[1]*t[1] + t[2]*t[2]).sqrt();
    if t_norm < 1e-9 {
        return vec![true; pts1.len()];
    }
    // Map GPS translation direction to approximate epipole in image
    let tx = t[0] / t_norm;
    let ty = t[2] / t_norm;   // ENU→camera frame rough mapping
    let tz = t[1] / t_norm;
    let ex = fx * tx / tz.max(1e-6) + cx;
    let ey = fy * ty / tz.max(1e-6) + cy;

    pts1.iter().zip(pts2.iter()).map(|(p1, p2)| {
        let flow     = [p2[0] - p1[0], p2[1] - p1[1]];
        let ep_dir   = [p2[0] as f64 - ex, p2[1] as f64 - ey];
        let dot      = flow[0] as f64 * ep_dir[0] + flow[1] as f64 * ep_dir[1];
        let mag      = ((flow[0]*flow[0]+flow[1]*flow[1]) as f64).sqrt()
                     * (ep_dir[0]*ep_dir[0]+ep_dir[1]*ep_dir[1]).sqrt() + 1e-9;
        (dot / mag) > -0.4     // allow broad ±112° cone — GPS is approximate
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// GEOMETRY HELPERS
// ─────────────────────────────────────────────────────────────────────────────

fn ground_dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx*dx + dy*dy).sqrt()    // 2D ground distance only
}

fn angular_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d >  180.0 { d -= 360.0; }
    while d <= -180.0 { d += 360.0; }
    d
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::TelemetryRecord;

    fn make_rec(i: usize, lat: f64, lon: f64,
                yaw: f64, pitch: f64, speed: f64, ts: f64) -> TelemetryRecord {
        TelemetryRecord {
            frame_idx: i, timestamp_s: ts,
            lat, lon, alt_m: 80.0,
            yaw_deg: yaw, pitch_deg: pitch, roll_deg: 0.0,
            vel_east_ms: speed, vel_north_ms: 0.0, vel_up_ms: 0.0,
            speed_ms: speed,
            enu_m: Some([i as f64 * 1.0, 0.0, 0.0]),  // 1 m east per record
            r_prior: None,
        }
    }

    #[test]
    fn test_first_frame_always_selected() {
        let recs = vec![make_rec(0, 18.542, 73.727, 0.0, -90.0, 1.0, 0.0)];
        let sel = KeyframeSelector::new(KeyframeThresholds::default()).select(&recs);
        assert_eq!(sel.len(), 1);
        assert_eq!(sel[0].reason, SelectionReason::FirstFrame);
    }

    #[test]
    fn test_distance_gate_2m() {
        // Records spaced 1 m apart — only every 3rd should fire the 2m gate
        let recs: Vec<TelemetryRecord> = (0..10)
            .map(|i| make_rec(i, 18.542, 73.727, 0.0, -90.0, 1.0, i as f64))
            .collect();
        let sel = KeyframeSelector::new(KeyframeThresholds::default()).select(&recs);
        // First + every ~2 steps = roughly 5 keyframes
        assert!(sel.len() >= 3 && sel.len() <= 7,
                "Expected 3–7 keyframes, got {}", sel.len());
        // At least one should be a distance trigger
        assert!(sel.iter().any(|k| matches!(k.reason, SelectionReason::Distance{..})));
    }

    #[test]
    fn test_yaw_gate_8deg() {
        // Records with 10° yaw step but no distance change
        let recs: Vec<TelemetryRecord> = (0..12).map(|i| {
            let mut r = make_rec(i, 18.542, 73.727, i as f64 * 10.0, -90.0, 0.5, i as f64);
            r.enu_m = Some([0.0, 0.0, 0.0]);   // no spatial movement
            r
        }).collect();
        let sel = KeyframeSelector::new(KeyframeThresholds::default()).select(&recs);
        assert!(sel.iter().any(|k| matches!(k.reason, SelectionReason::YawChange{..})),
                "Yaw gate should trigger");
    }

    #[test]
    fn test_pitch_gate_5deg() {
        let recs: Vec<TelemetryRecord> = (0..6).map(|i| {
            let mut r = make_rec(i, 18.542, 73.727, 0.0, i as f64 * 6.0, 0.5, i as f64);
            r.enu_m = Some([0.0, 0.0, 0.0]);
            r
        }).collect();
        let sel = KeyframeSelector::new(KeyframeThresholds::default()).select(&recs);
        assert!(sel.iter().any(|k| matches!(k.reason, SelectionReason::PitchChange{..})),
                "Pitch gate should trigger");
    }

    #[test]
    fn test_time_gap_gate_1s() {
        // Two records 2 seconds apart but no other changes
        let mut r0 = make_rec(0, 18.542, 73.727, 0.0, -90.0, 0.1, 0.0);
        let mut r1 = make_rec(1, 18.542, 73.727, 0.0, -90.0, 0.1, 2.0); // 2s gap
        r0.enu_m = Some([0.0, 0.0, 0.0]);
        r1.enu_m = Some([0.5, 0.0, 0.0]); // less than 2 m
        let sel = KeyframeSelector::new(KeyframeThresholds::default())
            .select(&[r0, r1]);
        assert!(sel.iter().any(|k| matches!(k.reason, SelectionReason::TimeGap{..})),
                "Time gate (>1 s) should trigger on 2 s gap");
    }

    #[test]
    fn test_high_speed_frames_rejected() {
        let recs: Vec<TelemetryRecord> = (0..5).map(|i| {
            let mut r = make_rec(i, 18.542 + i as f64*0.001, 73.727,
                                  0.0, -90.0, 12.0, i as f64);
            r.enu_m = Some([i as f64 * 5.0, 0.0, 0.0]);
            r
        }).collect();
        let sel = KeyframeSelector::new(KeyframeThresholds::default()).select(&recs);
        // All have speed 12 m/s > 8 m/s gate, so only first frame selected
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn test_pose_priors_built_from_keyframes() {
        let kfs: Vec<Keyframe> = (0..3).map(|i| Keyframe {
            record_idx: i, frame_idx: i, timestamp_s: i as f64,
            reason: SelectionReason::FirstFrame,
            enu_m: Some([i as f64 * 5.0, 0.0, 0.0]),
            r_prior: None,
        }).collect();
        let priors = build_pose_priors(&kfs);
        assert_eq!(priors.len(), 2);
        assert!((priors[0].baseline_m - 5.0).abs() < 0.01);
    }
}
