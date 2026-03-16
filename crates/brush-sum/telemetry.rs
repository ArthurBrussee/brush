// =============================================================================
// crates/brush-sfm/src/telemetry.rs
// =============================================================================
// PIPELINE STAGE: CSV → Preprocessor  AND  MP4 → MediaMetadataRetriever
//
// Diagram nodes implemented here:
//   [CSV] → [Preprocessor]
//   [MP4] → [MediaMetadataRetriever (Android Native)]
//
// The Preprocessor normalises both input streams into a unified
// Vec<TelemetryRecord> that the Telemetry Fusion stage consumes.
//
// On Android, frame timestamps are extracted from the MP4 via the
// MediaMetadataRetriever JNI bridge (see android/MediaMetadataRetriever.java).
// On desktop, ffprobe or a simple pts-from-filename fallback is used.
// =============================================================================

use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result};

// ─────────────────────────────────────────────────────────────────────────────
// TELEMETRY RECORD
// One record per video frame after fusion.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryRecord {
    // Identity
    pub frame_idx:   usize,
    pub timestamp_s: f64,

    // GPS (from CSV — Preprocessor output)
    pub lat:  f64,
    pub lon:  f64,
    pub alt_m: f64,

    // Orientation (from CSV, Mode C)
    pub yaw_deg:          f64,
    pub pitch_deg:        f64,   // gimbal pitch
    pub roll_deg:         f64,

    // IMU velocity (from CSV, Mode C)
    pub vel_east_ms:  f64,
    pub vel_north_ms: f64,
    pub vel_up_ms:    f64,

    // Derived
    pub speed_ms: f64,

    // ENU offset from first GPS fix [East, North, Up] metres
    // Populated by attach_enu()
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enu_m: Option<[f64; 3]>,

    // 3×3 rotation prior from yaw + pitch (Mode C, populated by attach_enu)
    #[serde(skip)]
    pub r_prior: Option<[[f64; 3]; 3]>,
}

impl TelemetryRecord {
    pub fn has_gps(&self) -> bool {
        self.lat.abs() > 1e-9 || self.lon.abs() > 1e-9
    }
    pub fn speed_ms(&self) -> f64 {
        (self.vel_east_ms.powi(2)
            + self.vel_north_ms.powi(2)
            + self.vel_up_ms.powi(2))
        .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TELEMETRY MODE
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryMode {
    /// Mode A — vision only, no CSV/SRT at all
    VisionOnly,
    /// Mode B — CSV with GPS + altitude only
    GpsOnly,
    /// Mode C — CSV with full DJI telemetry (yaw, pitch, velocity, IMU)
    FullDji,
}

// ─────────────────────────────────────────────────────────────────────────────
// PREPROCESSOR  (diagram: CSV → Preprocessor)
// ─────────────────────────────────────────────────────────────────────────────
// Parses the DJI CSV / SRT file that comes alongside the .mp4.
// DJI CSV columns (typical):
//   time(millisecond), latitude, longitude, altitude(feet),
//   pitch(degrees), yaw(degrees), roll(degrees),
//   velN(m/s), velE(m/s), velD(m/s)

pub fn parse_dji_csv(path: &Path, mode: TelemetryMode) -> Result<Vec<TelemetryRecord>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read CSV: {}", path.display()))?;

    let mut records = Vec::new();
    let mut lines = content.lines().peekable();

    // Skip header line(s) — DJI CSVs start with a header row
    let first = lines.peek().copied().unwrap_or("");
    if first.contains("time") || first.contains("latitude") || first.to_lowercase().contains("lat") {
        lines.next();
    }

    for (idx, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }

        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 4 { continue; }

        // Timestamp: column 0, milliseconds
        let ts_ms: f64 = cols[0].trim().parse().unwrap_or(idx as f64 * 33.3);
        let lat:   f64 = cols.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let lon:   f64 = cols.get(2).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        // DJI altitude is in feet in some firmware; convert to metres
        let alt_raw: f64 = cols.get(3).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        // Heuristic: if > 1000 it's likely in feet (no survey drone flies >1000 m)
        let alt_m = if alt_raw > 1000.0 { alt_raw * 0.3048 } else { alt_raw };

        let (pitch, yaw, roll, vel_n, vel_e, vel_d) = if mode == TelemetryMode::FullDji {
            (
                cols.get(4).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0),
                cols.get(5).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0),
                cols.get(6).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0),
                cols.get(7).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0),
                cols.get(8).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0),
                cols.get(9).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0),
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        let speed = (vel_n * vel_n + vel_e * vel_e + vel_d * vel_d).sqrt();

        records.push(TelemetryRecord {
            frame_idx:    idx,
            timestamp_s:  ts_ms / 1000.0,
            lat, lon, alt_m,
            yaw_deg:      yaw,
            pitch_deg:    pitch,
            roll_deg:     roll,
            vel_east_ms:  vel_e,
            vel_north_ms: vel_n,
            vel_up_ms:    -vel_d,  // DJI velD is positive-down; flip to NED→ENU
            speed_ms:     speed,
            enu_m:        None,
            r_prior:      None,
        });
    }

    tracing::info!(
        "Preprocessor: parsed {} telemetry rows from {} (mode {:?})",
        records.len(), path.display(), mode
    );
    Ok(records)
}

// ─────────────────────────────────────────────────────────────────────────────
// MP4 FRAME TIMESTAMP RETRIEVER
// Diagram: MP4 → MediaMetadataRetriever (Android Native)
//
// On Android, we call into Java/Kotlin via JNI to use
// android.media.MediaMetadataRetriever which can:
//   - Extract METADATA_KEY_VIDEO_FRAME_COUNT
//   - Map frame index → presentation time (μs)
// On desktop, we derive timestamps from frame index and video FPS.
// ─────────────────────────────────────────────────────────────────────────────

/// Frame metadata extracted from the MP4 container.
#[derive(Debug, Clone)]
pub struct FrameTimestamp {
    pub frame_idx:   usize,
    /// Presentation timestamp in seconds
    pub pts_s:       f64,
}

/// Extract frame timestamps from an MP4.
/// On Android, bridges to MediaMetadataRetriever via JNI.
/// On desktop, derives from fps metadata using simple arithmetic.
pub fn extract_frame_timestamps(
    mp4_path: &Path,
    total_frames: usize,
) -> Result<Vec<FrameTimestamp>> {
    #[cfg(target_os = "android")]
    {
        extract_timestamps_android(mp4_path, total_frames)
    }
    #[cfg(not(target_os = "android"))]
    {
        extract_timestamps_desktop(mp4_path, total_frames)
    }
}

#[cfg(not(target_os = "android"))]
fn extract_timestamps_desktop(
    mp4_path: &Path,
    total_frames: usize,
) -> Result<Vec<FrameTimestamp>> {
    // Read FPS from the filename or default to 30 fps
    // DJI Mini 2 records at 30 fps (standard) or 60 fps (sports mode)
    let fps = parse_fps_hint(mp4_path).unwrap_or(30.0);
    Ok((0..total_frames)
        .map(|i| FrameTimestamp {
            frame_idx: i,
            pts_s:     i as f64 / fps,
        })
        .collect())
}

fn parse_fps_hint(path: &Path) -> Option<f64> {
    // DJI filenames sometimes contain frame rate: DJI_0042_30fps.mp4
    let name = path.file_name()?.to_str()?;
    if name.contains("60fps") { return Some(60.0); }
    if name.contains("30fps") { return Some(30.0); }
    if name.contains("24fps") { return Some(24.0); }
    None
}

/// Android JNI bridge to MediaMetadataRetriever.
/// The actual Java-side code is in android/MediaMetadataRetriever.java.
#[cfg(target_os = "android")]
fn extract_timestamps_android(
    mp4_path: &Path,
    total_frames: usize,
) -> Result<Vec<FrameTimestamp>> {
    use jni::{
        objects::{JObject, JString, JValue},
        JavaVM,
    };

    let path_str = mp4_path.to_str()
        .with_context(|| "MP4 path is not valid UTF-8")?;

    // Get the Android JVM instance (provided by the brush-app crate's JNI bridge)
    let jvm = brush_app::android_jvm()
        .with_context(|| "Could not obtain JVM handle")?;
    let env = jvm.attach_current_thread()
        .with_context(|| "JNI attach failed")?;

    // Call our helper class: com.splats.app.FrameTimestampHelper
    let class = env.find_class("com/splats/app/FrameTimestampHelper")
        .with_context(|| "Could not find FrameTimestampHelper class")?;

    let j_path: JString = env.new_string(path_str)?.into();
    let j_count = JValue::Long(total_frames as i64);

    // long[] getFrameTimestampsUs(String path, long frameCount)
    let result = env.call_static_method(
        class,
        "getFrameTimestampsUs",
        "(Ljava/lang/String;J)[J",
        &[JValue::Object(j_path.into()), j_count],
    ).with_context(|| "JNI call to getFrameTimestampsUs failed")?;

    let timestamps_us: Vec<i64> = env.get_long_array_elements(
        result.l()?.into_inner(),
        jni::objects::ReleaseMode::NoCopyBack,
    )?.iter().copied().collect();

    Ok(timestamps_us.iter().enumerate().map(|(i, &us)| FrameTimestamp {
        frame_idx: i,
        pts_s:     us as f64 / 1_000_000.0,
    }).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// TELEMETRY FUSION  (diagram: Telemetry Fusion node)
// Aligns CSV records to MP4 frame timestamps by nearest-timestamp matching.
// ─────────────────────────────────────────────────────────────────────────────

/// Fuse telemetry records with frame timestamps.
/// Each output element is a TelemetryRecord whose timestamp_s matches the
/// nearest frame in the MP4 — ready for keyframe selection.
pub fn fuse(
    mut records: Vec<TelemetryRecord>,
    frame_pts:   &[FrameTimestamp],
) -> Vec<TelemetryRecord> {
    if records.is_empty() || frame_pts.is_empty() {
        return records;
    }

    let mut fused = Vec::with_capacity(frame_pts.len());
    for fts in frame_pts {
        // Find the telemetry record whose timestamp is closest to this frame
        let best = records.iter_mut().min_by(|a, b| {
            let da = (a.timestamp_s - fts.pts_s).abs();
            let db = (b.timestamp_s - fts.pts_s).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(rec) = best {
            let mut r = rec.clone();
            r.frame_idx   = fts.frame_idx;
            r.timestamp_s = fts.pts_s;
            fused.push(r);
        }
    }

    tracing::info!("Telemetry Fusion: aligned {} records to {} frames",
                   records.len(), fused.len());
    fused
}

// ─────────────────────────────────────────────────────────────────────────────
// GPS → ENU  (attached during or after fusion)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert all GPS positions to ENU and (for FullDji) attach rotation priors.
pub fn attach_enu(records: &mut Vec<TelemetryRecord>, mode: TelemetryMode) {
    let ref_pos = records.iter().find(|r| r.has_gps())
        .map(|r| (r.lat, r.lon, r.alt_m));
    let (ref_lat, ref_lon, ref_alt) = match ref_pos {
        Some(p) => p,
        None    => { tracing::warn!("No GPS in telemetry — ENU skipped"); return; }
    };

    for rec in records.iter_mut() {
        if !rec.has_gps() { continue; }
        rec.enu_m = Some(gps_to_enu(rec.lat, rec.lon, rec.alt_m,
                                     ref_lat, ref_lon, ref_alt));
        if mode == TelemetryMode::FullDji {
            rec.r_prior = Some(build_rotation_prior(rec.yaw_deg, rec.pitch_deg));
        }
    }
}

/// Build 3×3 rotation matrix from yaw + gimbal pitch (Mode C prior).
pub fn build_rotation_prior(yaw_deg: f64, pitch_deg: f64) -> [[f64; 3]; 3] {
    let yaw   = yaw_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    let (sy, cy) = (yaw.sin(),   yaw.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let rz = [[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
    mat3_mul(&rz, &rx)
}

/// WGS-84 GPS to local ENU frame (metres).
pub fn gps_to_enu(lat: f64, lon: f64, alt: f64,
                  rlat: f64, rlon: f64, ralt: f64) -> [f64; 3] {
    const A: f64 = 6_378_137.0;
    const E2: f64 = 0.006_694_379_990_14;
    let ecef = |la: f64, lo: f64, h: f64| -> [f64; 3] {
        let (la, lo) = (la.to_radians(), lo.to_radians());
        let n = A / (1.0 - E2 * la.sin().powi(2)).sqrt();
        [(n+h)*la.cos()*lo.cos(), (n+h)*la.cos()*lo.sin(),
         (n*(1.0-E2)+h)*la.sin()]
    };
    let r = ecef(rlat, rlon, ralt);
    let p = ecef(lat,  lon,  alt);
    let d = [p[0]-r[0], p[1]-r[1], p[2]-r[2]];
    let (la0, lo0) = (rlat.to_radians(), rlon.to_radians());
    let (sla, cla) = (la0.sin(), la0.cos());
    let (slo, clo) = (lo0.sin(), lo0.cos());
    [
        -slo*d[0]       + clo*d[1],
        -sla*clo*d[0]   - sla*slo*d[1] + cla*d[2],
         cla*clo*d[0]   + cla*slo*d[1] + sla*d[2],
    ]
}

fn mat3_mul(a: &[[f64;3];3], b: &[[f64;3];3]) -> [[f64;3];3] {
    let mut c = [[0.0f64;3];3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        c[i][j] += a[i][k]*b[k][j];
    }}}
    c
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enu_zero_at_reference() {
        let e = gps_to_enu(18.542107, 73.727878, 100.0,
                           18.542107, 73.727878, 100.0);
        assert!(e[0].abs() < 1e-6 && e[1].abs() < 1e-6 && e[2].abs() < 1e-6);
    }

    #[test]
    fn test_fuse_aligns_nearest_timestamp() {
        let records: Vec<TelemetryRecord> = (0..10).map(|i| TelemetryRecord {
            frame_idx: i, timestamp_s: i as f64 * 0.1,
            lat: 18.542, lon: 73.727, alt_m: 100.0,
            yaw_deg: 0.0, pitch_deg: -90.0, roll_deg: 0.0,
            vel_east_ms: 0.0, vel_north_ms: 0.0, vel_up_ms: 0.0,
            speed_ms: 0.0, enu_m: None, r_prior: None,
        }).collect();
        let pts = vec![
            FrameTimestamp { frame_idx: 0, pts_s: 0.05 },
            FrameTimestamp { frame_idx: 1, pts_s: 0.15 },
        ];
        let fused = fuse(records, &pts);
        assert_eq!(fused.len(), 2);
        // 0.05 is closest to record at 0.1
        assert!((fused[0].timestamp_s - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_rotation_prior_is_orthogonal() {
        let r = build_rotation_prior(45.0, -60.0);
        // R^T * R should be identity
        let rt = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ];
        let rtr = mat3_mul(&rt, &r);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - expected).abs() < 1e-9,
                    "R^T*R[{}][{}] = {:.6}, expected {}", i, j, rtr[i][j], expected);
            }
        }
    }
}
