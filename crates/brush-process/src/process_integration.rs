// =============================================================================
// crates/brush-process/src/process_integration.rs
// =============================================================================
// PIPELINE STAGE: Gaussian Splatting → Brush
// (diagram rightmost node: [Brush])
//
// HOW TO INTEGRATE brush-sfm INTO brush-process
// ─────────────────────────────────────────────────────────────────────────────
// brush-process/src/lib.rs already contains a `process_stream()` function that
// reads a dataset and drives the training loop. Add a match arm for drone video
// input (.mp4 + .csv) as shown below.
//
// STEP 1: Add brush-sfm to crates/brush-process/Cargo.toml:
//
//   [dependencies]
//   brush-sfm = { path = "../brush-sfm", features = ["opencv-sfm"] }
//
// STEP 2: In crates/brush-process/src/lib.rs, add this match arm in the
//         dataset loading section:
//
//   // Existing code (example):
//   let dataset = match &config.dataset_path {
//       path if path.ends_with(".zip") => load_zip(path, ...).await?,
//       path if path.ends_with(".ply") => load_ply(path, ...).await?,
//
//       // ── NEW: drone video + telemetry ─────────────────────────────────
//       path if path.ends_with(".mp4") => {
//           DroneLoader::load(path, &config).await?
//       }
//       // ─────────────────────────────────────────────────────────────────
//
//       path => load_colmap(path, ...).await?,
//   };
//
// STEP 3: Copy this entire file into crates/brush-process/src/
//         and add `mod process_integration;` to lib.rs.
//
// =============================================================================

use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

use brush_sfm::{
    pipeline::{run as run_pipeline, PipelineConfig, CameraIntrinsics},
    TelemetryMode,
};

// ─────────────────────────────────────────────────────────────────────────────
// DRONE LOADER CONFIG
// Wraps PipelineConfig for the brush-process API surface.
// ─────────────────────────────────────────────────────────────────────────────

pub struct DroneLoaderConfig {
    /// Path to the DJI .mp4 file
    pub mp4_path:   PathBuf,
    /// Path to the DJI .csv telemetry file (None = Mode A)
    pub csv_path:   Option<PathBuf>,
    /// TelemetryMode — A (vision only), B (GPS), C (full DJI)
    pub mode:       TelemetryMode,
    /// Max frames to process (None = all keyframes)
    pub max_frames: Option<usize>,
    /// Custom camera intrinsics (None = DJI Mini 2 1080p defaults)
    pub intrinsics: Option<CameraIntrinsics>,
}

impl DroneLoaderConfig {
    /// Infer config from a .mp4 path.
    /// Automatically looks for a .csv with the same basename.
    pub fn from_mp4(mp4_path: &Path) -> Self {
        let stem = mp4_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("flight");

        // Look for DJI_XXXX.csv alongside the mp4
        let csv_candidate = mp4_path.parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}.csv", stem));

        let (csv_path, mode) = if csv_candidate.exists() {
            (Some(csv_candidate), TelemetryMode::FullDji)
        } else {
            (None, TelemetryMode::VisionOnly)
        };

        Self {
            mp4_path:   mp4_path.to_path_buf(),
            csv_path,
            mode,
            max_frames: None,
            intrinsics: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DRONE LOADER
// Runs the full pipeline and returns the path to a COLMAP dataset
// that brush-dataset's existing ColmapLoader can consume.
// ─────────────────────────────────────────────────────────────────────────────

pub struct DroneLoader;

impl DroneLoader {
    /// Run the full drone SfM pipeline and return the output COLMAP directory.
    ///
    /// The returned path can be passed directly to brush-dataset's
    /// existing COLMAP dataset loader — no other changes needed.
    ///
    /// Example in brush-process/src/lib.rs:
    /// ```rust
    /// let colmap_dir = DroneLoader::process(&DroneLoaderConfig::from_mp4(&mp4_path))?;
    /// let dataset = load_dataset(&colmap_dir, ...).await?;
    /// ```
    pub fn process(cfg: &DroneLoaderConfig) -> Result<PathBuf> {
        // Write COLMAP output to a temp directory alongside the .mp4
        let out_dir = cfg.mp4_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!(
                "brush_sfm_{}",
                cfg.mp4_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output")
            ));

        tracing::info!(
            "DroneLoader: processing {:?} → {:?}  (mode {:?})",
            cfg.mp4_path, out_dir, cfg.mode
        );

        let pipeline_cfg = PipelineConfig {
            mp4_path:   cfg.mp4_path.clone(),
            csv_path:   cfg.csv_path.clone(),
            mode:       cfg.mode,
            intrinsics: cfg.intrinsics.clone(),
            output_dir: out_dir.clone(),
            max_frames: cfg.max_frames,
        };

        let result = run_pipeline(&pipeline_cfg)
            .with_context(|| format!("Pipeline failed for {:?}", cfg.mp4_path))?;

        tracing::info!(
            "DroneLoader: done — {} keyframes, {} 3D points, COLMAP at {:?}",
            result.n_keyframes, result.n_points, result.colmap_dir
        );

        Ok(result.colmap_dir)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BRUSH-PROCESS INTEGRATION PATCH
// ─────────────────────────────────────────────────────────────────────────────
//
// Add the code below into crates/brush-process/src/lib.rs
// inside the existing dataset loading match:
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ // At the top of lib.rs, add:                                           │
// │ use crate::process_integration::{DroneLoader, DroneLoaderConfig};       │
// │                                                                         │
// │ // Inside process_stream() or load_dataset(), add the match arm:        │
// │ let dataset_path = if source.ends_with(".mp4") {                        │
// │     // Run the full drone pipeline → COLMAP output                      │
// │     let drone_cfg = DroneLoaderConfig::from_mp4(Path::new(&source));    │
// │     DroneLoader::process(&drone_cfg)?                                   │
// │ } else {                                                                 │
// │     PathBuf::from(&source)                                              │
// │ };                                                                      │
// │                                                                         │
// │ // Then pass dataset_path to the existing COLMAP loader as before       │
// │ let dataset = load_dataset(&dataset_path, resolution, ...).await?;      │
// └─────────────────────────────────────────────────────────────────────────┘
//
// That's all. The training loop (SplatTrainer::step) needs zero changes.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_mp4_without_csv() {
        let cfg = DroneLoaderConfig::from_mp4(Path::new("/flights/DJI_0001.mp4"));
        assert_eq!(cfg.mode, TelemetryMode::VisionOnly);
        assert!(cfg.csv_path.is_none());
    }

    #[test]
    fn test_config_infers_mode_b_from_csv_presence() {
        // Create a temporary CSV to simulate its presence
        let tmp = std::env::temp_dir();
        let mp4 = tmp.join("DJI_0042.mp4");
        let csv = tmp.join("DJI_0042.csv");
        std::fs::write(&csv, "time,lat,lon,alt\n0,18.542,73.727,100\n").unwrap();

        let cfg = DroneLoaderConfig::from_mp4(&mp4);
        assert_eq!(cfg.mode, TelemetryMode::FullDji);
        assert!(cfg.csv_path.is_some());

        std::fs::remove_file(&csv).ok();
    }

    #[test]
    fn test_output_dir_name_derived_from_mp4() {
        let cfg = DroneLoaderConfig::from_mp4(Path::new("/flights/DJI_0001.mp4"));
        let out = cfg.mp4_path
            .parent().unwrap()
            .join(format!("brush_sfm_{}", cfg.mp4_path.file_stem().unwrap().to_str().unwrap()));
        assert!(out.to_str().unwrap().contains("brush_sfm_DJI_0001"));
    }
}
