// =============================================================================
// crates/brush-sfm/src/lib.rs
// =============================================================================
// brush-sfm crate — drone video → COLMAP → Brush pipeline
//
// DIAGRAM NODES → MODULES:
//   CSV  → Preprocessor              telemetry::parse_dji_csv
//   MP4  → MediaMetadataRetriever    telemetry::extract_frame_timestamps
//          Telemetry Fusion           telemetry::fuse + attach_enu
//          Keyframes                  keyframe::KeyframeSelector
//          Pose Priors                keyframe::build_pose_priors
//          Match pruning              keyframe::gps_epipolar_prune
//          Incremental SfM            sfm::* (OpenCV C++ via FFI)
//          Gaussian Splatting         scene_loader::write_colmap + write_init_ply
//          Brush                      → pass output_dir to brush-process
//
// Feature flags:
//   opencv-sfm   — enables the C++ wrapper for SIFT/RANSAC/triangulate
//                  Requires: OPENCV_ANDROID_DIR env var
//                  Disabled for: wasm32-unknown-unknown
// =============================================================================

pub mod telemetry;
pub mod keyframe;
pub mod scene_loader;
pub mod pipeline;

#[cfg(feature = "opencv-sfm")]
pub mod sfm;

#[cfg(feature = "opencv-sfm")]
mod colmap_io;   // internal — used by sfm.rs for CameraIntrinsics type

// Re-export the main entry point
pub use pipeline::{run, PipelineConfig, PipelineResult, CameraIntrinsics};
pub use telemetry::TelemetryMode;
pub use keyframe::KeyframeThresholds;
