// =============================================================================
// crates/brush-sfm/src/colmap_io.rs
// =============================================================================
// Internal types shared between sfm.rs and pipeline.rs.
// CameraIntrinsics holds the opencv-side camera model (f64 for OpenCV),
// while pipeline.rs uses CameraIntrinsics (f64) from pipeline.rs itself.
// This module exists so sfm.rs can import a clean type without a circular dep.
// =============================================================================

/// Pinhole camera model — used by the OpenCV C FFI layer.
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    pub width:  u32,
    pub height: u32,
    pub fx:     f64,
    pub fy:     f64,
    pub cx:     f64,
    pub cy:     f64,
}

impl CameraIntrinsics {
    pub fn from_fov(width: u32, height: u32, fov_h_deg: f64) -> Self {
        let fx = (width as f64 / 2.0) / (fov_h_deg / 2.0).to_radians().tan();
        Self { width, height, fx, fy: fx, cx: width as f64 / 2.0, cy: height as f64 / 2.0 }
    }
}

/// A triangulated 3D point returned by sfm::triangulate.
#[derive(Debug, Clone)]
pub struct Point3D {
    pub point_id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub error: f64,
}
