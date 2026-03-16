// =============================================================================
// crates/brush-sfm/src/sfm.rs
// =============================================================================
// PIPELINE STAGE: Incremental SfM  (diagram: "OpenCV SfM (CXX)" inside Rust Wrapper)
//
// This is the Rust FFI (Foreign Function Interface) layer that calls the
// OpenCV C++ wrapper (cpp/opencv_sfm_wrapper.cpp) compiled as a static library.
//
// Architecture from the diagram:
//   ┌─────────────────────────────────────────┐
//   │           Rust Wrapper                  │
//   │  ┌───────────────────────────────────┐  │
//   │  │        OpenCV SfM (CXX)           │  │
//   │  │  findEssentialMat (USAC_MAGSAC)   │  │
//   │  │  recoverPose                       │  │
//   │  │  triangulatePoints                 │  │
//   │  │  SIFT + FLANN                      │  │
//   │  └───────────────────────────────────┘  │
//   └─────────────────────────────────────────┘
//
// HOW FFI WORKS HERE:
//   1. cpp/opencv_sfm_wrapper.cpp is compiled by build.rs into a static .a
//   2. The extern "C" block below declares the C-ABI symbols from that .a
//   3. Safe Rust wrappers convert between Rust types and flat C arrays
//   4. pipeline.rs calls only the safe wrappers — never touches unsafe directly
//
// RANSAC budget by mode (slide 11 table):
//   VisionOnly / Mode A  →  max_iters = 2000
//   GpsOnly    / Mode B  →  max_iters = 500
//   FullDji    / Mode C  →  max_iters = 100
// =============================================================================

use anyhow::{anyhow, Result};
use crate::colmap_io::{CameraIntrinsics, Point3D};
use crate::telemetry::TelemetryMode;

// ─────────────────────────────────────────────────────────────────────────────
// C ENUM — must exactly match SfmMode in opencv_sfm_wrapper.h
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfmMode {
    ModeA = 0,   // vision only  — max_iters = 2000
    ModeB = 1,   // GPS fusion   — max_iters = 500
    ModeC = 2,   // full DJI     — max_iters = 100
}

impl From<TelemetryMode> for SfmMode {
    fn from(m: TelemetryMode) -> Self {
        match m {
            TelemetryMode::VisionOnly => SfmMode::ModeA,
            TelemetryMode::GpsOnly    => SfmMode::ModeB,
            TelemetryMode::FullDji    => SfmMode::ModeC,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C ERROR CODES — must match #defines in opencv_sfm_wrapper.h
// ─────────────────────────────────────────────────────────────────────────────

const SFM_OK:           i32 = 0;
const SFM_ERR_FEW_PTS:  i32 = -1;
const SFM_ERR_RANSAC:   i32 = -2;
const SFM_ERR_NULL:     i32 = -3;
const SFM_ERR_INTERNAL: i32 = -4;

// ─────────────────────────────────────────────────────────────────────────────
// OPAQUE C++ TYPE — Rust only holds a raw pointer
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque handle to the C++ SfmFeatures struct (keypoints + SIFT descriptors).
/// Never construct directly — only create via sift_detect().
#[repr(C)]
pub struct SfmFeaturesOpaque {
    _private: [u8; 0],
}

// ─────────────────────────────────────────────────────────────────────────────
// EXTERN "C" DECLARATIONS
// These symbols are defined in cpp/opencv_sfm_wrapper.cpp,
// compiled into libopencv_sfm_wrapper.a by build.rs.
// ─────────────────────────────────────────────────────────────────────────────

#[link(name = "opencv_sfm_wrapper", kind = "static")]
extern "C" {
    // cv::findEssentialMat(USAC_MAGSAC)
    fn sfm_find_essential_mat(
        pts1:         *const f32,
        pts2:         *const f32,
        n_pts:        i32,
        fx: f64, fy: f64, cx: f64, cy: f64,
        mode:         SfmMode,
        e_out:        *mut f64,      // [9] row-major 3×3
        inlier_mask:  *mut u8,       // [n_pts]
        n_inliers:    *mut i32,
    ) -> i32;

    // cv::recoverPose
    fn sfm_recover_pose(
        pts1:  *const f32,
        pts2:  *const f32,
        n_pts: i32,
        e_in:  *const f64,           // [9]
        fx: f64, fy: f64, cx: f64, cy: f64,
        r_out: *mut f64,             // [9] row-major 3×3
        t_out: *mut f64,             // [3]
    ) -> i32;

    // cv::triangulatePoints
    fn sfm_triangulate_points(
        proj1:     *const f64,       // [12] row-major 3×4
        proj2:     *const f64,       // [12] row-major 3×4
        pts1:      *const f32,
        pts2:      *const f32,
        n_pts:     i32,
        pts3d_out: *mut f32,         // [n_pts * 3] — caller allocates
        n_valid:   *mut i32,
    ) -> i32;

    // cv::SIFT::create()->detectAndCompute()
    fn sfm_sift_detect(
        img_gray:   *const u8,
        width:      i32,
        height:     i32,
        n_features: i32,             // 0 = use default (8000)
    ) -> *mut SfmFeaturesOpaque;

    // Returns keypoint count, -1 on null
    fn sfm_features_count(feat: *const SfmFeaturesOpaque) -> i32;

    // Copy keypoint (x, y) coords into buf — caller provides float[count*2]
    fn sfm_features_keypoints(
        feat: *const SfmFeaturesOpaque,
        buf:  *mut f32,
    ) -> i32;

    // FLANN + Lowe ratio test matching
    fn sfm_match_features(
        feat1:         *const SfmFeaturesOpaque,
        feat2:         *const SfmFeaturesOpaque,
        pts1_out:      *mut f32,
        pts2_out:      *mut f32,
        max_matches:   i32,
        n_matches_out: *mut i32,
    ) -> i32;

    // Free an SfmFeatures handle (safe to call with null)
    fn sfm_features_free(feat: *mut SfmFeaturesOpaque);
}

// ─────────────────────────────────────────────────────────────────────────────
// RAII WRAPPER — automatically frees the C++ SfmFeatures on drop
// ─────────────────────────────────────────────────────────────────────────────

pub struct Features {
    ptr: *mut SfmFeaturesOpaque,
}

// SAFETY: The C++ object is heap-allocated and not shared across threads
// in our single-threaded SfM pipeline.
unsafe impl Send for Features {}

impl Drop for Features {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sfm_features_free(self.ptr); }
        }
    }
}

impl Features {
    /// Number of keypoints detected.
    pub fn count(&self) -> usize {
        unsafe { sfm_features_count(self.ptr).max(0) as usize }
    }

    /// Get all keypoint (x, y) positions.
    pub fn keypoints(&self) -> Vec<[f32; 2]> {
        let n = self.count();
        if n == 0 { return vec![]; }
        let mut buf = vec![0f32; n * 2];
        unsafe { sfm_features_keypoints(self.ptr, buf.as_mut_ptr()); }
        buf.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PUBLIC SAFE API
// ─────────────────────────────────────────────────────────────────────────────

/// Detect SIFT keypoints and compute 128-D descriptors.
/// `img_gray` must be a contiguous row-major `width × height` u8 slice.
pub fn sift_detect(
    img_gray:   &[u8],
    width:      u32,
    height:     u32,
    n_features: u32,
) -> Result<Features> {
    assert_eq!(img_gray.len(), (width * height) as usize,
               "img_gray.len() must equal width*height");
    let ptr = unsafe {
        sfm_sift_detect(img_gray.as_ptr(), width as i32, height as i32, n_features as i32)
    };
    if ptr.is_null() {
        Err(anyhow!("sfm_sift_detect: no keypoints found (null return)"))
    } else {
        Ok(Features { ptr })
    }
}

/// FLANN-match two feature sets with Lowe's ratio test (threshold = 0.75).
/// Returns parallel `(pts1, pts2)` vectors of matched 2D coordinates.
pub fn match_features(f1: &Features, f2: &Features)
    -> Result<(Vec<[f32; 2]>, Vec<[f32; 2]>)>
{
    let max = f1.count().max(1);
    let mut pts1 = vec![0f32; max * 2];
    let mut pts2 = vec![0f32; max * 2];
    let mut n_out: i32 = 0;

    let rc = unsafe {
        sfm_match_features(
            f1.ptr, f2.ptr,
            pts1.as_mut_ptr(), pts2.as_mut_ptr(),
            max as i32, &mut n_out,
        )
    };
    check_rc(rc, "sfm_match_features")?;

    let n = n_out as usize;
    let out1 = pts1[..n*2].chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let out2 = pts2[..n*2].chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    Ok((out1, out2))
}

/// Result of Essential matrix estimation.
pub struct EssentialResult {
    /// Row-major 3×3 Essential matrix as [f64; 9]
    pub e:           [f64; 9],
    /// true = inlier for each input point
    pub inlier_mask: Vec<bool>,
    pub n_inliers:   usize,
}

/// Estimate the Essential matrix using USAC_MAGSAC.
///
/// RANSAC iterations:
///   Mode A (VisionOnly) → 2000  |  Mode B (GpsOnly) → 500  |  Mode C (FullDji) → 100
///
/// Reference: Barath et al., "MAGSAC++", CVPR 2020. arXiv:1912.05909
pub fn find_essential_mat(
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
    cam:  &CameraIntrinsics,
    mode: TelemetryMode,
) -> Result<EssentialResult> {
    let n = pts1.len();
    if n < 8 {
        return Err(anyhow!("find_essential_mat: need ≥8 point pairs, got {}", n));
    }
    assert_eq!(n, pts2.len());

    let flat1: Vec<f32> = pts1.iter().flat_map(|p| [p[0], p[1]]).collect();
    let flat2: Vec<f32> = pts2.iter().flat_map(|p| [p[0], p[1]]).collect();
    let mut e_out    = [0f64; 9];
    let mut mask     = vec![0u8; n];
    let mut n_in: i32 = 0;

    let rc = unsafe {
        sfm_find_essential_mat(
            flat1.as_ptr(), flat2.as_ptr(), n as i32,
            cam.fx, cam.fy, cam.cx, cam.cy,
            SfmMode::from(mode),
            e_out.as_mut_ptr(),
            mask.as_mut_ptr(),
            &mut n_in,
        )
    };
    check_rc(rc, "sfm_find_essential_mat")?;

    let inlier_mask: Vec<bool> = mask.iter().map(|&v| v != 0).collect();
    let n_inliers = n_in as usize;

    tracing::debug!(
        "find_essential_mat [{:?}]: {}/{} inliers ({:.1}%)",
        mode, n_inliers, n,
        n_inliers as f64 / n as f64 * 100.0
    );

    Ok(EssentialResult { e: e_out, inlier_mask, n_inliers })
}

/// Result of pose recovery.
pub struct PoseResult {
    /// Row-major 3×3 rotation matrix R (camera2_from_camera1) as [f64; 9]
    pub r: [f64; 9],
    /// Unit translation vector t (up to scale unless GPS-anchored) as [f64; 3]
    pub t: [f64; 3],
}

/// Recover camera rotation R and translation t from Essential matrix.
/// `pts1`, `pts2` should be the **inlier** points only.
pub fn recover_pose(
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
    e:    &[f64; 9],
    cam:  &CameraIntrinsics,
) -> Result<PoseResult> {
    let n = pts1.len();
    if n < 5 {
        return Err(anyhow!("recover_pose: need ≥5 inlier points, got {}", n));
    }
    let flat1: Vec<f32> = pts1.iter().flat_map(|p| [p[0], p[1]]).collect();
    let flat2: Vec<f32> = pts2.iter().flat_map(|p| [p[0], p[1]]).collect();
    let mut r = [0f64; 9];
    let mut t = [0f64; 3];

    let rc = unsafe {
        sfm_recover_pose(
            flat1.as_ptr(), flat2.as_ptr(), n as i32,
            e.as_ptr(),
            cam.fx, cam.fy, cam.cx, cam.cy,
            r.as_mut_ptr(), t.as_mut_ptr(),
        )
    };
    check_rc(rc, "sfm_recover_pose")?;
    Ok(PoseResult { r, t })
}

/// Triangulate matched 2D points into 3D world coordinates.
///
/// `proj1`, `proj2` are 3×4 projection matrices K[R|t] in row-major [f64; 12].
/// Only returns points with z > 0 (in front of both cameras).
/// `color` is a single RGB value assigned to all output points.
/// `id_start` is the first Point3D ID (for COLMAP output).
pub fn triangulate(
    proj1:    &[f64; 12],
    proj2:    &[f64; 12],
    pts1:     &[[f32; 2]],
    pts2:     &[[f32; 2]],
    color:    [u8; 3],
    id_start: usize,
) -> Result<Vec<Point3D>> {
    let n = pts1.len();
    if n == 0 { return Ok(vec![]); }

    let flat1: Vec<f32> = pts1.iter().flat_map(|p| [p[0], p[1]]).collect();
    let flat2: Vec<f32> = pts2.iter().flat_map(|p| [p[0], p[1]]).collect();
    let mut out3d    = vec![0f32; n * 3];
    let mut n_valid: i32 = 0;

    let rc = unsafe {
        sfm_triangulate_points(
            proj1.as_ptr(), proj2.as_ptr(),
            flat1.as_ptr(), flat2.as_ptr(), n as i32,
            out3d.as_mut_ptr(), &mut n_valid,
        )
    };
    check_rc(rc, "sfm_triangulate_points")?;

    let points = out3d[..(n_valid as usize * 3)]
        .chunks_exact(3)
        .enumerate()
        .map(|(i, c)| Point3D {
            point_id: id_start + i,
            x: c[0] as f64,
            y: c[1] as f64,
            z: c[2] as f64,
            r: color[0],
            g: color[1],
            b: color[2],
            error: 0.5,
        })
        .collect();

    Ok(points)
}

// ─────────────────────────────────────────────────────────────────────────────
// ERROR HELPER
// ─────────────────────────────────────────────────────────────────────────────

fn check_rc(rc: i32, fn_name: &str) -> Result<()> {
    match rc {
        SFM_OK           => Ok(()),
        SFM_ERR_FEW_PTS  => Err(anyhow!("{}: fewer than 8 point correspondences", fn_name)),
        SFM_ERR_RANSAC   => Err(anyhow!("{}: RANSAC failed to find a consistent model", fn_name)),
        SFM_ERR_NULL     => Err(anyhow!("{}: null pointer argument", fn_name)),
        SFM_ERR_INTERNAL => Err(anyhow!("{}: internal OpenCV error", fn_name)),
        other            => Err(anyhow!("{}: unknown error code {}", fn_name, other)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sfm_mode_values_match_c_header() {
        // C header: SFM_MODE_A=0, SFM_MODE_B=1, SFM_MODE_C=2
        assert_eq!(SfmMode::ModeA as i32, 0);
        assert_eq!(SfmMode::ModeB as i32, 1);
        assert_eq!(SfmMode::ModeC as i32, 2);
    }

    #[test]
    fn test_telemetry_mode_maps_correctly() {
        assert_eq!(SfmMode::from(TelemetryMode::VisionOnly), SfmMode::ModeA);
        assert_eq!(SfmMode::from(TelemetryMode::GpsOnly),    SfmMode::ModeB);
        assert_eq!(SfmMode::from(TelemetryMode::FullDji),    SfmMode::ModeC);
    }

    #[test]
    fn test_find_essential_mat_rejects_too_few_points() {
        let cam = CameraIntrinsics::from_fov(1920, 1080, 70.0);
        let pts: Vec<[f32; 2]> = (0..5).map(|i| [i as f32, i as f32]).collect();
        let result = find_essential_mat(&pts, &pts, &cam, TelemetryMode::VisionOnly);
        assert!(result.is_err(), "Should fail with fewer than 8 points");
        assert!(result.unwrap_err().to_string().contains("need ≥8"));
    }

    #[test]
    fn test_triangulate_empty_input() {
        let proj = [0f64; 12];
        let result = triangulate(&proj, &proj, &[], &[], [128, 128, 128], 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
}
