/* =============================================================================
 * crates/brush-sfm/cpp/opencv_sfm_wrapper.h
 * =============================================================================
 * C API over OpenCV C++ SfM functions.
 *
 * Diagram node: "OpenCV SfM (CXX)" inside the "Rust Wrapper" box.
 *
 * WHY A THIN C WRAPPER:
 *   Rust FFI only understands C ABI (no name-mangling, no C++ classes).
 *   OpenCV is C++ — we wrap the 5 calls we need with plain C functions that:
 *     - Accept flat float*/double* arrays (no cv::Mat crossing the boundary)
 *     - Construct cv::Mat internally inside the .cpp
 *     - Copy results back to caller-provided buffers
 *     - Return int error codes
 *
 * FUNCTIONS (all pipeline-stage relevant):
 *   sfm_sift_detect          → SIFT keypoints + 128-D descriptors
 *   sfm_match_features       → FLANN + Lowe ratio test
 *   sfm_find_essential_mat   → cv::findEssentialMat with USAC_MAGSAC
 *   sfm_recover_pose         → cv::recoverPose
 *   sfm_triangulate_points   → cv::triangulatePoints
 * ============================================================================= */

#ifndef OPENCV_SFM_WRAPPER_H
#define OPENCV_SFM_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ── Opaque handle — C++ object, Rust holds a raw pointer ─────────────────── */
typedef struct SfmFeatures SfmFeatures;

/* ── Return codes ─────────────────────────────────────────────────────────── */
#define SFM_OK             0
#define SFM_ERR_FEW_PTS   -1   /* < 8 point correspondences             */
#define SFM_ERR_RANSAC    -2   /* RANSAC produced no valid model         */
#define SFM_ERR_NULL      -3   /* null pointer argument                  */
#define SFM_ERR_INTERNAL  -4   /* internal OpenCV error                  */

/* ── RANSAC mode — RANSAC budget from slide 11 table ─────────────────────────
 *   MODE_A = vision only:    max_iters = 2000  (range 1000-5000)
 *   MODE_B = GPS fusion:     max_iters = 500   (range 200-800)
 *   MODE_C = full telemetry: max_iters = 100   (range 50-200)
 * ─────────────────────────────────────────────────────────────────────────── */
typedef enum {
    SFM_MODE_A = 0,
    SFM_MODE_B = 1,
    SFM_MODE_C = 2,
} SfmMode;

/* ── sfm_sift_detect ──────────────────────────────────────────────────────────
 * Calls: cv::SIFT::create(n_features)->detectAndCompute()
 *
 * img_gray   : uint8 grayscale image, width*height bytes, row-major
 * width,height: image dimensions
 * n_features : max keypoints to detect (0 = 8000 default)
 *
 * Returns opaque SfmFeatures* — free with sfm_features_free().
 * Returns NULL if no keypoints found or on error.
 * ─────────────────────────────────────────────────────────────────────────── */
SfmFeatures* sfm_sift_detect(
    const uint8_t* img_gray,
    int            width,
    int            height,
    int            n_features
);

/* ── sfm_features_count ────────────────────────────────────────────────────── */
int sfm_features_count(const SfmFeatures* feat);

/* ── sfm_features_keypoints ───────────────────────────────────────────────────
 * Copy keypoint (x, y) into buf[count*2]. Caller allocates buf.
 * Returns SFM_OK on success. */
int sfm_features_keypoints(const SfmFeatures* feat, float* buf);

/* ── sfm_match_features ───────────────────────────────────────────────────────
 * Calls: cv::FlannBasedMatcher::knnMatch + Lowe ratio test (ratio = 0.75)
 *
 * feat1, feat2    : handles from sfm_sift_detect
 * pts1_out,
 * pts2_out        : caller-allocated float[max_matches * 2]
 * max_matches     : buffer size (use sfm_features_count(feat1))
 * n_matches_out   : number of good matches written
 *
 * Returns SFM_OK on success.
 * ─────────────────────────────────────────────────────────────────────────── */
int sfm_match_features(
    const SfmFeatures* feat1,
    const SfmFeatures* feat2,
    float*             pts1_out,
    float*             pts2_out,
    int                max_matches,
    int*               n_matches_out
);

/* ── sfm_find_essential_mat ───────────────────────────────────────────────────
 * Calls: cv::findEssentialMat with USAC_MAGSAC
 *
 * Reference: Barath et al., "MAGSAC++, a Fast, Reliable and Accurate Robust
 *            Estimator", CVPR 2020. arXiv:1912.05909
 *
 * pts1, pts2  : float[n_pts * 2] — flattened [x0,y0, x1,y1, ...]
 * n_pts       : number of point correspondences (must be >= 8)
 * fx,fy,cx,cy : camera intrinsics (pixels)
 * mode        : RANSAC iteration budget (SFM_MODE_A/B/C)
 * e_out       : output double[9] — row-major 3×3 Essential matrix
 * inlier_mask : output uint8[n_pts] — 1=inlier, 0=outlier
 * n_inliers   : output: inlier count
 *
 * Returns SFM_OK on success.
 * ─────────────────────────────────────────────────────────────────────────── */
int sfm_find_essential_mat(
    const float* pts1,
    const float* pts2,
    int          n_pts,
    double       fx, double fy, double cx, double cy,
    SfmMode      mode,
    double*      e_out,
    uint8_t*     inlier_mask,
    int*         n_inliers
);

/* ── sfm_recover_pose ─────────────────────────────────────────────────────────
 * Calls: cv::recoverPose — decomposes E into R and t.
 *
 * pts1, pts2  : INLIER points only, float[n_pts * 2]
 * e_in        : Essential matrix double[9] from sfm_find_essential_mat
 * r_out       : output double[9] — row-major 3×3 rotation (cam2-from-cam1)
 * t_out       : output double[3] — unit translation (up to metric scale)
 *
 * Returns SFM_OK on success.
 * ─────────────────────────────────────────────────────────────────────────── */
int sfm_recover_pose(
    const float*  pts1,
    const float*  pts2,
    int           n_pts,
    const double* e_in,
    double        fx, double fy, double cx, double cy,
    double*       r_out,
    double*       t_out
);

/* ── sfm_triangulate_points ───────────────────────────────────────────────────
 * Calls: cv::triangulatePoints
 *
 * proj1, proj2 : 3×4 projection matrices double[12] row-major (K[R|t])
 * pts1, pts2   : matched 2D points float[n_pts * 2]
 * n_pts        : number of point pairs
 * pts3d_out    : output float[n_pts * 3] — only valid (z>0) points written
 * n_valid      : output: number of valid 3D points written
 *
 * Returns SFM_OK on success.
 * ─────────────────────────────────────────────────────────────────────────── */
int sfm_triangulate_points(
    const double* proj1,
    const double* proj2,
    const float*  pts1,
    const float*  pts2,
    int           n_pts,
    float*        pts3d_out,
    int*          n_valid
);

/* ── sfm_features_free ────────────────────────────────────────────────────── */
void sfm_features_free(SfmFeatures* feat);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENCV_SFM_WRAPPER_H */
