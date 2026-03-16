/* =============================================================================
 * crates/brush-sfm/cpp/opencv_sfm_wrapper.cpp
 * =============================================================================
 * OpenCV C++ implementation of every function declared in the header.
 * Compiled into libopencv_sfm_wrapper.a by build.rs using the `cc` crate.
 *
 * OpenCV modules used (all in BUILD_LIST from build_opencv_arm64.sh):
 *   opencv_calib3d   — findEssentialMat, recoverPose, triangulatePoints
 *   opencv_features2d — SIFT, FlannBasedMatcher
 *   opencv_flann     — KDTree index for FLANN
 *   opencv_imgproc   — (linked transitively)
 *   opencv_core      — Mat, Vector, etc.
 * ============================================================================= */

#include "opencv_sfm_wrapper.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

/* ── RANSAC iteration budgets from slide 11 table ─────────────────────────── */
static int ransac_iters(SfmMode mode) {
    switch (mode) {
        case SFM_MODE_A: return 2000;  /* vision only:    1000-5000 range */
        case SFM_MODE_B: return  500;  /* GPS fusion:      200-800  range */
        case SFM_MODE_C: return  100;  /* full telemetry:   50-200  range */
        default:         return 2000;
    }
}

/* ── Opaque struct definition ─────────────────────────────────────────────── */
struct SfmFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;   /* CV_32F, Nkp × 128 */
};

/* ── Helper: flat float[N*2] → cv::Mat 1×N of CV_32FC2 ────────────────────── */
static cv::Mat to_mat2f(const float* pts, int n) {
    cv::Mat m(1, n, CV_32FC2);
    for (int i = 0; i < n; i++)
        m.at<cv::Vec2f>(0, i) = cv::Vec2f(pts[i*2], pts[i*2+1]);
    return m;
}

/* ── Helper: build 3×3 camera matrix K ───────────────────────────────────── */
static cv::Mat build_K(double fx, double fy, double cx, double cy) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0,0) = fx;  K.at<double>(1,1) = fy;
    K.at<double>(0,2) = cx;  K.at<double>(1,2) = cy;
    return K;
}

/* ── sfm_sift_detect ──────────────────────────────────────────────────────── */
SfmFeatures* sfm_sift_detect(
    const uint8_t* img_gray, int width, int height, int n_features)
{
    if (!img_gray || width <= 0 || height <= 0) return nullptr;

    /* Wrap raw pixel buffer without copying */
    cv::Mat gray(height, width, CV_8U, const_cast<uint8_t*>(img_gray));

    int nf = (n_features <= 0) ? 8000 : n_features;
    auto sift = cv::SIFT::create(nf);

    SfmFeatures* feat = new SfmFeatures();
    sift->detectAndCompute(gray, cv::noArray(),
                           feat->keypoints, feat->descriptors);

    if (feat->keypoints.empty()) {
        delete feat;
        return nullptr;
    }
    return feat;
}

/* ── sfm_features_count ───────────────────────────────────────────────────── */
int sfm_features_count(const SfmFeatures* feat) {
    if (!feat) return -1;
    return static_cast<int>(feat->keypoints.size());
}

/* ── sfm_features_keypoints ───────────────────────────────────────────────── */
int sfm_features_keypoints(const SfmFeatures* feat, float* buf) {
    if (!feat || !buf) return SFM_ERR_NULL;
    for (size_t i = 0; i < feat->keypoints.size(); i++) {
        buf[i*2]   = feat->keypoints[i].pt.x;
        buf[i*2+1] = feat->keypoints[i].pt.y;
    }
    return SFM_OK;
}

/* ── sfm_match_features ───────────────────────────────────────────────────── */
int sfm_match_features(
    const SfmFeatures* feat1, const SfmFeatures* feat2,
    float* pts1_out, float* pts2_out,
    int max_matches, int* n_matches_out)
{
    if (!feat1 || !feat2 || !pts1_out || !pts2_out || !n_matches_out)
        return SFM_ERR_NULL;
    if (feat1->descriptors.empty() || feat2->descriptors.empty())
        return SFM_ERR_FEW_PTS;

    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(feat1->descriptors, feat2->descriptors, knn, 2);

    /* Lowe ratio test — threshold 0.75 */
    const float RATIO = 0.75f;
    int cnt = 0;
    for (const auto& pair : knn) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < RATIO * pair[1].distance) {
            if (cnt >= max_matches) break;
            auto& kp1 = feat1->keypoints[pair[0].queryIdx];
            auto& kp2 = feat2->keypoints[pair[0].trainIdx];
            pts1_out[cnt*2]   = kp1.pt.x;  pts1_out[cnt*2+1] = kp1.pt.y;
            pts2_out[cnt*2]   = kp2.pt.x;  pts2_out[cnt*2+1] = kp2.pt.y;
            cnt++;
        }
    }
    *n_matches_out = cnt;
    return SFM_OK;
}

/* ── sfm_find_essential_mat ───────────────────────────────────────────────────
 * USAC_MAGSAC: threshold-free, same optimal threshold across all datasets.
 * Reference: Barath et al., "MAGSAC++", CVPR 2020. arXiv:1912.05909
 * "USAC_MAGSAC is the only method whose optimal threshold is the same across
 *  all datasets — requires the least tuning." — OpenCV Blog, 2021
 * ─────────────────────────────────────────────────────────────────────────── */
int sfm_find_essential_mat(
    const float* pts1, const float* pts2, int n_pts,
    double fx, double fy, double cx, double cy,
    SfmMode mode,
    double* e_out, uint8_t* inlier_mask, int* n_inliers)
{
    if (!pts1 || !pts2 || !e_out || !inlier_mask || !n_inliers)
        return SFM_ERR_NULL;
    if (n_pts < 8)
        return SFM_ERR_FEW_PTS;

    cv::Mat p1 = to_mat2f(pts1, n_pts);
    cv::Mat p2 = to_mat2f(pts2, n_pts);
    cv::Mat K  = build_K(fx, fy, cx, cy);
    cv::Mat mask;

    cv::Mat E = cv::findEssentialMat(
        p1, p2, K,
        cv::USAC_MAGSAC,        /* threshold-free RANSAC                  */
        0.9999,                 /* confidence                             */
        1.0,                    /* threshold (MAGSAC adapts internally)   */
        ransac_iters(mode),     /* mode-dependent budget (slide 11)       */
        mask
    );

    if (E.empty()) return SFM_ERR_RANSAC;

    /* Copy 3×3 E matrix row-major */
    std::memcpy(e_out, E.ptr<double>(), 9 * sizeof(double));

    /* Copy inlier mask and count */
    int cnt = 0;
    for (int i = 0; i < n_pts; i++) {
        uint8_t v = (mask.at<uint8_t>(i) != 0) ? 1u : 0u;
        inlier_mask[i] = v;
        cnt += v;
    }
    *n_inliers = cnt;
    return SFM_OK;
}

/* ── sfm_recover_pose ─────────────────────────────────────────────────────── */
int sfm_recover_pose(
    const float* pts1, const float* pts2, int n_pts,
    const double* e_in,
    double fx, double fy, double cx, double cy,
    double* r_out, double* t_out)
{
    if (!pts1 || !pts2 || !e_in || !r_out || !t_out) return SFM_ERR_NULL;
    if (n_pts < 5) return SFM_ERR_FEW_PTS;

    cv::Mat p1 = to_mat2f(pts1, n_pts);
    cv::Mat p2 = to_mat2f(pts2, n_pts);
    cv::Mat K  = build_K(fx, fy, cx, cy);

    cv::Mat E(3, 3, CV_64F);
    std::memcpy(E.ptr<double>(), e_in, 9 * sizeof(double));

    cv::Mat R, t, pose_mask;
    int good = cv::recoverPose(E, p1, p2, K, R, t, pose_mask);
    if (good < 1) return SFM_ERR_RANSAC;

    std::memcpy(r_out, R.ptr<double>(), 9 * sizeof(double));
    std::memcpy(t_out, t.ptr<double>(), 3 * sizeof(double));
    return SFM_OK;
}

/* ── sfm_triangulate_points ───────────────────────────────────────────────── */
int sfm_triangulate_points(
    const double* proj1, const double* proj2,
    const float* pts1, const float* pts2, int n_pts,
    float* pts3d_out, int* n_valid)
{
    if (!proj1 || !proj2 || !pts1 || !pts2 || !pts3d_out || !n_valid)
        return SFM_ERR_NULL;
    if (n_pts < 1) return SFM_ERR_FEW_PTS;

    cv::Mat P1(3, 4, CV_64F), P2(3, 4, CV_64F);
    std::memcpy(P1.ptr<double>(), proj1, 12 * sizeof(double));
    std::memcpy(P2.ptr<double>(), proj2, 12 * sizeof(double));

    /* triangulatePoints expects 2×N */
    cv::Mat m1 = to_mat2f(pts1, n_pts);
    cv::Mat m2 = to_mat2f(pts2, n_pts);
    cv::Mat m1t, m2t;
    cv::transpose(m1.reshape(1, n_pts), m1t);
    cv::transpose(m2.reshape(1, n_pts), m2t);

    cv::Mat pts4d;
    cv::triangulatePoints(P1, P2, m1t, m2t, pts4d);  /* 4×N homogeneous */

    int valid = 0;
    for (int j = 0; j < n_pts; j++) {
        float w = pts4d.at<float>(3, j);
        if (std::abs(w) < 1e-9f) continue;
        float x = pts4d.at<float>(0, j) / w;
        float y = pts4d.at<float>(1, j) / w;
        float z = pts4d.at<float>(2, j) / w;
        if (z > 0.0f) {
            pts3d_out[valid*3]   = x;
            pts3d_out[valid*3+1] = y;
            pts3d_out[valid*3+2] = z;
            valid++;
        }
    }
    *n_valid = valid;
    return SFM_OK;
}

/* ── sfm_features_free ────────────────────────────────────────────────────── */
void sfm_features_free(SfmFeatures* feat) {
    delete feat;   /* delete nullptr is well-defined in C++ */
}
