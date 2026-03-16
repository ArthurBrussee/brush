// =============================================================================
// crates/brush-sfm/src/pipeline.rs
// =============================================================================
// TOP-LEVEL PIPELINE ORCHESTRATOR
//
// Executes every node in the diagram in order:
//
//  [CSV]  → [Preprocessor]  ─┐
//                             ├→ [Telemetry Fusion] → [Keyframes]
//  [MP4]  → [MediaMetadata]  ─┘          ↑               │
//                                  (loop back)       ┌────┴────┐
//                                                    ▼         ▼
//                                           [Pose Priors]  [Match pruning]
//                                                    │         │
//                                                    └────┬────┘
//                                                         ▼
//                                              [Incremental SfM]
//                                              (OpenCV C++ via FFI)
//                                                         │
//                                                         ▼
//                                             [Gaussian Splatting]
//                                             (write COLMAP → Brush)
//                                                         │
//                                                         ▼
//                                                      [Brush]
// =============================================================================

use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

use crate::telemetry::{
    self, TelemetryMode, TelemetryRecord, FrameTimestamp,
};
use crate::keyframe::{
    KeyframeSelector, KeyframeThresholds, Keyframe, build_pose_priors, gps_epipolar_prune,
};
use crate::scene_loader::{
    BrushCamera, BrushScene, SparsePoint,
    write_colmap, write_init_ply, extract_keyframe_images,
};

#[cfg(feature = "opencv-sfm")]
use crate::sfm::{
    sift_detect, match_features,
    find_essential_mat, recover_pose, triangulate,
};

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE CONFIG
// ─────────────────────────────────────────────────────────────────────────────

/// Full configuration for one pipeline run.
pub struct PipelineConfig {
    /// Path to the DJI MP4 video
    pub mp4_path:  PathBuf,
    /// Path to the DJI CSV telemetry file (None = Mode A)
    pub csv_path:  Option<PathBuf>,
    /// Telemetry mode (A / B / C)
    pub mode:      TelemetryMode,
    /// Camera intrinsics — use DJI Mini 2 defaults if None
    pub intrinsics: Option<CameraIntrinsics>,
    /// Output directory — COLMAP sparse/ and images/ written here
    pub output_dir: PathBuf,
    /// Max frames to process (None = all)
    pub max_frames: Option<usize>,
}

/// DJI Mini 2 defaults (4K, 70° FOV, 30fps)
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
    /// Build from image size and horizontal FOV.
    pub fn from_fov(width: u32, height: u32, fov_h_deg: f64) -> Self {
        let fx = (width as f64 / 2.0) / (fov_h_deg / 2.0).to_radians().tan();
        Self { width, height, fx, fy: fx, cx: width as f64 / 2.0, cy: height as f64 / 2.0 }
    }

    /// DJI Mini 2 at 4K 70° FOV
    pub fn dji_mini2_4k() -> Self {
        Self::from_fov(3840, 2160, 70.0)
    }

    /// DJI Mini 2 at 1080p 70° FOV (common recording setting)
    pub fn dji_mini2_1080p() -> Self {
        Self::from_fov(1920, 1080, 70.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE RESULT
// ─────────────────────────────────────────────────────────────────────────────

pub struct PipelineResult {
    /// Number of keyframes selected
    pub n_keyframes: usize,
    /// Number of 3D points triangulated
    pub n_points: usize,
    /// Path to the COLMAP output directory — pass to Brush
    pub colmap_dir: PathBuf,
    /// Path to init.ply (sparse cloud seed for Gaussian init)
    pub init_ply:   PathBuf,
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN PIPELINE RUN
// ─────────────────────────────────────────────────────────────────────────────

pub fn run(cfg: &PipelineConfig) -> Result<PipelineResult> {
    tracing::info!("=== Drone SfM Pipeline  |  mode {:?} ===", cfg.mode);

    let cam = cfg.intrinsics.clone().unwrap_or_else(CameraIntrinsics::dji_mini2_1080p);

    // ── STAGE 1: CSV → Preprocessor ──────────────────────────────────────────
    let mut tele_records: Vec<TelemetryRecord> = if let Some(ref csv) = cfg.csv_path {
        tracing::info!("[1] Preprocessor: parsing {}", csv.display());
        telemetry::parse_dji_csv(csv, cfg.mode)
            .with_context(|| "CSV parse failed")?
    } else {
        tracing::info!("[1] Mode A — no CSV, generating synthetic timestamps");
        let n = cfg.max_frames.unwrap_or(300);
        (0..n).map(|i| TelemetryRecord {
            frame_idx: i, timestamp_s: i as f64 / 30.0,
            lat: 0.0, lon: 0.0, alt_m: 0.0,
            yaw_deg: 0.0, pitch_deg: -90.0, roll_deg: 0.0,
            vel_east_ms: 0.0, vel_north_ms: 0.0, vel_up_ms: 0.0,
            speed_ms: 0.0, enu_m: None, r_prior: None,
        }).collect()
    };
    tracing::info!("    → {} telemetry rows", tele_records.len());

    // ── STAGE 2: MP4 → MediaMetadataRetriever ────────────────────────────────
    let n_frames = cfg.max_frames.unwrap_or(tele_records.len());
    tracing::info!("[2] MediaMetadataRetriever: extracting {} frame timestamps", n_frames);
    let frame_pts = telemetry::extract_frame_timestamps(&cfg.mp4_path, n_frames)
        .with_context(|| "Frame timestamp extraction failed")?;

    // ── STAGE 3: Telemetry Fusion ─────────────────────────────────────────────
    tracing::info!("[3] Telemetry Fusion: aligning CSV to MP4 frames");
    let mut fused = telemetry::fuse(tele_records, &frame_pts);
    telemetry::attach_enu(&mut fused, cfg.mode);
    tracing::info!("    → {} fused records", fused.len());

    // ── STAGE 4: Keyframe Selection ───────────────────────────────────────────
    // Applies the diagram's condition box:
    //   distance > 2 m  OR  yaw > 8°  OR  pitch > 5°  OR  time > 1 s
    tracing::info!("[4] Keyframe selection (dist>2m OR yaw>8° OR pitch>5° OR t>1s)");
    let selector = KeyframeSelector::new(KeyframeThresholds::default());
    let keyframes = selector.select(&fused);
    tracing::info!("    → {} keyframes selected from {} frames ({:.1}% reduction)",
        keyframes.len(), fused.len(),
        (1.0 - keyframes.len() as f64 / fused.len() as f64) * 100.0);

    // ── STAGE 5: Pose Priors ──────────────────────────────────────────────────
    // Builds translation direction + rotation priors from GPS+IMU.
    // The diagram's loop-back arrow (Keyframes → Pose Priors) means:
    //   each keyframe's ENU position primes the RANSAC search.
    tracing::info!("[5] Pose Priors: building GPS/IMU priors for {} pairs",
                   keyframes.len().saturating_sub(1));
    let pose_priors = build_pose_priors(&keyframes);

    // ── STAGE 6: Extract Keyframe JPEGs ──────────────────────────────────────
    tracing::info!("[6] Extracting {} keyframe images from MP4", keyframes.len());
    let kf_indices: Vec<usize> = keyframes.iter().map(|k| k.frame_idx).collect();
    let image_paths = extract_keyframe_images(
        &cfg.mp4_path,
        &kf_indices,
        &cfg.output_dir,
    ).with_context(|| "Frame extraction failed")?;

    // ── STAGE 7: Incremental SfM (OpenCV via FFI) ─────────────────────────────
    // Match pruning → RANSAC+F/E matrix → triangulation
    tracing::info!("[7] Incremental SfM via OpenCV FFI");

    let mut cameras:  Vec<BrushCamera>  = Vec::new();
    let mut points3d: Vec<SparsePoint>  = Vec::new();
    let mut point_id_counter = 0usize;

    // First camera: identity pose (world origin)
    use glam::{Mat3, Vec3};
    let mut r_prev = Mat3::IDENTITY;
    let mut t_prev = Vec3::ZERO;

    // Register first keyframe
    if let Some(img_path) = image_paths.first() {
        cameras.push(BrushCamera {
            width: cam.width, height: cam.height,
            fx: cam.fx as f32, fy: cam.fy as f32,
            cx: cam.cx as f32, cy: cam.cy as f32,
            world_from_cam_r: r_prev,
            world_from_cam_t: t_prev,
            image_path: img_path.clone(),
        });
    }

    #[cfg(feature = "opencv-sfm")]
    {
        use crate::colmap_io::CameraIntrinsics as CvCam;
        let cv_cam = CvCam {
            width: cam.width, height: cam.height,
            fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy,
        };

        for i in 0..keyframes.len().saturating_sub(1) {
            let j = i + 1;

            // Load grayscale images
            let img_a = load_gray(&image_paths[i])?;
            let img_b = load_gray(&image_paths[j])?;

            // SIFT detect (slide 4 step 2–3)
            let feat_a = sift_detect(&img_a.pixels, cam.width, cam.height, 0)
                .with_context(|| format!("SIFT failed on frame {}", i))?;
            let feat_b = sift_detect(&img_b.pixels, cam.width, cam.height, 0)
                .with_context(|| format!("SIFT failed on frame {}", j))?;

            // FLANN match + ratio test (slide 4 step 4)
            let (mut pts1, mut pts2) = match_features(&feat_a, &feat_b)
                .with_context(|| format!("Matching failed pair ({},{})", i, j))?;

            if pts1.len() < 8 {
                tracing::warn!("    pair ({},{}) only {} matches — skip", i, j, pts1.len());
                continue;
            }

            // Match pruning via GPS epipolar filter (diagram: Match pruning node)
            if let Some(prior) = pose_priors.get(i) {
                let keep = gps_epipolar_prune(
                    &pts1, &pts2,
                    cam.fx, cam.fy, cam.cx, cam.cy,
                    prior,
                );
                pts1 = pts1.into_iter().zip(&keep).filter(|(_,&k)| k).map(|(p,_)| p).collect();
                pts2 = pts2.into_iter().zip(keep).filter(|(_,k)| *k).map(|(p,_)| p).collect();
            }

            tracing::info!("    pair ({},{}) {} matches after prune", i, j, pts1.len());

            if pts1.len() < 8 { continue; }

            // RANSAC Essential matrix (USAC_MAGSAC, mode budget)
            let ess_result = match find_essential_mat(&pts1, &pts2, &cv_cam, cfg.mode.into()) {
                Ok(r)  => r,
                Err(e) => { tracing::warn!("    RANSAC failed: {}", e); continue; }
            };

            let inlier_pts1: Vec<[f32;2]> = pts1.iter().zip(&ess_result.inlier_mask)
                .filter(|(_,&m)| m).map(|(p,_)| *p).collect();
            let inlier_pts2: Vec<[f32;2]> = pts2.iter().zip(&ess_result.inlier_mask)
                .filter(|(_,&m)| m).map(|(p,_)| *p).collect();

            tracing::info!("    RANSAC: {}/{} inliers", inlier_pts1.len(), pts1.len());

            // Recover R, t
            let pose = match recover_pose(&inlier_pts1, &inlier_pts2, &ess_result.e, &cv_cam) {
                Ok(p)  => p,
                Err(e) => { tracing::warn!("    recoverPose failed: {}", e); continue; }
            };

            // Accumulate global pose
            let r_rel = mat3_from_array(&pose.r);
            let t_rel = Vec3::new(pose.t[0] as f32, pose.t[1] as f32, pose.t[2] as f32);

            // GPS scale anchor: scale SfM translation to metric units
            let scale = if let Some(prior) = pose_priors.get(i) {
                if prior.baseline_m > 0.1 {
                    prior.baseline_m as f32 / t_rel.length().max(1e-6)
                } else { 1.0 }
            } else { 1.0 };

            let t_scaled  = t_rel * scale;
            let r_curr    = r_prev * r_rel;
            let t_curr    = t_prev + r_prev * t_scaled;

            // Register this camera
            cameras.push(BrushCamera {
                width: cam.width, height: cam.height,
                fx: cam.fx as f32, fy: cam.fy as f32,
                cx: cam.cx as f32, cy: cam.cy as f32,
                world_from_cam_r: r_curr,
                world_from_cam_t: t_curr,
                image_path: image_paths[j].clone(),
            });

            // Triangulate
            let proj1 = make_proj_mat(&cv_cam, &mat3_to_array(&r_prev), &t_prev);
            let proj2 = make_proj_mat(&cv_cam, &mat3_to_array(&r_curr), &t_curr);
            let color = sample_color_from_path(&image_paths[i]);

            let new_pts = triangulate(&proj1, &proj2, &inlier_pts1, &inlier_pts2,
                                      color, point_id_counter)?;
            point_id_counter += new_pts.len();

            for pt in new_pts {
                points3d.push(SparsePoint {
                    position: Vec3::new(pt.x as f32, pt.y as f32, pt.z as f32),
                    color:    [pt.r, pt.g, pt.b],
                });
            }

            tracing::info!("    triangulated {} pts (total {})", new_pts.len(), points3d.len());
            r_prev = r_curr;
            t_prev = t_curr;
        }
    }

    #[cfg(not(feature = "opencv-sfm"))]
    {
        tracing::warn!("    opencv-sfm feature not enabled — cameras registered from GPS only");
        // Fallback: register cameras from GPS ENU positions without visual SfM
        for (i, kf) in keyframes.iter().enumerate().skip(1) {
            if let (Some(img_path), Some(enu)) = (image_paths.get(i), kf.enu_m) {
                let t = Vec3::new(enu[0] as f32, enu[2] as f32, -enu[1] as f32);
                let r = kf.r_prior.map(|rp| mat3_from_array(&rp)).unwrap_or(Mat3::IDENTITY);
                cameras.push(BrushCamera {
                    width: cam.width, height: cam.height,
                    fx: cam.fx as f32, fy: cam.fy as f32,
                    cx: cam.cx as f32, cy: cam.cy as f32,
                    world_from_cam_r: r,
                    world_from_cam_t: t,
                    image_path: img_path.clone(),
                });
            }
        }
    }

    tracing::info!("[7] SfM complete: {} cameras, {} 3D points",
                   cameras.len(), points3d.len());

    // ── STAGE 8: Gaussian Splatting init — write COLMAP + init.ply → Brush ────
    tracing::info!("[8] Writing COLMAP + init.ply for Brush");
    let scene = BrushScene { cameras, points: points3d };
    write_colmap(&cfg.output_dir, &scene)?;
    write_init_ply(&cfg.output_dir, &scene.points)?;

    let colmap_dir = cfg.output_dir.join("sparse/0");
    let init_ply   = cfg.output_dir.join("init.ply");

    tracing::info!("=== Pipeline complete! ===");
    tracing::info!("    COLMAP: {:?}", colmap_dir);
    tracing::info!("    init.ply: {:?}", init_ply);
    tracing::info!("    Load {:?} in Brush to start training.", cfg.output_dir);

    Ok(PipelineResult {
        n_keyframes: keyframes.len(),
        n_points:    scene.points.len(),
        colmap_dir,
        init_ply,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPER STUBS  (real implementations depend on image loading crate chosen)
// ─────────────────────────────────────────────────────────────────────────────

struct GrayImage { pixels: Vec<u8>, }

fn load_gray(path: &Path) -> Result<GrayImage> {
    // In real brush_pbl: use the `image` crate (already a dep of brush-dataset)
    //   let img = image::open(path)?.to_luma8();
    //   Ok(GrayImage { pixels: img.into_raw() })
    // Stub returns empty buffer for compilation
    let _ = path;
    Ok(GrayImage { pixels: vec![128u8; 1920 * 1080] })
}

fn sample_color_from_path(_path: &Path) -> [u8; 3] {
    // In real brush_pbl: sample centre pixel of the image
    [128, 128, 128]
}

fn mat3_from_array(a: &[[f64;3];3]) -> glam::Mat3 {
    glam::Mat3::from_cols(
        glam::Vec3::new(a[0][0] as f32, a[1][0] as f32, a[2][0] as f32),
        glam::Vec3::new(a[0][1] as f32, a[1][1] as f32, a[2][1] as f32),
        glam::Vec3::new(a[0][2] as f32, a[1][2] as f32, a[2][2] as f32),
    )
}

fn mat3_to_array(m: &glam::Mat3) -> [[f64;3];3] {
    let c = m.to_cols_array_2d();
    // c is column-major; convert to row-major
    [[c[0][0] as f64, c[1][0] as f64, c[2][0] as f64],
     [c[0][1] as f64, c[1][1] as f64, c[2][1] as f64],
     [c[0][2] as f64, c[1][2] as f64, c[2][2] as f64]]
}

fn make_proj_mat(cam: &CameraIntrinsics, r: &[[f64;3];3], t: &glam::Vec3) -> [f64; 12] {
    // K[R|t]  — 3×4 flattened row-major
    let k = [[cam.fx, 0.0, cam.cx],
             [0.0, cam.fy, cam.cy],
             [0.0, 0.0,    1.0  ]];
    let rt = [
        [r[0][0], r[0][1], r[0][2], t.x as f64],
        [r[1][0], r[1][1], r[1][2], t.y as f64],
        [r[2][0], r[2][1], r[2][2], t.z as f64],
    ];
    let mut p = [0f64; 12];
    for i in 0..3 {
        for j in 0..4 {
            let mut v = 0.0f64;
            for kk in 0..3 { v += k[i][kk] * rt[kk][j]; }
            p[i*4+j] = v;
        }
    }
    p
}
