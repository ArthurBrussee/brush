// =============================================================================
// crates/brush-sfm/src/scene_loader.rs
// =============================================================================
// PIPELINE STAGE: Incremental SfM → Gaussian Splatting → Brush
//
// This file is the INTEGRATION POINT with the actual brush_pbl codebase.
//
// Brush's training pipeline (brush-process → brush-train) expects data in
// COLMAP format. This module:
//   1. Takes the output of Incremental SfM (registered cameras + sparse points)
//   2. Writes COLMAP sparse/0/ text files to a temp directory in brush-vfs
//   3. Hands that directory to Brush's existing ColmapLoader
//   4. Brush's SplatTrainer then initialises Gaussians from the sparse cloud
//
// HOW TO HOOK INTO brush-process/src/lib.rs:
//
//   In process_stream() where datasets are loaded, add a match arm:
//
//   ProcessConfig::DroneVideo { mp4, csv, mode } => {
//       let scene = DroneSceneLoader::load(mp4, csv, mode).await?;
//       // scene is a BrushDataset already — pass to existing training loop
//       scene
//   }
//
// The BrushDataset type (from brush-dataset) wraps:
//   - Vec<Camera>  with intrinsics and world-to-camera extrinsics
//   - Vec<Arc<DynamicImage>>  with the actual JPEG frames
//   - Vec<glam::Vec3>  as the initial 3D point cloud seed for Gaussians
// =============================================================================

use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

// Brush internal types (from brush-dataset crate)
// These are the exact types brush-dataset exposes:
//   brush_dataset::scene::Scene          — holds cameras + point cloud
//   brush_dataset::camera::Camera        — focal, principal, world_from_cam
//   brush_dataset::colmap::load_dataset  — the existing COLMAP loader
use glam::{Mat3, Vec3, Quat};

/// A camera ready for Brush's training pipeline.
/// Matches brush-dataset's internal Camera struct layout.
#[derive(Debug, Clone)]
pub struct BrushCamera {
    /// Image width and height in pixels
    pub width:  u32,
    pub height: u32,
    /// Focal lengths in pixels
    pub fx: f32,
    pub fy: f32,
    /// Principal point in pixels
    pub cx: f32,
    pub cy: f32,
    /// World-from-camera rotation (camera pose in world space)
    pub world_from_cam_r: Mat3,
    /// World-from-camera translation
    pub world_from_cam_t: Vec3,
    /// Path to the extracted JPEG for this camera
    pub image_path: PathBuf,
}

/// A sparse 3D point from triangulation — seeds the initial Gaussians.
#[derive(Debug, Clone)]
pub struct SparsePoint {
    pub position: Vec3,
    pub color:    [u8; 3],
}

/// Everything Brush needs to start training.
pub struct BrushScene {
    pub cameras: Vec<BrushCamera>,
    pub points:  Vec<SparsePoint>,
}

// ─────────────────────────────────────────────────────────────────────────────
// COLMAP TEXT WRITER
// Writes COLMAP sparse/0/ that brush-dataset's ColmapLoader reads directly.
// ─────────────────────────────────────────────────────────────────────────────

/// Write BrushScene as COLMAP sparse/0/ text files.
///
/// Output structure:
///   <dir>/
///     images/           ← JPEG frames (already extracted to here)
///     sparse/
///       0/
///         cameras.txt
///         images.txt
///         points3D.txt
///
/// Pass <dir> to Brush as the dataset path — its existing COLMAP loader
/// handles the rest.
pub fn write_colmap(dir: &Path, scene: &BrushScene) -> Result<()> {
    let sparse = dir.join("sparse").join("0");
    std::fs::create_dir_all(&sparse)?;

    write_cameras_txt(&sparse, scene)?;
    write_images_txt(&sparse, scene)?;
    write_points3d_txt(&sparse, scene)?;

    tracing::info!(
        "Wrote COLMAP sparse to {:?} ({} cameras, {} points)",
        sparse, scene.cameras.len(), scene.points.len()
    );
    Ok(())
}

fn write_cameras_txt(dir: &Path, scene: &BrushScene) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(dir.join("cameras.txt"))?;
    writeln!(f, "# Camera list with one line of data per camera:")?;
    writeln!(f, "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]")?;
    writeln!(f, "# Number of cameras: 1")?;
    // All frames share one camera model (DJI Mini 2 fixed lens)
    let c = &scene.cameras[0];
    writeln!(f, "1 PINHOLE {} {} {:.6} {:.6} {:.6} {:.6}",
             c.width, c.height, c.fx, c.fy, c.cx, c.cy)?;
    Ok(())
}

fn write_images_txt(dir: &Path, scene: &BrushScene) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(dir.join("images.txt"))?;
    writeln!(f, "# Image list with two lines of data per image:")?;
    writeln!(f, "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME")?;
    writeln!(f, "# POINTS2D[] as (X, Y, POINT3D_ID)")?;

    for (i, cam) in scene.cameras.iter().enumerate() {
        // Convert world-from-cam rotation to camera-from-world (COLMAP convention)
        // COLMAP stores R_cw (world→camera), not R_wc (camera→world)
        let r_cw = cam.world_from_cam_r.transpose();
        let t_cw = -(r_cw * cam.world_from_cam_t);

        // R_cw → quaternion [qw, qx, qy, qz]
        let q = Quat::from_mat3(&r_cw);
        let name = cam.image_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("frame.jpg");

        writeln!(f,
            "{} {:.8} {:.8} {:.8} {:.8} {:.8} {:.8} {:.8} 1 {}",
            i + 1,
            q.w, q.x, q.y, q.z,
            t_cw.x, t_cw.y, t_cw.z,
            name
        )?;
        writeln!(f)?;   // empty line (COLMAP format: 2 lines per image)
    }
    Ok(())
}

fn write_points3d_txt(dir: &Path, scene: &BrushScene) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(dir.join("points3D.txt"))?;
    writeln!(f, "# 3D point list with one line of data per point:")?;
    writeln!(f, "# POINT3D_ID X Y Z R G B ERROR TRACK[]")?;
    for (i, pt) in scene.points.iter().enumerate() {
        writeln!(f, "{} {:.6} {:.6} {:.6} {} {} {} 0.5",
                 i + 1, pt.position.x, pt.position.y, pt.position.z,
                 pt.color[0], pt.color[1], pt.color[2])?;
    }
    Ok(())
}

/// Write the sparse point cloud as a PLY file at <dir>/init.ply
/// Brush 0.3 picks this up automatically if present alongside the dataset.
pub fn write_init_ply(dir: &Path, points: &[SparsePoint]) -> Result<()> {
    use std::io::Write;
    let path = dir.join("init.ply");
    let mut f = std::fs::File::create(&path)?;
    writeln!(f, "ply")?;
    writeln!(f, "format ascii 1.0")?;
    writeln!(f, "element vertex {}", points.len())?;
    writeln!(f, "property float x")?;
    writeln!(f, "property float y")?;
    writeln!(f, "property float z")?;
    writeln!(f, "property uchar red")?;
    writeln!(f, "property uchar green")?;
    writeln!(f, "property uchar blue")?;
    writeln!(f, "end_header")?;
    for pt in points {
        writeln!(f, "{:.6} {:.6} {:.6} {} {} {}",
                 pt.position.x, pt.position.y, pt.position.z,
                 pt.color[0], pt.color[1], pt.color[2])?;
    }
    tracing::info!("Wrote init.ply ({} points) → {:?}", points.len(), path);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// FRAME EXTRACTOR
// Extracts the selected keyframe JPEGs from the MP4 into <dir>/images/
// On Android: uses the MediaMetadataRetriever bridge from telemetry.rs
// On desktop: uses image extraction via frame index arithmetic
// ─────────────────────────────────────────────────────────────────────────────

/// Extract keyframe images from the MP4 into <out_dir>/images/.
/// Returns Vec of paths in the same order as keyframe_indices.
///
/// On Android this calls through to the JNI bridge.
/// On desktop this is a no-op stub — images are expected to already be
/// extracted (e.g. via ffmpeg before running the pipeline).
pub fn extract_keyframe_images(
    mp4_path:         &Path,
    keyframe_indices: &[usize],
    out_dir:          &Path,
) -> Result<Vec<PathBuf>> {
    let images_dir = out_dir.join("images");
    std::fs::create_dir_all(&images_dir)?;

    #[cfg(target_os = "android")]
    {
        extract_images_android(mp4_path, keyframe_indices, &images_dir)
    }
    #[cfg(not(target_os = "android"))]
    {
        // Desktop: images assumed to be at <mp4_dir>/frames/frame_NNNN.jpg
        let frame_dir = mp4_path.parent().unwrap_or(Path::new(".")).join("frames");
        let paths: Vec<PathBuf> = keyframe_indices.iter().map(|&idx| {
            frame_dir.join(format!("frame_{:04}.jpg", idx))
        }).collect();
        // Symlink or copy into images_dir
        for (i, src) in paths.iter().enumerate() {
            let dst = images_dir.join(format!("frame_{:04}.jpg", keyframe_indices[i]));
            if src.exists() && !dst.exists() {
                std::fs::copy(src, &dst)
                    .with_context(|| format!("copy {:?} → {:?}", src, dst))?;
            }
        }
        Ok(keyframe_indices.iter()
            .map(|&i| images_dir.join(format!("frame_{:04}.jpg", i)))
            .collect())
    }
}

/// Android JNI call to MediaMetadataRetriever for frame extraction.
#[cfg(target_os = "android")]
fn extract_images_android(
    mp4_path: &Path,
    indices:  &[usize],
    out_dir:  &Path,
) -> Result<Vec<PathBuf>> {
    // This calls com.splats.app.FrameExtractorHelper.extractFrames(path, indices[], outDir)
    // The Java implementation is in android/FrameExtractorHelper.java
    use jni::{objects::{JObject, JString, JValue}, JavaVM};

    let path_str = mp4_path.to_str().context("non-UTF8 path")?;
    let out_str  = out_dir.to_str().context("non-UTF8 out_dir")?;

    let jvm = brush_app::android_jvm().context("no JVM")?;
    let env = jvm.attach_current_thread().context("JNI attach")?;
    let cls = env.find_class("com/splats/app/FrameExtractorHelper")?;

    // Build int[] of frame indices
    let j_indices = env.new_int_array(indices.len() as i32)?;
    let indices_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    env.set_int_array_region(j_indices, 0, &indices_i32)?;

    let j_path: JString = env.new_string(path_str)?.into();
    let j_out:  JString = env.new_string(out_str)?.into();

    env.call_static_method(
        cls, "extractFrames",
        "(Ljava/lang/String;[ILjava/lang/String;)V",
        &[JValue::Object(j_path.into()),
          JValue::Object(j_indices.into()),
          JValue::Object(j_out.into())],
    ).context("extractFrames JNI call")?;

    Ok(indices.iter()
        .map(|&i| out_dir.join(format!("frame_{:04}.jpg", i)))
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Vec3};

    fn make_scene() -> BrushScene {
        BrushScene {
            cameras: vec![
                BrushCamera {
                    width: 1920, height: 1080,
                    fx: 1462.0, fy: 1462.0, cx: 960.0, cy: 540.0,
                    world_from_cam_r: Mat3::IDENTITY,
                    world_from_cam_t: Vec3::ZERO,
                    image_path: PathBuf::from("images/frame_0000.jpg"),
                },
                BrushCamera {
                    width: 1920, height: 1080,
                    fx: 1462.0, fy: 1462.0, cx: 960.0, cy: 540.0,
                    world_from_cam_r: Mat3::IDENTITY,
                    world_from_cam_t: Vec3::new(5.0, 0.0, 0.0),
                    image_path: PathBuf::from("images/frame_0003.jpg"),
                },
            ],
            points: vec![
                SparsePoint { position: Vec3::new(2.5, 0.0, 10.0), color: [200, 150, 100] },
                SparsePoint { position: Vec3::new(3.0, 1.0, 12.0), color: [100, 200, 150] },
            ],
        }
    }

    #[test]
    fn test_write_colmap_creates_files() {
        let tmp = std::env::temp_dir().join("brush_test_colmap");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let scene = make_scene();
        write_colmap(&tmp, &scene).unwrap();

        let sparse = tmp.join("sparse/0");
        assert!(sparse.join("cameras.txt").exists());
        assert!(sparse.join("images.txt").exists());
        assert!(sparse.join("points3D.txt").exists());

        let cam_txt = std::fs::read_to_string(sparse.join("cameras.txt")).unwrap();
        assert!(cam_txt.contains("PINHOLE"));
        assert!(cam_txt.contains("1920"));

        let img_txt = std::fs::read_to_string(sparse.join("images.txt")).unwrap();
        assert!(img_txt.contains("frame_0000.jpg"));
        assert!(img_txt.contains("frame_0003.jpg"));

        let pts_txt = std::fs::read_to_string(sparse.join("points3D.txt")).unwrap();
        assert!(pts_txt.contains("10.000000"));

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_write_init_ply_header() {
        let tmp = std::env::temp_dir().join("brush_test_ply");
        std::fs::create_dir_all(&tmp).unwrap();
        let scene = make_scene();
        write_init_ply(&tmp, &scene.points).unwrap();

        let content = std::fs::read_to_string(tmp.join("init.ply")).unwrap();
        assert!(content.starts_with("ply"));
        assert!(content.contains("element vertex 2"));

        std::fs::remove_dir_all(&tmp).ok();
    }
}
