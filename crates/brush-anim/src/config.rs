use brush_render::camera::{Camera, focal_to_fov, fov_to_focal};
use brush_render::kernels::camera_model::CameraModel;
use glam::{Affine3A, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// How camera poses are interpolated between keyframes.
#[derive(Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Interpolation {
    Linear,
    Smoothstep,
    #[default]
    CatmullRom,
}

impl Interpolation {
    pub const ALL: [Self; 3] = [Self::Linear, Self::Smoothstep, Self::CatmullRom];

    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Smoothstep => "Smoothstep",
            Self::CatmullRom => "Catmull-Rom",
        }
    }
}

/// A camera pose pinned to a frame on the timeline.
#[derive(Clone, Serialize, Deserialize)]
pub struct Keyframe {
    pub frame: usize,
    pub position: Vec3,
    pub rotation: Quat,
}

/// A complete, serializable animation: the timeline, the keyframe poses, and
/// the viewer state (FOV, model transform, background) needed to reproduce the
/// rendered view outside the app.
#[derive(Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    pub num_frames: usize,
    pub fps: usize,
    pub interpolation: Interpolation,
    pub keyframes: Vec<Keyframe>,

    /// Field of view captured from the viewer camera.
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: Vec2,
    /// Splat model-to-world transform from the viewer.
    pub model_local_to_world: Affine3A,
    /// Background color and splat scale from the viewer.
    pub background: Vec3,
    pub splat_scale: Option<f32>,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            num_frames: 120,
            fps: 30,
            interpolation: Interpolation::default(),
            keyframes: Vec::new(),
            fov_x: 0.8,
            fov_y: 0.8,
            center_uv: Vec2::splat(0.5),
            model_local_to_world: Affine3A::IDENTITY,
            background: Vec3::ZERO,
            splat_scale: None,
        }
    }
}

impl AnimationConfig {
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Smallest frame count that still fits every keyframe.
    pub fn min_frames(&self) -> usize {
        self.keyframes
            .iter()
            .map(|k| k.frame)
            .max()
            .map_or(1, |f| f + 1)
    }

    /// Insert a keyframe at `frame`, replacing any existing keyframe there.
    pub fn set_keyframe(&mut self, frame: usize, position: Vec3, rotation: Quat) {
        if let Some(existing) = self.keyframes.iter_mut().find(|k| k.frame == frame) {
            existing.position = position;
            existing.rotation = rotation;
        } else {
            self.keyframes.push(Keyframe {
                frame,
                position,
                rotation,
            });
        }
    }

    /// The interpolated camera pose at `frame`, using the selected method. The
    /// keyframes form a closed loop, so frames wrap smoothly from the last
    /// keyframe back to the first. Returns `None` if there are no keyframes.
    pub fn pose_at_frame(&self, frame: usize) -> Option<(Vec3, Quat)> {
        let mut kfs: Vec<&Keyframe> = self.keyframes.iter().collect();
        kfs.sort_by_key(|k| k.frame);
        let n = kfs.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some((kfs[0].position, kfs[0].rotation));
        }

        // Segment `s` runs from kfs[s] to kfs[(s + 1) % n]; `t` is the position
        // within it. Segment n-1 is the wrap from the last keyframe to the first.
        let (s, t) = self.segment_at(&kfs, frame);
        let p1 = kfs[s];
        let p2 = kfs[(s + 1) % n];

        let pose = match self.interpolation {
            Interpolation::Linear => (
                p1.position.lerp(p2.position, t),
                p1.rotation.slerp(p2.rotation, t),
            ),
            Interpolation::Smoothstep => {
                let t = t * t * (3.0 - 2.0 * t);
                (
                    p1.position.lerp(p2.position, t),
                    p1.rotation.slerp(p2.rotation, t),
                )
            }
            Interpolation::CatmullRom => {
                let p0 = kfs[(s + n - 1) % n];
                let p3 = kfs[(s + 2) % n];
                (
                    catmull_rom(p0.position, p1.position, p2.position, p3.position, t),
                    catmull_rom_quat(p0.rotation, p1.rotation, p2.rotation, p3.rotation, t),
                )
            }
        };
        Some(pose)
    }

    /// Finds the loop segment containing `frame` and the `t` within it, given
    /// keyframes sorted by frame.
    fn segment_at(&self, kfs: &[&Keyframe], frame: usize) -> (usize, f32) {
        let n = kfs.len();
        let first = kfs[0].frame;
        let last = kfs[n - 1].frame;

        if (first..last).contains(&frame) {
            let s = (0..n - 1)
                .find(|&s| frame < kfs[s + 1].frame)
                .expect("frame < last");
            let t = (frame - kfs[s].frame) as f32 / (kfs[s + 1].frame - kfs[s].frame) as f32;
            (s, t)
        } else {
            let wrap_len = self.num_frames - (last - first);
            let steps = (frame + self.num_frames - last) % self.num_frames;
            (n - 1, steps as f32 / wrap_len as f32)
        }
    }

    /// Builds one render camera per frame: the stored viewer camera (FOV, model
    /// transform) with the interpolated keyframe pose, fitting the FOV to the
    /// target resolution exactly like the scene view does.
    pub fn render_cameras(&self, width: u32, height: u32) -> Vec<Camera> {
        let template = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            self.fov_x,
            self.fov_y,
            self.center_uv,
            CameraModel::Pinhole,
        );

        (0..self.num_frames)
            .map(|frame| {
                let mut camera = template;
                if let Some((position, rotation)) = self.pose_at_frame(frame) {
                    camera.position = position;
                    camera.rotation = rotation;
                }

                // Fold in the model transform, exactly like the scene view.
                let view_eff = (camera.world_to_local() * self.model_local_to_world).inverse();
                let (_, rotation, position) = view_eff.to_scale_rotation_translation();
                camera.position = position;
                camera.rotation = rotation;

                // Fit the FOV to the target aspect ratio (same logic as the scene).
                let camera_aspect = fov_to_focal(camera.fov_y, 2, &camera.camera_model)
                    / fov_to_focal(camera.fov_x, 2, &camera.camera_model);
                let viewport_aspect = width as f64 / height as f64;
                if viewport_aspect > camera_aspect {
                    let focal_y = fov_to_focal(camera.fov_y, height, &camera.camera_model);
                    camera.fov_x = focal_to_fov(focal_y, width, &camera.camera_model);
                } else {
                    let focal_x = fov_to_focal(camera.fov_x, width, &camera.camera_model);
                    camera.fov_y = focal_to_fov(focal_x, height, &camera.camera_model);
                }
                camera
            })
            .collect()
    }
}

/// Catmull-Rom spline through `p1` and `p2`, using `p0`/`p3` to derive tangents.
fn catmull_rom<V>(p0: V, p1: V, p2: V, p3: V, t: f32) -> V
where
    V: Copy
        + std::ops::Add<Output = V>
        + std::ops::Sub<Output = V>
        + std::ops::Mul<f32, Output = V>,
{
    let t2 = t * t;
    let t3 = t2 * t;
    (p1 * 2.0
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
        + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
        * 0.5
}

/// Catmull-Rom for rotations: aligns the quaternions to one hemisphere (so the
/// spline takes the short way), interpolates component-wise, then renormalizes.
fn catmull_rom_quat(q0: Quat, q1: Quat, q2: Quat, q3: Quat, t: f32) -> Quat {
    let q0 = if q1.dot(q0) < 0.0 { -q0 } else { q0 };
    let q2 = if q1.dot(q2) < 0.0 { -q2 } else { q2 };
    let q3 = if q2.dot(q3) < 0.0 { -q3 } else { q3 };
    let blended = catmull_rom(
        glam::Vec4::from(q0.to_array()),
        glam::Vec4::from(q1.to_array()),
        glam::Vec4::from(q2.to_array()),
        glam::Vec4::from(q3.to_array()),
        t,
    );
    Quat::from_vec4(blended).normalize()
}
