use brush_render::{bounding_box::BoundingBox, camera::Camera};
use glam::{vec3, Vec3};
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ViewType {
    Train,
    Eval,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewImageType {
    Alpha,
    Masked,
}

#[derive(Debug, Clone)]
pub struct SceneView {
    pub name: String,
    pub camera: Camera,
    pub image: Arc<image::DynamicImage>,
    pub img_type: ViewImageType,
}

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Debug, Clone)]
pub struct Scene {
    pub views: Arc<Vec<SceneView>>,
}

fn camera_distance_score(cam: &Camera, reference: &Camera) -> f32 {
    let mut score = 0.0;
    for off_x in [-1.0, 0.0, 1.0] {
        for off_y in [-1.0, 0.0, 1.0] {
            let offset = vec3(off_x, off_y, 1.0);
            let cam_pos = cam.position + cam.rotation * offset;
            let ref_pos = reference.position + reference.rotation * offset;
            score += (cam_pos - ref_pos).length();
        }
    }
    score
}

impl Scene {
    pub fn new(views: Vec<SceneView>) -> Self {
        Self {
            views: Arc::new(views),
        }
    }

    // Returns the extent of the cameras in the scene.
    pub fn bounds(&self) -> BoundingBox {
        self.adjusted_bounds(0.0, 0.0)
    }

    // Returns the extent of the cameras in the scene, taking into account
    // the near and far plane of the cameras.
    pub fn adjusted_bounds(&self, cam_near: f32, cam_far: f32) -> BoundingBox {
        let (min, max) = self.views.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), view| {
                let cam = &view.camera;
                let pos1 = cam.position + cam.rotation * Vec3::Z * cam_near;
                let pos2 = cam.position + cam.rotation * Vec3::Z * cam_far;
                (min.min(pos1).min(pos2), max.max(pos1).max(pos2))
            },
        );
        BoundingBox::from_min_max(min, max)
    }

    pub fn get_nearest_view(&self, reference: &Camera) -> Option<usize> {
        self.views
            .iter()
            .enumerate() // This will give us (index, view) pairs
            .min_by(|(_, a), (_, b)| {
                let score_a = camera_distance_score(&a.camera, reference);
                let score_b = camera_distance_score(&b.camera, reference);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index) // We return the index instead of the camera
    }
}
