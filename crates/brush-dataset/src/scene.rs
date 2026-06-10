use brush_render::{AlphaMode, bounding_box::BoundingBox, camera::Camera};
use burn::tensor::TensorData;
use glam::{Affine3A, Vec3, vec3};
use image::DynamicImage;
use std::sync::Arc;

pub use crate::load_depth::LoadDepth;
pub use crate::load_image::LoadImage;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ViewType {
    Train,
    Eval,
    Test,
}

#[derive(Clone)]
pub struct SceneView {
    pub image: LoadImage,
    pub camera: Camera,
    /// Optional per-view metric `LiDAR` depth + confidence (for depth supervision).
    pub depth: Option<LoadDepth>,
}

// Encapsulates a multi-view scene including cameras and the splats.
// Also provides methods for checkpointing the training process.
#[derive(Clone)]
pub struct Scene {
    pub views: Arc<Vec<SceneView>>,
}

fn camera_distance_penalty(cam_local_to_world: Affine3A, reference: Affine3A) -> f32 {
    let mut penalty = 0.0;
    for off_x in [-1.0, 0.0, 1.0] {
        for off_y in [-1.0, 0.0, 1.0] {
            let offset = vec3(off_x, off_y, 1.0);
            let cam_pos = cam_local_to_world.transform_point3(offset);
            let ref_pos = reference.transform_point3(offset);
            penalty += (cam_pos - ref_pos).length();
        }
    }
    penalty
}

impl Scene {
    pub fn new(views: Vec<SceneView>) -> Self {
        Self {
            views: Arc::new(views),
        }
    }

    // Returns the extent of the cameras in the scene.
    pub fn bounds(&self) -> BoundingBox {
        let (min, max) = self.views.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), view| {
                let cam = &view.camera;
                (min.min(cam.position), max.max(cam.position))
            },
        );
        BoundingBox::from_min_max(min, max)
    }

    pub fn with_image_scale(self, scale: f32) -> Self {
        let views = Arc::unwrap_or_clone(self.views)
            .into_iter()
            .map(|v| SceneView {
                image: v.image.with_scale(scale),
                camera: v.camera,
                depth: v.depth,
            })
            .collect();
        Self::new(views)
    }

    pub fn get_nearest_view(&self, reference: Affine3A) -> Option<usize> {
        self.views
            .iter()
            .enumerate() // This will give us (index, view) pairs
            .min_by(|(_, a), (_, b)| {
                let score_a = camera_distance_penalty(a.camera.local_to_world(), reference);
                let score_b = camera_distance_penalty(b.camera.local_to_world(), reference);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index) // We return the index instead of the camera
    }
}

// Converts an image to a train sample. The tensor will be a floating point image with a [0, 1] image.
//
// This assume the input image has un-premultiplied alpha, whereas the output has pre-multiplied alpha.
pub fn view_to_sample_image(image: DynamicImage, alpha_mode: AlphaMode) -> DynamicImage {
    if image.color().has_alpha() && alpha_mode == AlphaMode::Transparent {
        let mut rgba_bytes = image.to_rgba8();
        // Assume image has un-multiplied alpha and convert it to pre-multiplied.
        // Perform multiplication in byte space before converting to float.
        for pixel in rgba_bytes.chunks_exact_mut(4) {
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];
            let a = pixel[3];

            pixel[0] = ((r as u16 * a as u16 + 127) / 255) as u8;
            pixel[1] = ((g as u16 * a as u16 + 127) / 255) as u8;
            pixel[2] = ((b as u16 * a as u16 + 127) / 255) as u8;
            pixel[3] = a;
        }
        DynamicImage::ImageRgba8(rgba_bytes)
    } else {
        image
    }
}

/// Convert a sample into the GPU-side packed representation: `[H, W]` u32,
/// each entry packing `[r8 g8 b8 a8]`. Images without alpha get `a = 255`
/// (fully opaque) so the kernel always sees a valid alpha byte. Returns
/// `(packed, has_alpha)` so the trainer knows whether to apply
/// alpha-dependent loss terms.
pub fn sample_to_packed_data(sample: DynamicImage) -> (TensorData, bool) {
    let _span = tracing::trace_span!("sample_to_packed").entered();
    let (w, h) = (sample.width(), sample.height());
    let has_alpha = sample.color().has_alpha();
    let bytes = if has_alpha {
        sample.into_rgba8().into_vec()
    } else {
        let rgb = sample.into_rgb8().into_vec();
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for px in rgb.chunks_exact(3) {
            bytes.extend_from_slice(px);
            bytes.push(255);
        }
        bytes
    };
    // Reinterpret the `[r g b a r g b a ...]` byte stream as `[i32]` little-endian
    // (i32 bit-pattern same as the underlying u32; we use i32 because the burn
    // dispatch backend's default int dtype is i32 and refuses to cast u32
    // values >= 2^31). The kernel reads the same way (`val & 0xff` is `r`,
    // `>> 24` is `a`) — the signedness only affects the host-side TensorData
    // metadata, not the GPU bytes.
    let packed: Vec<i32> = bytemuck::pod_collect_to_vec(&bytes);
    (TensorData::new(packed, [h as usize, w as usize]), has_alpha)
}

/// Per-view metric depth supervision: `z`-depth (metres) + `ARKit` confidence,
/// each `[H_d, W_d, 1]` f32 at the depth's native (low) resolution.
#[derive(Clone, Debug)]
pub struct DepthSample {
    pub depth: TensorData,
    pub conf: TensorData,
}

#[derive(Clone, Debug)]
pub struct SceneBatch {
    /// `[H, W]` u32, each entry packs `[r g b a]` u8.
    pub img_packed: TensorData,
    /// True when the source image had an alpha channel that the trainer
    /// should consume (mask weight, alpha-matching loss, bg compositing).
    pub has_alpha: bool,
    pub alpha_mode: AlphaMode,
    pub camera: Camera,
    pub depth: Option<DepthSample>,
}

impl SceneBatch {
    pub fn img_size(&self) -> [usize; 2] {
        [self.img_packed.shape[0], self.img_packed.shape[1]]
    }
}
