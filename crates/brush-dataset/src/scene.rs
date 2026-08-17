use brush_render::{AlphaMode, bounding_box::BoundingBox, camera::Camera};
use burn::tensor::TensorData;
use glam::{Affine3A, Vec3, vec3};
use image::DynamicImage;
use std::sync::Arc;

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

/// Convert a loaded view into the GPU-side packed representation: `[H, W]`
/// u32, each entry packing `[r8 g8 b8 a8]`. Images without alpha get
/// `a = 255` (fully opaque) so the kernel always sees a valid alpha byte.
/// Returns `(packed, has_alpha)` so the trainer knows whether to apply
/// alpha-dependent loss terms.
///
/// Transparent-alpha views arrive un-premultiplied and leave premultiplied.
/// That happens here rather than in a pass of its own: premultiplying,
/// widening to RGBA and packing all read the same pixel, so they cost one
/// walk over the image and one allocation between them.
pub fn view_to_packed_data(image: DynamicImage, alpha_mode: AlphaMode) -> (TensorData, bool) {
    let _span = tracing::trace_span!("view_to_packed").entered();
    let (w, h) = (image.width(), image.height());
    let has_alpha = image.color().has_alpha();
    // Premultiplication is what `Transparent` means; a mask multiplies the
    // loss instead and must keep its colours intact.
    let premultiply = has_alpha && alpha_mode == AlphaMode::Transparent;

    // Pack as `[i32]` little-endian (same bit pattern as u32; i32 because the
    // burn dispatch backend's default int dtype is i32 and refuses to cast
    // u32 values >= 2^31). The kernel reads the same way (`val & 0xff` is
    // `r`, `>> 24` is `a`) — the signedness only affects the host-side
    // TensorData metadata, not the GPU bytes.
    let packed: Vec<i32> = match image {
        DynamicImage::ImageRgb8(img) => img
            .pixels()
            .map(|p| i32::from_le_bytes([p[0], p[1], p[2], 255]))
            .collect(),
        DynamicImage::ImageRgba8(img) => pack_rgba(&img, premultiply),
        // 16-bit, luma and cmyk sources are rare; normalise them first.
        other => pack_rgba(&other.into_rgba8(), premultiply),
    };

    (TensorData::new(packed, [h as usize, w as usize]), has_alpha)
}

fn pack_rgba(img: &image::RgbaImage, premultiply: bool) -> Vec<i32> {
    img.pixels()
        .map(|p| {
            let [r, g, b, a] = p.0;
            if premultiply {
                // Multiply in byte space, before anything converts to float.
                let mul = |c: u8| ((c as u16 * a as u16 + 127) / 255) as u8;
                i32::from_le_bytes([mul(r), mul(g), mul(b), a])
            } else {
                i32::from_le_bytes([r, g, b, a])
            }
        })
        .collect()
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
}

impl SceneBatch {
    /// Host bytes the packed image occupies.
    pub fn packed_bytes(&self) -> u64 {
        self.img_packed
            .as_bytes()
            .len()
            .try_into()
            .expect("shouldn't exceed ~18 Exabytes...")
    }

    pub fn img_size(&self) -> [usize; 2] {
        [self.img_packed.shape[0], self.img_packed.shape[1]]
    }
}

#[cfg(test)]
mod tests {
    use super::view_to_packed_data;
    use brush_render::AlphaMode;
    use image::{DynamicImage, ImageBuffer, RgbImage, RgbaImage};

    #[test]
    fn packs_rgba_samples_without_changing_channels() {
        let image =
            RgbaImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).expect("valid RGBA image");

        let (packed, has_alpha) =
            view_to_packed_data(DynamicImage::ImageRgba8(image), AlphaMode::Masked);

        assert!(has_alpha);
        assert_eq!(packed.shape.dims(), [1, 2]);
        assert_eq!(
            packed.as_slice::<i32>().expect("i32 tensor"),
            &[0x0403_0201, 0x0807_0605]
        );
    }

    #[test]
    fn fills_missing_alpha_with_opaque_for_rgb_samples() {
        let image: RgbImage =
            ImageBuffer::from_raw(2, 1, vec![9, 10, 11, 12, 13, 14]).expect("valid RGB image");

        let (packed, has_alpha) =
            view_to_packed_data(DynamicImage::ImageRgb8(image), AlphaMode::Transparent);

        assert!(!has_alpha);
        assert_eq!(packed.shape.dims(), [1, 2]);
        assert_eq!(
            packed.as_slice::<i32>().expect("i32 tensor"),
            &[0xff0b_0a09_u32 as i32, 0xff0e_0d0c_u32 as i32]
        );
    }
}
