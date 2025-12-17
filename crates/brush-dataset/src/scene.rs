use brush_render::{AlphaMode, bounding_box::BoundingBox, camera::Camera};
use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use glam::{Affine3A, Vec3, vec3};
use image::{DynamicImage, GenericImageView};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::io::AsyncReadExt;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ViewType {
    Train,
    Eval,
    Test,
}

#[derive(Clone, Debug)]
pub struct LoadImage {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
    mask_path: Option<PathBuf>,
    max_resolution: u32,
    alpha_mode: AlphaMode,
}

impl PartialEq for LoadImage {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.mask_path == other.mask_path
            && self.max_resolution == other.max_resolution
    }
}

impl LoadImage {
    pub fn new(
        vfs: Arc<BrushVfs>,
        path: PathBuf,
        mask_path: Option<PathBuf>,
        max_resolution: u32,
        override_alpha_mode: Option<AlphaMode>,
    ) -> Self {
        let alpha_mode = override_alpha_mode.unwrap_or_else(|| {
            if mask_path.is_some() {
                AlphaMode::Masked
            } else {
                AlphaMode::Transparent
            }
        });

        Self {
            vfs,
            path,
            mask_path,
            max_resolution,
            alpha_mode,
        }
    }

    pub async fn load(&self) -> image::ImageResult<DynamicImage> {
        let mut img_bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut img_bytes)
            .await?;
        let mut img = image::load_from_memory(&img_bytes)?;

        // Copy over mask.
        // TODO: Interleave this work better & speed things up here.
        if let Some(mask_path) = &self.mask_path {
            // Add in alpha channel if needed to the image to copy the mask into.
            let mut masked_img = img.into_rgba8();
            let mut mask_bytes = vec![];
            self.vfs
                .reader_at_path(mask_path)
                .await?
                .read_to_end(&mut mask_bytes)
                .await?;
            let mut mask_img = image::load_from_memory(&mask_bytes)?;

            // Resize mask image if needed. This is allowed to squash the mask.
            if mask_img.dimensions() != masked_img.dimensions() {
                mask_img = mask_img.resize_exact(
                    masked_img.width(),
                    masked_img.height(),
                    image::imageops::FilterType::Triangle,
                );
            }

            if mask_img.color().has_alpha() {
                let mask_img = mask_img.into_rgba8();
                for (pixel, mask_pixel) in masked_img.pixels_mut().zip(mask_img.pixels()) {
                    pixel[3] = mask_pixel[3];
                }
            } else {
                let mask_img = mask_img.into_rgb8();
                for (pixel, mask_pixel) in masked_img.pixels_mut().zip(mask_img.pixels()) {
                    pixel[3] = mask_pixel[0];
                }
            }

            img = masked_img.into();
        }
        if img.width() <= self.max_resolution && img.height() <= self.max_resolution {
            return Ok(img);
        }
        Ok(img.resize(
            self.max_resolution,
            self.max_resolution,
            image::imageops::FilterType::Triangle,
        ))
    }

    pub fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    pub fn img_name(&self) -> String {
        Path::new(&self.path)
            .file_name()
            .expect("No file name for eval view.")
            .to_string_lossy()
            .to_string()
    }
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

pub fn sample_to_tensor_data(sample: DynamicImage) -> TensorData {
    let _span = tracing::trace_span!("sample_to_tensor").entered();

    let (w, h) = (sample.width(), sample.height());
    tracing::trace_span!("Img to vec").in_scope(|| {
        if sample.color().has_alpha() {
            TensorData::new(
                sample.into_rgba32f().into_vec(),
                [h as usize, w as usize, 4],
            )
        } else {
            TensorData::new(sample.into_rgb32f().into_vec(), [h as usize, w as usize, 3])
        }
    })
}

#[derive(Clone, Debug)]
pub struct SceneBatch {
    pub img_tensor: TensorData,
    pub alpha_mode: AlphaMode,
    pub camera: Camera,
}

impl SceneBatch {
    pub fn has_alpha(&self) -> bool {
        self.img_tensor.shape[2] == 4
    }
}
