use anyhow::{Context, Result};
use brush_render::{bounding_box::BoundingBox, camera::Camera};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use glam::{Affine3A, Vec3, vec3};
use image::{ColorType, DynamicImage, ImageDecoder, ImageReader};
use std::{path::PathBuf, sync::Arc};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::{brush_vfs::BrushVfs, clamp_img_to_max_size};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ViewType {
    Train,
    Eval,
    Test,
}

pub struct LoadImage {
    pub vfs: Arc<BrushVfs>,
    pub path: PathBuf,
    pub mask_path: Option<PathBuf>,
    color: image::ColorType,
    size: glam::UVec2,
    max_resolution: u32,
}

/// Gets the dimensions of an image from an [`AsyncRead`] source
pub async fn get_image_data<R>(reader: &mut R) -> Result<(u32, u32, ColorType)>
where
    R: AsyncRead + Unpin,
{
    // 4kb should be PLENTY to guess the size of the image (at least for jpg and png). Don't need to read the full image.
    let mut buffer = Vec::with_capacity(4096);
    let mut temp_buf = [0u8; 4096];

    let n = reader.read(&mut temp_buf).await?;
    buffer.extend_from_slice(&temp_buf[..n]);

    // Now get a normal cursor.
    let cursor = std::io::Cursor::new(buffer);
    let dimensions = ImageReader::new(cursor.clone())
        .with_guessed_format()
        .context("Failed to guess format")?
        .into_dimensions()
        .context("Failed to read dimensions")?;
    let decoder = ImageReader::new(cursor)
        .with_guessed_format()
        .context("Failed to guess format")?
        .into_decoder()
        .context("Failed to read color type")?;

    Ok((dimensions.0, dimensions.1, decoder.color_type()))
}

impl LoadImage {
    pub async fn new(
        vfs: Arc<BrushVfs>,
        path: PathBuf,
        mask_path: Option<PathBuf>,
        max_resolution: u32,
    ) -> Result<Self> {
        let reader = &mut vfs.reader_at_path(&path).await?;
        let data = get_image_data(reader).await?;

        Ok(Self {
            vfs,
            path,
            mask_path,
            max_resolution,
            size: glam::uvec2(data.0, data.1),
            color: data.2,
        })
    }

    pub fn color(&self) -> image::ColorType {
        self.color
    }

    pub fn width(&self) -> u32 {
        self.size.x
    }

    pub fn height(&self) -> u32 {
        self.size.y
    }

    pub async fn load(&self) -> Result<DynamicImage> {
        let mut img_bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut img_bytes)
            .await?;
        let mut img = image::load_from_memory(&img_bytes)?;

        // Copy over mask
        if let Some(mask_path) = &self.mask_path {
            let mut mask_bytes = vec![];
            self.vfs
                .reader_at_path(mask_path)
                .await?
                .read_to_end(&mut mask_bytes)
                .await?;
            let mask_img = image::load_from_memory(&mask_bytes)?;

            let mut img_masked = img.to_rgba8();

            if mask_img.color().has_alpha() {
                let mask_img = mask_img.to_rgba8();
                for (buf, mask) in img_masked.pixels_mut().zip(mask_img.pixels()) {
                    buf[3] = mask[0];
                }
            } else {
                let mask_img = mask_img.grayscale().to_rgb8();
                for (buf, mask) in img_masked.pixels_mut().zip(mask_img.pixels()) {
                    buf[3] = mask[0];
                }
            }

            img = img_masked.into();
        }

        let img = clamp_img_to_max_size(img, self.max_resolution);
        Ok(img)
    }

    pub fn is_masked(&self) -> bool {
        self.mask_path.is_some()
    }
}

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

fn find_two_smallest(v: Vec3) -> (f32, f32) {
    let mut arr = v.to_array();
    arr.sort_by(|a, b| a.partial_cmp(b).expect("NaN"));
    (arr[0], arr[1])
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

    pub fn estimate_extent(&self) -> Option<f32> {
        if self.views.len() < 5 {
            None
        } else {
            // TODO: This is really sensitive to outliers.
            let bounds = self.bounds();
            let smallest = find_two_smallest(bounds.extent * 2.0);
            Some(smallest.0.hypot(smallest.1))
        }
    }
}

// Converts an image to a train sample. The tensor will be a floating point image with a [0, 1] image.
//
// This assume the input image has un-premultiplied alpha, whereas the output has pre-multiplied alpha.
pub fn view_to_sample_image(image: Arc<DynamicImage>, alpha_is_mask: bool) -> Arc<DynamicImage> {
    if image.color().has_alpha() && !alpha_is_mask {
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
        Arc::new(DynamicImage::ImageRgba8(rgba_bytes))
    } else {
        image
    }
}

pub fn sample_to_tensor<B: Backend>(sample: &DynamicImage, device: &B::Device) -> Tensor<B, 3> {
    let (w, h) = (sample.width(), sample.height());
    let data = if sample.color().has_alpha() {
        TensorData::new(sample.to_rgba32f().into_vec(), [h as usize, w as usize, 4])
    } else {
        TensorData::new(sample.to_rgb32f().into_vec(), [h as usize, w as usize, 3])
    };
    Tensor::from_data(data, device)
}

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub img_tensor: Tensor<B, 3>,
    pub alpha_is_mask: bool,
    pub camera: Camera,
}

impl<B: Backend> SceneBatch<B> {
    pub fn has_alpha(&self) -> bool {
        self.img_tensor.shape().dims[2] == 4
    }
}
