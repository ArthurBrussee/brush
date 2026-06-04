use brush_render::AlphaMode;
use brush_vfs::BrushVfs;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::io::AsyncReadExt;

#[derive(Clone, Debug)]
pub struct LoadImage {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
    mask_path: Option<PathBuf>,
    max_resolution: u32,
    alpha_mode: AlphaMode,
    scale: f32,
}

impl PartialEq for LoadImage {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.mask_path == other.mask_path
            && self.max_resolution == other.max_resolution
            && self.scale == other.scale
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
            scale: 1.0,
        }
    }

    pub async fn load(&self) -> image::ImageResult<DynamicImage> {
        let mut img_bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut img_bytes)
            .await?;
        let mut img = decode_with_cap(&img_bytes, &self.path, self.max_resolution)?;

        // Copy over mask.
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

        let max = self.max_resolution;
        let cap = max as f32 / img.width().max(img.height()).max(max) as f32;
        let scale = (cap * self.scale).min(1.0);
        if scale < 1.0 {
            let new_w = (img.width() as f32 * scale).max(1.0) as u32;
            let new_h = (img.height() as f32 * scale).max(1.0) as u32;
            Ok(img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3))
        } else {
            Ok(img)
        }
    }

    /// Read just the image dimensions from the file header, without decoding
    /// the pixels. Much cheaper than `load()` when only the size is needed
    /// (e.g. formats that carry intrinsics but not image dimensions).
    pub async fn dimensions(&self) -> image::ImageResult<(u32, u32)> {
        let mut img_bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut img_bytes)
            .await?;
        image::ImageReader::new(Cursor::new(&img_bytes))
            .with_guessed_format()?
            .into_dimensions()
    }

    pub fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_max_resolution(mut self, max_resolution: u32) -> Self {
        self.max_resolution = max_resolution;
        self
    }

    pub fn img_name(&self) -> String {
        Path::new(&self.path)
            .file_name()
            .expect("No file name for eval view.")
            .to_string_lossy()
            .to_string()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Decode `bytes`, hinting `jpeg-decoder`'s IDCT scaler to land at or just
/// above `max_resolution` on the long edge for JPEG inputs — that cuts decode
/// cost by ~4-16× on oversized source images. Falls back to `image` for
/// non-JPEG files and for JPEG pixel formats we don't unpack directly.
fn decode_with_cap(
    bytes: &[u8],
    path: &Path,
    max_resolution: u32,
) -> image::ImageResult<DynamicImage> {
    let is_jpeg = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"));
    if is_jpeg && let Some(img) = decode_jpeg_scaled(bytes, max_resolution) {
        return Ok(img);
    }
    image::load_from_memory(bytes)
}

fn decode_jpeg_scaled(bytes: &[u8], max_resolution: u32) -> Option<DynamicImage> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    let target = max_resolution.min(u16::MAX as u32) as u16;
    decoder.scale(target, target).ok()?;
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let w = info.width as u32;
    let h = info.height as u32;
    match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => {
            ImageBuffer::from_raw(w, h, pixels).map(DynamicImage::ImageRgb8)
        }
        jpeg_decoder::PixelFormat::L8 => {
            ImageBuffer::from_raw(w, h, pixels).map(DynamicImage::ImageLuma8)
        }
        // CMYK32 / L16 are rare in photogrammetry data; fall back to image crate.
        _ => None,
    }
}
