use brush_render::AlphaMode;
use brush_vfs::BrushVfs;
use image::{DynamicImage, GrayImage, ImageBuffer};
use std::{
    io::{self, Cursor},
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
    invert_mask: bool,
    scale: f32,
}

impl PartialEq for LoadImage {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.mask_path == other.mask_path
            && self.invert_mask == other.invert_mask
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
        invert_mask: bool,
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
            invert_mask,
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
            let mask_img = image::load_from_memory(&mask_bytes)?;

            // Only one channel of the mask ever reaches the alpha channel, so
            // reduce it to 8bpp before doing any work on it. A grayscale mask
            // (the usual case) then costs no conversion at all, and the resize
            // below runs over one channel instead of three or four.
            let mut mask = if mask_img.color().has_alpha() {
                let rgba = mask_img.into_rgba8();
                let (w, h) = rgba.dimensions();
                let alpha = rgba.pixels().map(|p| p[3]).collect();
                GrayImage::from_raw(w, h, alpha).expect("one byte per pixel")
            } else {
                mask_img.into_luma8()
            };

            // Resize mask image if needed. This is allowed to squash the mask.
            if mask.dimensions() != masked_img.dimensions() {
                mask = image::imageops::resize(
                    &mask,
                    masked_img.width(),
                    masked_img.height(),
                    image::imageops::FilterType::Triangle,
                );
            }

            for (pixel, mask_pixel) in masked_img.pixels_mut().zip(mask.pixels()) {
                pixel[3] = if self.invert_mask {
                    255 - mask_pixel[0]
                } else {
                    mask_pixel[0]
                };
            }

            img = masked_img.into();
        }

        let scale = self.output_scale(img.width(), img.height());
        if scale < 1.0 {
            let new_w = (img.width() as f32 * scale).max(1.0) as u32;
            let new_h = (img.height() as f32 * scale).max(1.0) as u32;
            Ok(img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3))
        } else {
            Ok(img)
        }
    }

    /// Factor `load()` applies to a source of size `w`x`h`: the long edge is
    /// capped to `max_resolution` and multiplied by `scale`.
    fn output_scale(&self, w: u32, h: u32) -> f32 {
        let max = self.max_resolution;
        let cap = max as f32 / w.max(h).max(max) as f32;
        (cap * self.scale).min(1.0)
    }

    /// Read just the image dimensions from the file header, without decoding
    /// the pixels. Much cheaper than `load()` when only the size is needed
    /// (e.g. formats that carry intrinsics but not image dimensions).
    ///
    /// Reads the file in chunks and stops as soon as the header parses, so for
    /// typical formats only the first chunk is fetched rather than the whole
    /// (potentially many-MB) file. A truncated prefix only ever fails to parse
    /// (the dimension fields are reported once fully present), so a partial
    /// buffer can't yield wrong dimensions.
    pub async fn dimensions(&self) -> image::ImageResult<(u32, u32)> {
        let mut reader = self.vfs.reader_at_path(&self.path).await?;
        let dims = brush_vfs::read_until_parsed(&mut reader, 64 * 1024, |bytes| {
            image::ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .ok()
                .and_then(|r| r.into_dimensions().ok())
        })
        .await?;
        dims.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("could not determine image dimensions for {:?}", self.path),
            )
            .into()
        })
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

    /// Dataset-relative path flattened with `-` for collision-free disk names.
    ///
    /// COLMAP multi-camera layouts often share basenames across per-camera
    /// folders (`y000/station_0000.jpg`, `y045/station_0000.jpg`). Using only
    /// [`Self::img_name`] for `--eval-save-to-disk` would overwrite those
    /// renders; this keeps the directory components in the output filename.
    pub fn flattened_path_name(&self) -> String {
        flattened_path_name(&self.path)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Flatten a dataset-relative path into a single filename component.
fn flattened_path_name(path: &Path) -> String {
    let parts: Vec<_> = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy()),
            _ => None,
        })
        .collect();
    if parts.is_empty() {
        path.to_string_lossy().into_owned()
    } else {
        parts.join("-")
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

#[cfg(test)]
mod flattened_path_tests {
    use super::flattened_path_name;
    use std::path::Path;

    #[test]
    fn basename_only_unchanged() {
        assert_eq!(flattened_path_name(Path::new("station_0000.jpg")), "station_0000.jpg");
    }

    #[test]
    fn preserves_camera_subdir_components() {
        assert_eq!(
            flattened_path_name(Path::new("images/y000/station_0000.jpg")),
            "images-y000-station_0000.jpg"
        );
        assert_eq!(
            flattened_path_name(Path::new("images/y045/station_0000.jpg")),
            "images-y045-station_0000.jpg"
        );
    }

    #[test]
    fn distinct_camera_folders_do_not_collide() {
        let a = flattened_path_name(Path::new("y000/station_0000.jpg"));
        let b = flattened_path_name(Path::new("y045/station_0000.jpg"));
        assert_ne!(a, b);
        assert_eq!(a, "y000-station_0000.jpg");
        assert_eq!(b, "y045-station_0000.jpg");
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::LoadImage;
    use brush_vfs::BrushVfs;
    use image::{DynamicImage, GrayImage, Rgb, RgbImage, RgbaImage};
    use std::sync::Arc;

    /// Write an image + mask pair to a temp dir and load them back.
    async fn load_with_mask(mask: DynamicImage, invert: bool) -> RgbaImage {
        let dir = tempfile::tempdir().expect("temp dir");
        RgbImage::from_pixel(4, 2, Rgb([10, 20, 30]))
            .save(dir.path().join("img.png"))
            .expect("save image");
        mask.save(dir.path().join("mask.png")).expect("save mask");

        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.expect("vfs"));
        LoadImage::new(
            vfs,
            "img.png".into(),
            Some("mask.png".into()),
            1920,
            None,
            invert,
        )
        .load()
        .await
        .expect("load image")
        .into_rgba8()
    }

    #[tokio::test]
    async fn mask_becomes_alpha() {
        let mask = GrayImage::from_raw(4, 2, (0..8).map(|i| i * 30).collect()).expect("mask");
        let img = load_with_mask(mask.into(), false).await;
        let alpha: Vec<u8> = img.pixels().map(|p| p[3]).collect();
        assert_eq!(alpha, (0..8).map(|i| i * 30).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn inverted_mask_flips_alpha() {
        let mask = GrayImage::from_raw(4, 2, (0..8).map(|i| i * 30).collect()).expect("mask");
        let img = load_with_mask(mask.into(), true).await;
        let alpha: Vec<u8> = img.pixels().map(|p| p[3]).collect();
        assert_eq!(alpha, (0..8).map(|i| 255 - i * 30).collect::<Vec<_>>());
    }
}
