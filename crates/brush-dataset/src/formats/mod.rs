use crate::{Dataset, config::LoadDataseConfig};
use brush_serde::{DeserializeError, SplatMessage, load_splat_from_ply};
use brush_vfs::BrushVfs;
use image::ImageError;
use std::{path::Path, sync::Arc};

pub mod colmap;
pub mod nerfstudio;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("IO error while loading dataset.")]
    Io(#[from] std::io::Error),

    #[error("Error decoding JSON file.")]
    Json(#[from] serde_json::Error),

    #[error("Error decoding camera parameters: {0}")]
    InvalidCamera(String),

    #[error("Error when decoding format: {0}")]
    InvalidFormat(String),

    #[error("Error loading splat data: {0}")]
    PlyError(#[from] DeserializeError),

    #[error("Error loading image in data: {0}")]
    ImageError(#[from] ImageError),
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("Failed to load format.")]
    FormatError(#[from] FormatError),

    #[error("Failed to load initial point cloud.")]
    InitialPointCloudError(#[from] DeserializeError),

    #[error("Format not recognized: Only colmap and nerfstudio json are supported.")]
    FormatNotSupported,
}

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
) -> Result<(Option<SplatMessage>, Dataset), DatasetError> {
    let mut dataset = colmap::load_dataset(vfs.clone(), load_args).await;

    if dataset.is_none() {
        dataset = nerfstudio::read_dataset(vfs.clone(), load_args).await;
    }

    let Some(dataset) = dataset else {
        return Err(DatasetError::FormatNotSupported);
    };

    let (data_splat_init, dataset) = dataset?;

    // If there's an initial ply file, override the init stream with that.
    let mut ply_paths: Vec<_> = vfs.files_with_extension("ply").collect();
    ply_paths.sort();

    let main_ply = ply_paths
        .iter()
        .find(|p| p.file_name().is_some_and(|n| n == "init.ply"))
        .or_else(|| ply_paths.last());

    let init_splat = if let Some(main_ply) = main_ply {
        log::info!("Using ply {main_ply:?} as initial point cloud.");
        let reader = vfs
            .reader_at_path(main_ply)
            .await
            .map_err(DeserializeError)?;
        Some(load_splat_from_ply(reader, load_args.subsample_points).await?)
    } else {
        data_splat_init
    };

    Ok((init_splat, dataset))
}

/// Best-effort detection of the maximum source image resolution (largest image dimension) in a dataset.
///
/// This is used by the UI to set sensible slider ranges before the dataset is fully loaded.
pub async fn detect_max_image_resolution(vfs: Arc<BrushVfs>) -> Option<u32> {
    if let Some(max_res) = colmap::detect_max_image_resolution(vfs.clone()).await {
        return Some(max_res);
    }
    nerfstudio::detect_max_image_resolution(vfs).await
}

fn find_mask_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Option<&'a Path> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");
    let mut search_mask = search_stem.to_owned();
    search_mask.push(".mask");
    let search_mask = &search_mask;

    vfs.iter_files().find(|candidate| {
        // For the target, we don't care about its actual extension. Lets see if either the name or stem matches.
        let Some(stem) = candidate.file_stem() else {
            return false;
        };

        // We have the name of the file a la img.png, and the stem a la img.
        // We now want to accept any of img.png.*, img.*, img.mask.*.
        if stem.eq_ignore_ascii_case(search_name)
            || stem.eq_ignore_ascii_case(search_stem)
            || stem.eq_ignore_ascii_case(search_mask)
        {
            // Find "masks" directory in candidate path
            let masks_idx = candidate
                .components()
                .position(|c| c.as_os_str().eq_ignore_ascii_case("masks"));

            // Check if the image directory path ends with the directory subpath after "masks/"
            // e.g., masks/foo/bar/bla.png should match images/foo/bar/bla.jpeg
            masks_idx.is_some_and(|idx| {
                let candidate_components: Vec<_> = candidate.components().collect();

                // Get directory components only (excluding filename)
                let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
                let mask_dir_subpath =
                    &candidate_components[idx + 1..candidate_components.len() - 1];
                path_dir_components.ends_with(mask_dir_subpath)
            })
        } else {
            false
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn test_find_mask() {
        // Basic matching with same extension
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.png"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/img.png")),
            Some(Path::new("masks/img.png"))
        );
        // Different extensions are ok.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.jpeg"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/img.jpeg")),
            Some(Path::new("masks/img.png"))
        );
    }

    #[test]
    fn test_find_mask_formats() {
        // Test img.png.mask format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo.png"),
            PathBuf::from("masks/foo.png.mask"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/foo.png")),
            Some(Path::new("masks/foo.png.mask"))
        );

        // Test img.mask.png format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/bar.jpeg"),
            PathBuf::from("masks/bar.mask.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/bar.jpeg")),
            Some(Path::new("masks/bar.mask.png"))
        );
    }

    #[test]
    fn test_find_nested_dirs() {
        // Nested directories must match
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo/bar/img.png"),
            PathBuf::from("masks/foo/bar/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/foo/bar/img.png")),
            Some(Path::new("masks/foo/bar/img.png"))
        );
        // Should not match wrong subpath
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/baz/img.png"),
            PathBuf::from("masks/foo/img.png"),
        ]);
        assert_eq!(find_mask_path(&vfs, Path::new("images/baz/img.png")), None);
    }

    #[test]
    fn test_find_case_insensitive() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/IMG.PNG"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/IMG.PNG")),
            Some(Path::new("masks/img.png"))
        );
    }
}
