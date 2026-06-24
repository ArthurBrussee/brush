use crate::{Dataset, config::LoadDataseConfig, scene::SceneView};
use brush_serde::{DeserializeError, SplatMessage, load_splat_from_ply};

use brush_vfs::BrushVfs;
use image::ImageError;
use itertools::{Either, Itertools};
use std::{path::Path, sync::Arc};

pub mod colmap;
pub mod nerfstudio;
pub mod realitycapture;

use thiserror::Error;

pub struct DatasetLoadResult {
    pub init_splat: Option<SplatMessage>,
    pub dataset: Dataset,
    pub warnings: Vec<String>,
}

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("I/O error while loading dataset: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error decoding JSON: {0}")]
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
    #[error(transparent)]
    FormatError(#[from] FormatError),

    #[error("Failed to load initial point cloud: {0}")]
    InitialPointCloudError(#[from] DeserializeError),

    #[error(
        "Format not recognized: only colmap, nerfstudio json and RealityCapture csv are supported"
    )]
    FormatNotSupported,
}

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDataseConfig,
) -> Result<DatasetLoadResult, DatasetError> {
    let mut dataset = colmap::load_dataset(vfs.clone(), load_args).await;

    if dataset.is_none() {
        dataset = nerfstudio::read_dataset(vfs.clone(), load_args).await;
    }

    if dataset.is_none() {
        dataset = realitycapture::read_dataset(vfs.clone(), load_args).await;
    }

    let Some(dataset) = dataset else {
        return Err(DatasetError::FormatNotSupported);
    };

    let result = dataset?;

    // A dataset that parsed but has no usable training views (e.g. every image
    // was missing or filtered out) would otherwise "load" and then crash on the
    // first training batch. Reject it here with a typed error instead.
    if result.dataset.train.views.is_empty() {
        return Err(FormatError::InvalidFormat(
            "dataset contains no usable training views (all images missing or filtered out)"
                .to_owned(),
        )
        .into());
    }

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
        result.init_splat
    };

    Ok(DatasetLoadResult {
        init_splat,
        dataset: result.dataset,
        warnings: result.warnings,
    })
}

/// Resolve a bare image name (as stored by colmap / `RealityCapture`, which only
/// record a filename) to a path in the VFS by brute-force suffix search. Masks
/// are skipped so an image never resolves to its own mask.
fn find_image_by_name<'a>(vfs: &'a BrushVfs, name: &str) -> Option<&'a Path> {
    vfs.files_ending_in(name)
        .filter(|p| !p.iter().any(|f| f == "masks"))
        .min()
}

/// Convert an OpenGL/Blender camera-to-world matrix (the nerfstudio
/// `transform_matrix` convention: +X right, +Y up, +Z back) into brush's
/// camera pose (+X right, +Y down, +Z forward).
fn opengl_c2w_to_pose(mut c2w: glam::Mat4) -> (glam::Vec3, glam::Quat) {
    c2w.y_axis *= -1.0;
    c2w.z_axis *= -1.0;
    let (_, rotation, translation) = c2w.to_scale_rotation_translation();
    (translation, rotation)
}

/// Split views into (train, eval) by selecting every `eval_split_every`-th view
/// for eval. With `None`, every view is a train view.
fn split_eval_every(
    views: Vec<SceneView>,
    eval_split_every: Option<usize>,
) -> (Vec<SceneView>, Vec<SceneView>) {
    views.into_iter().enumerate().partition_map(|(i, v)| {
        if let Some(split) = eval_split_every
            && i % split == 0
        {
            Either::Right(v)
        } else {
            Either::Left(v)
        }
    })
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

/// Locate the `points3d.{txt,bin}` belonging to the chosen reconstruction.
fn find_points3d_path<'a>(vfs: &'a BrushVfs, points_dir: &'a Path) -> Option<(&'a Path, bool)> {
    let path = vfs
        .files_ending_in("points3d.txt")
        .chain(vfs.files_ending_in("points3d.bin"))
        .find(|p| p.parent() == Some(points_dir))?;
    let is_binary = matches!(path.extension().and_then(|e| e.to_str()), Some("bin"));
    Some((path, is_binary))
}

/// Locate a per-image depth map.
fn find_depth_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Option<&'a Path> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");

    vfs.iter_files().find(|candidate| {
        let Some(stem) = candidate.file_stem() else {
            return false;
        };
        if !(stem.eq_ignore_ascii_case(search_name) || stem.eq_ignore_ascii_case(search_stem)) {
            return false;
        }
        let depth_idx = candidate
            .components()
            .position(|c| c.as_os_str().eq_ignore_ascii_case("depth"));
        depth_idx.is_some_and(|idx| {
            let candidate_components: Vec<_> = candidate.components().collect();
            let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
            let depth_dir_subpath = &candidate_components[idx + 1..candidate_components.len() - 1];
            path_dir_components.ends_with(depth_dir_subpath)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(unsupported = test)]
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

    #[wasm_bindgen_test(unsupported = test)]
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

    #[wasm_bindgen_test(unsupported = test)]
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

    #[wasm_bindgen_test(unsupported = test)]
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
