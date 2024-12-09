//
// This class helps working with an archive as a somewhat more regular filesystem.
//
// [1] really we want to just read directories.
// The reason is that picking directories isn't supported on
// rfd on wasm, nor is drag-and-dropping folders in egui.
use std::{
    collections::HashMap,
    io::{Cursor, Read},
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::Context;
use tokio::{io::AsyncRead, io::AsyncReadExt, sync::Mutex};

use zip::{
    result::{ZipError, ZipResult},
    ZipArchive,
};

type DynRead = Box<dyn AsyncRead + Send + Unpin>;

#[derive(Clone)]
pub struct ZipData {
    data: Arc<Vec<u8>>,
}

impl AsRef<[u8]> for ZipData {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

pub(crate) fn normalized_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|c| !matches!(c, Component::CurDir | Component::ParentDir))
        .collect()
}

#[derive(Clone, Default)]
pub struct PathReader {
    paths: HashMap<PathBuf, Arc<Mutex<Option<DynRead>>>>,
}

impl PathReader {
    fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.paths.keys()
    }

    pub fn add(&mut self, path: &Path, reader: impl AsyncRead + Send + Unpin + 'static) {
        self.paths.insert(
            path.to_path_buf(),
            Arc::new(Mutex::new(Some(Box::new(reader)))),
        );
    }

    async fn open(&mut self, path: &Path) -> anyhow::Result<Box<dyn AsyncRead + Send + Unpin>> {
        let entry = self.paths.remove(path).context("File not found")?;
        let reader = entry.lock().await.take();
        reader.context("Missing reader")
    }
}

#[derive(Clone)]
pub enum BrushVfs {
    Zip(ZipArchive<Cursor<ZipData>>),
    Manual(PathReader),
    #[cfg(not(target_family = "wasm"))]
    Directory(PathBuf, Vec<PathBuf>),
}

// TODO: This is all awfully ad-hoc.
impl BrushVfs {
    pub async fn from_zip_reader(reader: impl AsyncRead + Unpin) -> ZipResult<Self> {
        let mut bytes = vec![];
        let mut reader = reader;
        reader.read_to_end(&mut bytes).await?;

        let zip_data = ZipData {
            data: Arc::new(bytes),
        };
        let archive = ZipArchive::new(Cursor::new(zip_data))?;
        Ok(BrushVfs::Zip(archive))
    }

    pub fn from_paths(paths: PathReader) -> Self {
        BrushVfs::Manual(paths)
    }

    #[cfg(not(target_family = "wasm"))]
    pub async fn from_directory(dir: &Path) -> anyhow::Result<Self> {
        let mut read = ::tokio::fs::read_dir(dir).await?;
        let mut paths = vec![];
        while let Some(entry) = read.next_entry().await? {
            paths.push(entry.path());
        }
        Ok(BrushVfs::Directory(dir.to_path_buf(), paths))
    }

    pub fn file_names(&self) -> impl Iterator<Item = &Path> + '_ {
        let iterator: Box<dyn Iterator<Item = &Path>> = match self {
            BrushVfs::Zip(archive) => Box::new(archive.file_names().map(Path::new)),
            BrushVfs::Manual(map) => Box::new(map.paths().map(|p| p.as_path())),
            #[cfg(not(target_family = "wasm"))]
            BrushVfs::Directory(_, paths) => Box::new(paths.iter().map(|p| p.as_path())),
        };
        // stupic macOS.
        iterator.filter(|p| !p.starts_with("__MACOSX"))
    }

    pub async fn open_path(&mut self, path: &Path) -> anyhow::Result<DynRead> {
        match self {
            BrushVfs::Zip(archive) => {
                let name = archive
                    .file_names()
                    .find(|name| path == Path::new(name))
                    .ok_or(ZipError::FileNotFound)?;
                let name = name.to_owned();
                let mut buffer = vec![];
                archive.by_name(&name)?.read_to_end(&mut buffer)?;
                Ok(Box::new(Cursor::new(buffer)))
            }
            BrushVfs::Manual(map) => map.open(path).await,
            #[cfg(not(target_family = "wasm"))]
            BrushVfs::Directory(path_buf, _) => {
                let file = tokio::fs::File::open(path_buf).await?;
                Ok(Box::new(file))
            }
        }
    }
}
