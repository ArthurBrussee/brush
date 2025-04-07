mod data_source;

// This class helps working with an archive as a somewhat more regular filesystem.
//
// [1] really we want to just read directories.
// The reason is that picking directories isn't supported on
// rfd on wasm, nor is drag-and-dropping folders in egui.
use std::{
    collections::HashMap,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Context;
use path_clean::PathClean;
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    sync::Mutex,
};

use zip::{
    ZipArchive,
    result::{ZipError, ZipResult},
};

// On wasm, lots of things aren't Send that are send on non-wasm.
// Non-wasm tokio requires :Send for futures, tokio_with_wasm doesn't.
// So, it can help to annotate futures/objects as send only on not-wasm.
#[cfg(target_family = "wasm")]
mod wasm_send {
    pub trait WasmNotSend {}
    impl<T> WasmNotSend for T {}
}
#[cfg(not(target_family = "wasm"))]
mod wasm_send {
    pub trait WasmNotSend: Send {}
    impl<T: Send> WasmNotSend for T {}
}
pub use data_source::DataSource;
pub use wasm_send::*;

pub trait DynRead: AsyncRead + WasmNotSend + Unpin {}
impl<T: AsyncRead + WasmNotSend + Unpin> DynRead for T {}

// Sometimes rust is beautiful - sometimes it's ArcMutexOptionBox
type SharedRead = Arc<Mutex<Option<Box<dyn DynRead>>>>;

#[derive(Clone)]
pub struct ZipData {
    data: Arc<Vec<u8>>,
}

impl AsRef<[u8]> for ZipData {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

#[derive(Clone, Default)]
pub struct PathReader {
    paths: HashMap<PathBuf, SharedRead>,
}

impl PathReader {
    fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.paths.keys()
    }

    pub fn add(&mut self, path: &Path, reader: impl AsyncRead + WasmNotSend + Unpin + 'static) {
        self.paths.insert(
            path.to_path_buf(),
            #[allow(clippy::arc_with_non_send_sync)] // Not send/sync on wasm but that's ok.
            Arc::new(Mutex::new(Some(Box::new(reader)))),
        );
    }

    async fn open(&self, path: &Path) -> anyhow::Result<Box<dyn DynRead>> {
        let entry = self.paths.get(path).context("File not found")?;
        let reader = entry.lock().await.take();
        reader.context("Missing reader")
    }
}

pub enum BrushVfs {
    Zip(ZipArchive<Cursor<ZipData>>),
    Manual(PathReader),
    #[cfg(not(target_family = "wasm"))]
    Directory(PathBuf, Vec<PathBuf>),
}

impl BrushVfs {
    pub async fn from_zip_reader(reader: impl AsyncRead + Unpin) -> ZipResult<Self> {
        let mut bytes = vec![];
        let mut reader = reader;
        reader.read_to_end(&mut bytes).await?;

        let zip_data = ZipData {
            data: Arc::new(bytes),
        };
        let archive = ZipArchive::new(Cursor::new(zip_data))?;
        Ok(Self::Zip(archive))
    }

    pub fn from_paths(paths: PathReader) -> Self {
        Self::Manual(paths)
    }

    pub async fn from_directory(dir: &Path) -> anyhow::Result<Self> {
        #[cfg(not(target_family = "wasm"))]
        {
            if dir.is_file() {
                let file = tokio::fs::File::open(dir).await?;

                if dir.extension().is_some_and(|e| e == "zip") {
                    Ok(Self::from_zip_reader(file).await?)
                } else {
                    // Make a VFS with just this file.
                    let mut paths = PathReader::default();
                    paths.add(dir, file);
                    Ok(Self::from_paths(paths))
                }
            } else {
                // Make a VFS with all files contained in the directory.
                async fn walk_dir(dir: impl AsRef<Path>) -> std::io::Result<Vec<PathBuf>> {
                    let dir = PathBuf::from(dir.as_ref());

                    let mut paths = Vec::new();
                    let mut stack = vec![dir.clone()];

                    while let Some(path) = stack.pop() {
                        let mut read_dir = tokio::fs::read_dir(&path).await?;

                        while let Some(entry) = read_dir.next_entry().await? {
                            let path = entry.path();
                            if path.is_dir() {
                                stack.push(path.clone());
                            }
                            paths.push(
                                path.strip_prefix(dir.clone())
                                    .map_err(|_e| std::io::ErrorKind::InvalidInput)?
                                    .to_path_buf(),
                            );
                        }
                    }
                    Ok(paths)
                }

                Ok(Self::Directory(dir.to_path_buf(), walk_dir(dir).await?))
            }
        }

        #[cfg(target_family = "wasm")]
        {
            let _ = dir;
            panic!("Cannot read paths on wasm");
        }
    }

    pub fn file_names(&self) -> impl Iterator<Item = PathBuf> + '_ {
        let iterator: Box<dyn Iterator<Item = &Path>> = match self {
            Self::Zip(archive) => Box::new(archive.file_names().map(Path::new)),
            Self::Manual(map) => Box::new(map.paths().map(|p| p.as_path())),
            #[cfg(not(target_family = "wasm"))]
            Self::Directory(_, paths) => Box::new(paths.iter().map(|p| p.as_path())),
        };
        iterator.filter_map(|p| {
            // stupic macOS.
            if !p.starts_with("__MACOSX") {
                Some(p.clean())
            } else {
                None
            }
        })
    }

    pub async fn reader_at_path(&self, path: &Path) -> anyhow::Result<Box<dyn DynRead>> {
        match self {
            Self::Zip(archive) => {
                let name = archive
                    .file_names()
                    .find(|name| path == Path::new(name))
                    .ok_or(ZipError::FileNotFound)?;
                let name = name.to_owned();
                let mut buffer = vec![];
                // Archive is cheap to clone, as the data is an Arc<[u8]>.
                archive.clone().by_name(&name)?.read_to_end(&mut buffer)?;
                Ok(Box::new(Cursor::new(buffer)))
            }
            Self::Manual(map) => map.open(path).await,
            #[cfg(not(target_family = "wasm"))]
            Self::Directory(dir, _) => {
                let total_path = dir.join(path);
                let file = tokio::fs::File::open(total_path).await?;
                let file = tokio::io::BufReader::new(file);
                Ok(Box::new(file))
            }
        }
    }
}
