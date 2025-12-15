mod data_source;

use std::{
    collections::HashMap,
    fmt::Debug,
    io::{self, Cursor, Error},
    path::{Path, PathBuf},
    sync::Arc,
};

use async_zip::base::read::stream::ZipFileReader;
use path_clean::PathClean;
use thiserror::Error;
use tokio::{
    io::{AsyncBufRead, AsyncRead, AsyncReadExt, BufReader},
    sync::Mutex,
};
use tokio_util::compat::{FuturesAsyncReadCompatExt, TokioAsyncReadCompatExt};
use tokio_with_wasm::alias as tokio_wasm;

pub use data_source::{DataSource, DataSourceError};

// WASM doesn't require Send, but native tokio does.
#[cfg(target_family = "wasm")]
pub trait SendNotWasm {}
#[cfg(target_family = "wasm")]
impl<T> SendNotWasm for T {}
#[cfg(not(target_family = "wasm"))]
pub trait SendNotWasm: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send> SendNotWasm for T {}

pub trait DynRead: AsyncBufRead + SendNotWasm + Unpin {}
impl<T: AsyncBufRead + SendNotWasm + Unpin> DynRead for T {}

type StreamingReader = Arc<Mutex<Option<Box<dyn DynRead>>>>;

/// Wrapper so Cursor can use Arc<Vec<u8>> without cloning.
struct ArcVec(Arc<Vec<u8>>);
impl AsRef<[u8]> for ArcVec {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Normalized path key for case-insensitive lookups.
#[derive(Debug, Eq, PartialEq, Hash)]
struct PathKey(String);

impl PathKey {
    fn from_str(path: &str) -> Self {
        let key = path.to_lowercase().replace('\\', "/");
        Self(if key.starts_with('/') {
            key
        } else {
            format!("/{key}")
        })
    }

    fn from_path(path: &Path) -> Self {
        Self::from_str(path.clean().to_str().expect("Invalid path"))
    }
}

async fn read_at_most<R: AsyncRead + Unpin>(reader: &mut R, limit: usize) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0; limit];
    let bytes_read = reader.read(&mut buffer).await?;
    buffer.truncate(bytes_read);
    Ok(buffer)
}

enum VfsContainer {
    /// Raw data stored in memory (from zip files)
    InMemory {
        entries: HashMap<PathBuf, Arc<Vec<u8>>>,
    },
    /// A single file being streamed (e.g., PLY from HTTP).
    /// The reader can only be consumed once.
    Streaming { reader: StreamingReader },
    /// Native directory - reads from disk on demand
    #[cfg(not(target_family = "wasm"))]
    Directory { base_path: PathBuf },
    /// WASM directory - uses File System Access API to read files on demand
    #[cfg(target_family = "wasm")]
    Directory {
        dir_handle: rrfd::wasm::DirectoryHandle,
    },
}

impl Debug for VfsContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory { .. } => f.debug_struct("InMemory").finish(),
            Self::Streaming { .. } => f.debug_struct("Streaming").finish(),
            Self::Directory { .. } => f.debug_struct("Directory").finish(),
        }
    }
}

#[derive(Debug)]
pub struct BrushVfs {
    lookup: HashMap<PathKey, PathBuf>,
    container: VfsContainer,
}

fn lookup_from_paths(paths: &[PathBuf]) -> HashMap<PathKey, PathBuf> {
    paths
        .iter()
        .map(|p| p.clean())
        // Skip directories and __MACOSX metadata
        .filter(|p| p.extension().is_some() && !p.components().any(|c| c.as_os_str() == "__MACOSX"))
        .map(|p| (PathKey::from_path(&p), p))
        .collect()
}

fn zip_error(e: async_zip::error::ZipError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

#[derive(Debug, Error)]
pub enum VfsConstructError {
    #[error("I/O error while constructing BrushVfs.")]
    IoError(#[from] std::io::Error),
    #[error("Got a status page instead of content: \n\n {0}")]
    InvalidHtml(String),
    #[error("Unknown data type. Only zip and ply files are supported")]
    UnknownDataType,
}

impl BrushVfs {
    pub fn file_count(&self) -> usize {
        self.lookup.len()
    }

    pub fn file_paths(&self) -> impl Iterator<Item = PathBuf> {
        self.lookup.values().cloned()
    }

    pub async fn from_reader(
        reader: impl DynRead + 'static,
        name: Option<String>,
    ) -> Result<Self, VfsConstructError> {
        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let mut data = BufReader::new(reader);
        let peek = read_at_most(&mut data, 64).await?;
        let mut reader: Box<dyn DynRead> =
            Box::new(AsyncReadExt::chain(Cursor::new(peek.clone()), data));

        if peek.starts_with(b"ply") {
            // For single PLY files, keep the reader for streaming
            let path = PathBuf::from(name.unwrap_or_else(|| "input.ply".to_owned()));

            Ok(Self {
                lookup: lookup_from_paths(std::slice::from_ref(&path)),
                container: VfsContainer::Streaming {
                    reader: Arc::new(Mutex::new(Some(reader))),
                },
            })
        } else if peek.starts_with(b"PK") {
            let mut zip_reader = ZipFileReader::new(reader.compat());
            let mut entries = HashMap::new();

            while let Some(mut entry) = zip_reader.next_with_entry().await.map_err(zip_error)? {
                if let Ok(filename) = entry.reader().entry().filename().clone().as_str() {
                    let mut data = vec![];
                    let mut reader = entry.reader_mut().compat();
                    reader.read_to_end(&mut data).await?;
                    entries.insert(PathBuf::from(filename), Arc::new(data));
                    zip_reader = entry.skip().await.map_err(zip_error)?;
                } else {
                    zip_reader = entry.skip().await.map_err(zip_error)?;
                }

                tokio_wasm::task::yield_now().await;
            }

            let path_bufs = entries.keys().cloned().collect::<Vec<_>>();

            Ok(Self {
                lookup: lookup_from_paths(&path_bufs),
                container: VfsContainer::InMemory { entries },
            })
        } else if peek.starts_with(b"<!DOCTYPE html>") {
            let mut html = String::new();
            reader.read_to_string(&mut html).await?;
            Err(VfsConstructError::InvalidHtml(html))
        } else {
            Err(VfsConstructError::UnknownDataType)
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub async fn from_path(dir: &Path) -> Result<Self, VfsConstructError> {
        if dir.is_file() {
            // Construct a reader. This is needed for zip files, as
            // it's not really just a single path.
            let file = tokio::fs::File::open(dir).await?;
            let reader = BufReader::new(file);
            let name = dir.file_name().and_then(|n| n.to_str()).map(String::from);
            Self::from_reader(reader, name).await
        } else {
            // Make a VFS with all files contained in the directory.
            async fn walk_dir(dir: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
                let dir = PathBuf::from(dir.as_ref());

                let mut paths = Vec::new();
                let mut stack = vec![dir.clone()];

                while let Some(path) = stack.pop() {
                    let mut read_dir = tokio::fs::read_dir(&path).await?;

                    while let Some(entry) = read_dir.next_entry().await? {
                        let path = entry.path();
                        if path.is_dir() {
                            stack.push(path.clone());
                        } else {
                            let path = path
                                .strip_prefix(dir.clone())
                                .map_err(|_e| io::ErrorKind::InvalidInput)?
                                .to_path_buf();
                            paths.push(path);
                        }

                        tokio_wasm::task::yield_now().await;
                    }
                }
                Ok(paths)
            }

            let files = walk_dir(dir).await?;
            Ok(Self {
                lookup: lookup_from_paths(&files),
                container: VfsContainer::Directory {
                    base_path: dir.to_path_buf(),
                },
            })
        }
    }

    #[cfg(target_family = "wasm")]
    pub async fn from_directory_handle(
        dir_handle: rrfd::wasm::DirectoryHandle,
    ) -> Result<Self, VfsConstructError> {
        // List all files in the directory
        let paths = dir_handle.list_files().await.map_err(|_| {
            VfsConstructError::IoError(io::Error::new(
                io::ErrorKind::Other,
                "Failed to list directory contents",
            ))
        })?;

        Ok(Self {
            lookup: lookup_from_paths(&paths),
            container: VfsContainer::Directory { dir_handle },
        })
    }

    pub fn files_with_extension<'a>(
        &'a self,
        extension: &'a str,
    ) -> impl Iterator<Item = PathBuf> + 'a {
        let extension = extension.to_lowercase();

        self.lookup.values().filter_map(move |path| {
            let ext = path
                .extension()
                .and_then(|ext| ext.to_str())?
                .to_lowercase();
            (ext == extension).then(|| path.clone())
        })
    }

    pub fn files_ending_in<'a>(&'a self, end_path: &str) -> impl Iterator<Item = &'a Path> + 'a {
        let end_keyed = PathKey::from_str(end_path).0;

        self.lookup
            .iter()
            .filter(move |kv| kv.0.0.ends_with(&end_keyed))
            .map(|kv| kv.1.as_path())
    }

    /// Iterate over all files in the VFS.
    pub fn iter_files<'a>(&'a self) -> impl Iterator<Item = &'a Path> + 'a {
        self.lookup.values().map(|path| path.as_path())
    }

    pub async fn reader_at_path(&self, path: &Path) -> io::Result<Box<dyn DynRead>> {
        let key = PathKey::from_path(path);
        let path = self.lookup.get(&key).ok_or_else(|| {
            Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            )
        })?;

        match &self.container {
            VfsContainer::InMemory { entries } => {
                let data = entries.get(path).expect("Unreachable").clone();
                Ok(Box::new(Cursor::new(ArcVec(data))))
            }
            VfsContainer::Streaming { reader } => {
                // Streaming reader can only be consumed once
                let reader = reader
                    .lock()
                    .await
                    .take()
                    .ok_or_else(|| Error::other("Streaming file has already been read"))?;
                Ok(reader)
            }
            #[cfg(not(target_family = "wasm"))]
            VfsContainer::Directory { base_path } => {
                let total_path = base_path.join(path);
                // Higher capacity buffer helps performance
                let file = tokio::io::BufReader::with_capacity(
                    5 * 1024 * 1024,
                    tokio::fs::File::open(total_path).await?,
                );
                Ok(Box::new(file))
            }
            #[cfg(target_family = "wasm")]
            VfsContainer::Directory { dir_handle } => {
                use futures_util::StreamExt;
                use tokio_util::io::StreamReader;
                use wasm_bindgen::JsCast;

                let file = dir_handle.get_file(path).await.map_err(|_| {
                    Error::new(
                        io::ErrorKind::NotFound,
                        format!("File not found: {}", path.display()),
                    )
                })?;

                let stream = wasm_streams::ReadableStream::from_raw(file.stream())
                    .into_stream()
                    .map(|result| {
                        result
                            .map_err(|e| Error::new(io::ErrorKind::Other, format!("{e:?}")))
                            .and_then(|chunk| {
                                let array =
                                    chunk.dyn_into::<js_sys::Uint8Array>().map_err(|_| {
                                        Error::new(io::ErrorKind::InvalidData, "Invalid chunk")
                                    })?;
                                let mut data = vec![0u8; array.length() as usize];
                                array.copy_to(&mut data);
                                Ok(tokio_util::bytes::Bytes::from(data))
                            })
                    });

                Ok(Box::new(BufReader::new(StreamReader::new(stream))))
            }
        }
    }

    pub fn empty() -> Self {
        Self {
            lookup: HashMap::new(),
            container: VfsContainer::InMemory {
                entries: HashMap::new(),
            },
        }
    }

    /// Create a test VFS from file paths with empty content.
    #[doc(hidden)]
    pub fn create_test_vfs(paths: Vec<PathBuf>) -> Self {
        let lookup = lookup_from_paths(&paths);

        let entries = paths
            .into_iter()
            .filter(|p| p.extension().is_some())
            .map(|p| (p, Arc::new(Vec::new())))
            .collect();

        Self {
            lookup,
            container: VfsContainer::InMemory { entries },
        }
    }

    pub fn base_path(&self) -> Option<PathBuf> {
        match &self.container {
            VfsContainer::InMemory { .. } => None,
            VfsContainer::Streaming { .. } => None,
            #[cfg(not(target_family = "wasm"))]
            VfsContainer::Directory { base_path } => Some(base_path.clone()),
            #[cfg(target_family = "wasm")]
            VfsContainer::Directory { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tokio::io::AsyncReadExt;

    async fn create_test_zip() -> Vec<u8> {
        use async_zip::base::write::ZipFileWriter;
        use async_zip::{Compression, ZipEntryBuilder};

        let mut buffer = Vec::new();
        let mut writer = ZipFileWriter::new(&mut buffer);

        // Add test.txt
        let entry = ZipEntryBuilder::new("test.txt".into(), Compression::Stored);
        writer
            .write_entry_whole(entry, b"hello world")
            .await
            .unwrap();

        // Add data.json
        let entry = ZipEntryBuilder::new("data.json".into(), Compression::Stored);
        writer
            .write_entry_whole(entry, b"{\"key\": \"value\"}")
            .await
            .unwrap();

        writer.close().await.unwrap();
        buffer
    }

    #[tokio::test]
    async fn test_zip_vfs_workflow() {
        let zip_data = create_test_zip().await;
        let vfs = BrushVfs::from_reader(Cursor::new(zip_data), None)
            .await
            .unwrap();
        assert_eq!(vfs.file_count(), 2);

        let txt_files: Vec<_> = vfs.files_with_extension("txt").collect();
        assert_eq!(txt_files.len(), 1);

        let json_files: Vec<_> = vfs.files_with_extension("json").collect();
        assert_eq!(json_files.len(), 1);

        let mut content = String::new();
        vfs.reader_at_path(&txt_files[0])
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "hello world");

        // Test JSON file
        let mut content = String::new();
        vfs.reader_at_path(&json_files[0])
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "{\"key\": \"value\"}");

        // Test case-insensitive access
        let mut content = String::new();
        vfs.reader_at_path(Path::new("TEST.TXT"))
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "hello world");

        assert!(
            vfs.reader_at_path(Path::new("nonexistent.txt"))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_format_detection_and_errors() {
        // Test PLY format
        let vfs = BrushVfs::from_reader(
            Cursor::new(b"ply\nformat ascii 1.0\nend_header\nvertex data"),
            None,
        )
        .await
        .unwrap();
        let mut content = String::new();
        vfs.reader_at_path(Path::new("input.ply"))
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "ply\nformat ascii 1.0\nend_header\nvertex data");

        // Test error cases
        assert!(matches!(
            BrushVfs::from_reader(Cursor::new(b"unknown"), None).await,
            Err(VfsConstructError::UnknownDataType)
        ));
        assert!(matches!(
            BrushVfs::from_reader(Cursor::new(b"<!DOCTYPE html>"), None).await,
            Err(VfsConstructError::InvalidHtml(_))
        ));
    }
}
