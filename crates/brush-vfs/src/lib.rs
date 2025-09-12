mod data_source;

// This class helps working with an archive as a somewhat more regular filesystem.
//
// [1] really we want to just read directories.
// The reason is that picking directories isn't supported on
// rfd on wasm, nor is drag-and-dropping folders in egui.
use std::{
    collections::HashMap,
    fmt::Debug,
    io::{self, Cursor, Error},
    path::{Path, PathBuf},
    sync::Arc,
};

pub use data_source::{DataSource, DataSourceError};

use path_clean::PathClean;
use tokio::{
    io::{AsyncBufRead, AsyncRead, AsyncReadExt, BufReader},
    sync::Mutex,
};
use tokio_with_wasm::alias as tokio_wasm;

use async_zip::base::read::stream::ZipFileReader;
use tokio_util::compat::{FuturesAsyncReadCompatExt, TokioAsyncReadCompatExt};

// On wasm, lots of things aren't Send that are send on non-wasm.
// Non-wasm tokio requires :Send for futures, tokio_with_wasm doesn't.

// So, it can help to annotate futures/objects as send only on not-wasm.
#[cfg(target_family = "wasm")]
mod wasm_send {
    pub trait SendNotWasm {}
    impl<T> SendNotWasm for T {}
}
#[cfg(not(target_family = "wasm"))]
mod wasm_send {
    pub trait SendNotWasm: Send {}
    impl<T: Send> SendNotWasm for T {}
}

pub use wasm_send::*;

pub trait DynRead: AsyncBufRead + SendNotWasm + Unpin {}
impl<T: AsyncBufRead + SendNotWasm + Unpin> DynRead for T {}

// Sometimes rust is beautiful - sometimes it's ArcMutexOptionBox
type SharedRead = Arc<Mutex<Option<Box<dyn DynRead>>>>;

// New type to keep track that this string-y path might not correspond to
// a physical file path.
//
#[derive(Debug, Eq, PartialEq, Hash)]
struct PathKey(String);

impl PathKey {
    fn from_path(path: &Path) -> Self {
        let key = path
            .clean()
            .to_str()
            .expect("Path is not valid ascii")
            .to_lowercase()
            .replace('\\', "/");
        let key = if key.starts_with('/') {
            key
        } else {
            '/'.to_string() + &key
        };
        Self(key)
    }
}

// Simple wrapper for Arc<Vec<u8>> that implements AsRef<[u8]> for Cursor
struct ZipVec(Arc<Vec<u8>>);

impl AsRef<[u8]> for ZipVec {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

async fn read_at_most<R: AsyncRead + Unpin>(reader: &mut R, limit: usize) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0; limit];
    let bytes_read = reader.read(&mut buffer).await?;
    buffer.truncate(bytes_read);
    Ok(buffer)
}

enum VfsContainer {
    Zip {
        // TODO: Fill this in.
        entries: HashMap<PathBuf, Arc<Vec<u8>>>,
    },
    Manual {
        readers: HashMap<PathBuf, SharedRead>,
    },
    Directory {
        base_path: PathBuf,
        #[cfg(target_family = "wasm")]
        files: Option<HashMap<PathBuf, web_sys::File>>,
    },
}

impl Debug for VfsContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zip { .. } => f.debug_struct("Zip").finish(),
            Self::Manual { .. } => f.debug_struct("Manual").finish(),
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
    let mut result = HashMap::new();
    for path in paths {
        let path = path.clean();

        // Only consider files with extensions for now. Zip files report directories as paths with no extension (not ending in '/')
        // so can't really differentiate extensionless files from directories. We don't need any files without extensions
        // so just skip them.
        if path.extension().is_some() && !path.components().any(|c| c.as_os_str() == "__MACOSX") {
            let key = PathKey::from_path(&path);
            assert!(
                result.insert(key, path.clone()).is_none(),
                "Duplicate path found: {}. Paths must be unique (case non-sensitive)",
                path.display()
            );
        }
    }
    result
}

use thiserror::Error;

fn io_error(e: async_zip::error::ZipError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
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

    pub async fn from_reader(reader: impl DynRead + 'static) -> Result<Self, VfsConstructError> {
        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let mut data = BufReader::new(reader);
        let peek = read_at_most(&mut data, 64).await?;
        let mut reader: Box<dyn DynRead> =
            Box::new(AsyncReadExt::chain(Cursor::new(peek.clone()), data));

        if peek.starts_with(b"ply") {
            let path = PathBuf::from("input.ply");
            let reader_ref = Arc::new(Mutex::new(Some(reader)));
            Ok(Self {
                lookup: lookup_from_paths(std::slice::from_ref(&path)),
                container: VfsContainer::Manual {
                    readers: HashMap::from([(path, reader_ref)]),
                },
            })
        } else if peek.starts_with(b"PK") {
            let mut zip_reader = ZipFileReader::new(reader.compat());
            let mut entries = HashMap::new();

            while let Some(mut entry) = zip_reader.next_with_entry().await.map_err(io_error)? {
                if let Ok(filename) = entry.reader().entry().filename().clone().as_str() {
                    let mut data = vec![];
                    let mut reader = entry.reader_mut().compat();
                    reader.read_to_end(&mut data).await?;
                    entries.insert(PathBuf::from(filename), Arc::new(data));
                    zip_reader = entry.skip().await.map_err(io_error)?;
                } else {
                    zip_reader = entry.skip().await.map_err(io_error)?;
                }

                tokio_wasm::task::yield_now().await;
            }

            let path_bufs = entries.keys().cloned().collect::<Vec<_>>();

            Ok(Self {
                lookup: lookup_from_paths(&path_bufs),
                container: VfsContainer::Zip { entries },
            })
        } else if peek.starts_with(b"<!DOCTYPE html>") {
            let mut html = String::new();
            reader.read_to_string(&mut html).await?;
            Err(VfsConstructError::InvalidHtml(html))
        } else {
            Err(VfsConstructError::UnknownDataType)
        }
    }

    pub async fn from_path(dir: &Path) -> Result<Self, VfsConstructError> {
        #[cfg(not(target_family = "wasm"))]
        {
            if dir.is_file() {
                // Construct a reader. This is needed for zip files, as
                // it's not really just a single path.
                let file = tokio::fs::File::open(dir).await?;
                let reader = BufReader::new(file);
                Self::from_reader(reader).await
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
                        #[cfg(target_family = "wasm")]
                        files: None,
                    },
                })
            }
        }

        #[cfg(target_family = "wasm")]
        {
            let _ = dir;
            panic!("Cannot read paths on wasm");
        }
    }

    #[cfg(target_family = "wasm")]
    pub fn from_wasm_files(
        files: HashMap<PathBuf, web_sys::File>,
    ) -> Result<Self, VfsConstructError> {
        let paths: Vec<PathBuf> = files.keys().cloned().collect();

        Ok(Self {
            lookup: lookup_from_paths(&paths),
            container: VfsContainer::Directory {
                base_path: PathBuf::new(), // Empty path for WASM
                files: Some(files),
            },
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

    pub fn files_ending_in<'a>(&'a self, end_path: &'a str) -> impl Iterator<Item = PathBuf> + 'a {
        let end_keyed = PathKey::from_path(Path::new(end_path)).0;

        self.lookup
            .iter()
            .filter(move |kv| kv.0.0.ends_with(&end_keyed))
            .map(|kv| kv.1.clone())
    }

    pub fn files_with_stem<'a>(&'a self, filestem: &'a str) -> impl Iterator<Item = PathBuf> + 'a {
        let filestem = filestem.to_lowercase();
        self.lookup.values().filter_map(move |path| {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())?
                .to_lowercase();
            (stem == filestem).then(|| path.clone())
        })
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
            VfsContainer::Zip { entries } => {
                let data = entries.get(path).expect("Unreachable").clone();
                Ok(Box::new(Cursor::new(ZipVec(data))))
            }
            VfsContainer::Manual { readers } => {
                // Readers get taken out of the map as they are not cloneable.
                // This means that unlike other methods this path can only be loaded
                // once.
                let reader_mut = readers.get(path).expect("Unreachable");
                let reader = reader_mut.lock().await.take();
                reader.ok_or_else(|| {
                    Error::new(
                        io::ErrorKind::NotFound,
                        format!("File not found: {}", path.display()),
                    )
                })
            }
            VfsContainer::Directory {
                base_path: _dir,
                #[cfg(target_family = "wasm")]
                files,
            } => {
                #[cfg(target_family = "wasm")]
                if let Some(files) = files {
                    let file = files.get(path).ok_or_else(|| {
                        Error::new(
                            io::ErrorKind::NotFound,
                            format!("File not found: {}", path.display()),
                        )
                    })?;

                    // Create a stream reader from the File object
                    use futures_util::StreamExt;
                    use tokio_util::bytes::Bytes;
                    use tokio_util::io::StreamReader;
                    use wasm_bindgen::JsCast;
                    use wasm_streams::ReadableStream as WasmReadableStream;

                    let readable_stream: web_sys::ReadableStream = file.stream();
                    let wasm_stream = WasmReadableStream::from_raw(readable_stream);

                    let byte_stream = wasm_stream.into_stream().map(|result| {
                        result
                            .map_err(|e| {
                                io::Error::new(
                                    io::ErrorKind::Other,
                                    format!("Stream error: {:?}", e),
                                )
                            })
                            .and_then(|chunk| {
                                if let Ok(uint8_array) = chunk.dyn_into::<js_sys::Uint8Array>() {
                                    let mut data = vec![0; uint8_array.length() as usize];
                                    uint8_array.copy_to(&mut data);
                                    Ok(Bytes::from(data))
                                } else {
                                    Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "Invalid chunk type",
                                    ))
                                }
                            })
                    });

                    let stream_reader = StreamReader::new(byte_stream);
                    Ok(Box::new(tokio::io::BufReader::new(stream_reader)))
                } else {
                    Err(Error::new(
                        io::ErrorKind::InvalidInput,
                        "Cannot read filesystem paths on WASM",
                    ))
                }

                #[cfg(not(target_family = "wasm"))]
                {
                    let total_path = _dir.join(path);
                    let file = tokio::fs::File::open(total_path).await?;
                    let file = tokio::io::BufReader::new(file);
                    Ok(Box::new(file))
                }
            }
        }
    }
    pub fn empty() -> Self {
        Self {
            lookup: HashMap::new(),
            container: VfsContainer::Manual {
                readers: HashMap::new(),
            },
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
        let vfs = BrushVfs::from_reader(Cursor::new(zip_data)).await.unwrap();
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
        let vfs = BrushVfs::from_reader(Cursor::new(
            b"ply\nformat ascii 1.0\nend_header\nvertex data",
        ))
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
            BrushVfs::from_reader(Cursor::new(b"unknown")).await,
            Err(VfsConstructError::UnknownDataType)
        ));
        assert!(matches!(
            BrushVfs::from_reader(Cursor::new(b"<!DOCTYPE html>")).await,
            Err(VfsConstructError::InvalidHtml(_))
        ));
    }
}
