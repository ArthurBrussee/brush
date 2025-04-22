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
    io::{AsyncRead, AsyncReadExt, BufReader},
    sync::Mutex,
};

use tokio_stream::Stream;
use zip::ZipArchive;

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
pub use data_source::DataSource;
pub use wasm_send::*;

pub trait DynStream<Item>: Stream<Item = Item> + SendNotWasm {}
impl<Item, T: Stream<Item = Item> + SendNotWasm> DynStream<Item> for T {}

pub trait DynRead: AsyncRead + SendNotWasm + Unpin {}
impl<T: AsyncRead + SendNotWasm + Unpin> DynRead for T {}

// Sometimes rust is beautiful - sometimes it's ArcMutexOptionBox
type SharedRead = Arc<Mutex<Option<Box<dyn DynRead>>>>;

// New type to keep track that this string-y path might not correspond to
// a physical file path.
//
#[derive(Debug, Eq, PartialEq, Hash)]
struct PathKey(String);

impl PathKey {
    fn from_path(path: &Path) -> Self {
        Self(
            path.clean()
                .to_str()
                .expect("Path is not valid ascii")
                .to_lowercase(),
        )
    }
}

#[derive(Clone)]
pub struct ZipData {
    data: Arc<Vec<u8>>,
}

impl AsRef<[u8]> for ZipData {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

async fn read_at_most<R: AsyncRead + Unpin>(
    reader: &mut R,
    limit: usize,
) -> std::io::Result<Vec<u8>> {
    let mut buffer = vec![0; limit];
    let bytes_read = reader.read(&mut buffer).await?;
    buffer.truncate(bytes_read);
    Ok(buffer)
}

enum VfsContainer {
    Zip {
        archive: ZipArchive<Cursor<ZipData>>,
    },
    Manual {
        readers: HashMap<PathBuf, SharedRead>,
    },
    #[cfg(not(target_family = "wasm"))]
    Directory { base_path: PathBuf },
}

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

impl BrushVfs {
    pub fn file_count(&self) -> usize {
        self.lookup.len()
    }

    pub fn file_paths(&self) -> impl Iterator<Item = PathBuf> {
        self.lookup.values().cloned()
    }

    pub async fn from_reader(
        reader: impl AsyncRead + SendNotWasm + Unpin + 'static,
    ) -> anyhow::Result<Self> {
        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let mut data = BufReader::new(reader);
        let peek = read_at_most(&mut data, 64).await?;
        let mut reader: Box<dyn DynRead> =
            Box::new(AsyncReadExt::chain(Cursor::new(peek.clone()), data));

        if peek.as_slice().starts_with(b"ply") {
            let path = PathBuf::from("input.ply");
            let reader = Arc::new(Mutex::new(Some(reader)));
            Ok(Self {
                lookup: lookup_from_paths(&[path.clone()]),
                container: VfsContainer::Manual {
                    readers: HashMap::from([(path, reader)]),
                },
            })
        } else if peek.starts_with(b"PK") {
            let mut bytes = vec![];
            reader.read_to_end(&mut bytes).await?;
            let archive = ZipArchive::new(Cursor::new(ZipData {
                data: Arc::new(bytes),
            }))?;
            let file_names: Vec<_> = archive.file_names().map(PathBuf::from).collect();
            Ok(Self {
                lookup: lookup_from_paths(&file_names),
                container: VfsContainer::Zip { archive },
            })
        } else if peek.starts_with(b"<!DOCTYPE html>") {
            let mut html = String::new();
            reader.read_to_string(&mut html).await?;

            // TODO: On WASM, display the actual HTML? Idk.
            anyhow::bail!("Failed to download data. \n\n {html}")
        } else {
            anyhow::bail!("only zip and ply files are supported. Unknown data type.")
        }
    }

    pub async fn from_path(dir: &Path) -> anyhow::Result<Self> {
        #[cfg(not(target_family = "wasm"))]
        {
            if dir.is_file() {
                let file = tokio::fs::File::open(dir).await?;
                let reader = BufReader::new(file);
                Self::from_reader(reader).await
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
                            } else {
                                let path = path
                                    .strip_prefix(dir.clone())
                                    .map_err(|_e| std::io::ErrorKind::InvalidInput)?
                                    .to_path_buf();
                                paths.push(path);
                            }
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
        {
            let _ = dir;
            panic!("Cannot read paths on wasm");
        }
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

    pub fn files_with_filename<'a>(
        &'a self,
        filename: &'a str,
    ) -> impl Iterator<Item = PathBuf> + 'a {
        let filename = filename.to_lowercase();
        self.lookup.values().filter_map(move |path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())?
                .to_lowercase();
            (name == filename).then(|| path.clone())
        })
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

    pub async fn reader_at_path(&self, path: &Path) -> anyhow::Result<Box<dyn DynRead>> {
        let key = PathKey::from_path(path);
        let path = self.lookup.get(&key).context("File not found")?;

        match &self.container {
            VfsContainer::Zip { archive } => {
                let name = path
                    .to_str()
                    .expect("Invalid UTF-8 in zip file")
                    .replace('\\', "/");
                let mut buffer = vec![];
                // Archive is cheap to clone, as the data is an Arc<[u8]>.
                archive.clone().by_name(&name)?.read_to_end(&mut buffer)?;
                Ok(Box::new(Cursor::new(buffer)))
            }
            VfsContainer::Manual { readers } => {
                // Readers get taken out of the map as they are not cloneable.
                // This means that unlike other methods this path can only be loaded
                // once.
                let reader_mut = readers.get(path).context("File not found")?;
                let reader = reader_mut.lock().await.take();
                reader.context("Missing reader")
            }
            #[cfg(not(target_family = "wasm"))]
            VfsContainer::Directory { base_path: dir } => {
                // TODO: Use a string -> PathBuf cache.
                let total_path = dir.join(path);
                let file = tokio::fs::File::open(total_path).await?;
                let file = tokio::io::BufReader::new(file);
                Ok(Box::new(file))
            }
        }
    }
}
