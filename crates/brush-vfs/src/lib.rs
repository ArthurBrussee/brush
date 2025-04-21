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
#[derive(Eq, PartialEq, Hash)]
struct PathKey(String);

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

fn compare_paths_lowercase(path_a: &Path, path_b: &Path) -> bool {
    path_to_key(path_a) == path_to_key(path_b)
}

fn path_to_key(path: &Path) -> PathKey {
    PathKey(
        path.to_str()
            .expect("Path is not valid ascii")
            .to_lowercase(),
    )
}

pub enum BrushVfs {
    Zip(ZipArchive<Cursor<ZipData>>),
    Manual(HashMap<PathKey, SharedRead>),
    #[cfg(not(target_family = "wasm"))]
    Directory(PathBuf, HashMap<PathBuf, PathKey>),
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

    async fn from_reader(
        reader: impl AsyncRead + SendNotWasm + Unpin + 'static,
    ) -> anyhow::Result<Self> {
        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let mut data = BufReader::new(reader);
        let peek = read_at_most(&mut data, 64).await?;
        let reader: Box<dyn DynRead> =
            Box::new(AsyncReadExt::chain(Cursor::new(peek.clone()), data));

        if peek.as_slice().starts_with(b"ply") {
            let mut map = HashMap::new();
            let key = PathKey("input.ply".to_owned());
            let reader = Arc::new(Mutex::new(Some(reader)));
            map.insert(key, reader);
            Ok(Self::Manual(map))
        } else if peek.starts_with(b"PK") {
            Self::from_zip_reader(reader)
                .await
                .map_err(|e| anyhow::anyhow!(e))
        } else if peek.starts_with(b"<!DOCTYPE html>") {
            // TODO: Display HTML page on WASM maybe?
            anyhow::bail!("Failed to download data.")
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

    // pub fn file_names(&self) -> impl Iterator<Item = PathBuf> + '_ {
    //     let iterator: Box<dyn Iterator<Item = &Path>> = match self {
    //         Self::Zip(archive) => Box::new(archive.file_names().map(Path::new)),
    //         Self::Manual(map) => Box::new(map.keys()),
    //         #[cfg(not(target_family = "wasm"))]
    //         Self::Directory(_, paths) => Box::new(paths.iter().map(|p| p.as_path())),
    //     };
    //     iterator.filter_map(|p| {
    //         if !p.starts_with("__MACOSX") {
    //             Some(p.clean())
    //         } else {
    //             None
    //         }
    //     })
    // }

    pub async fn reader_at_path(&self, path: &Path) -> anyhow::Result<Box<dyn DynRead>> {
        match self {
            Self::Zip(archive) => {
                // Zip file doesn't have a quick lookup. Maybe should add a HashMap<String, String> to go from
                // path key to name in zip file.
                let name = archive
                    .file_names()
                    .find(|name| compare_paths_lowercase(Path::new(name), path))
                    .ok_or(ZipError::FileNotFound)?;
                let name = name.to_owned();
                let mut buffer = vec![];
                // Archive is cheap to clone, as the data is an Arc<[u8]>.
                archive.clone().by_name(&name)?.read_to_end(&mut buffer)?;
                Ok(Box::new(Cursor::new(buffer)))
            }
            Self::Manual(map) => {
                let key = path_to_key(path);
                // Readers get taken out of the map as they are not cloneable.
                // This means that unlike other methods this path can only be loaded
                // once.
                let reader_mut = map.get(&key).context("File not found")?;
                let reader = reader_mut.lock().await.take();
                reader.context("Missing reader")
            }
            #[cfg(not(target_family = "wasm"))]
            Self::Directory(dir, _) => {
                // TODO: Use a string -> PathBuf cache.
                let total_path = dir.join(path);
                let file = tokio::fs::File::open(total_path).await?;
                let file = tokio::io::BufReader::new(file);
                Ok(Box::new(file))
            }
        }
    }
}
