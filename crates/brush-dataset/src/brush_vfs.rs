//
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
use tokio::io::AsyncReadExt;
use tokio::{io::AsyncRead, sync::Mutex};
use zip::{
    result::{ZipError, ZipResult},
    ZipArchive,
};

#[derive(Clone)]
pub struct ZipData {
    data: Arc<Vec<u8>>,
}

type ZipReader = Cursor<ZipData>;

impl AsRef<[u8]> for ZipData {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl ZipData {
    pub fn open_for_read(&self) -> ZipReader {
        Cursor::new(self.clone())
    }
}

impl From<Vec<u8>> for ZipData {
    fn from(value: Vec<u8>) -> Self {
        Self {
            data: Arc::new(value),
        }
    }
}

pub(crate) fn normalized_path(path: &Path) -> PathBuf {
    Path::new(path)
        .components()
        .skip_while(|c| matches!(c, std::path::Component::CurDir))
        .collect::<PathBuf>()
}

#[derive(Clone)]
pub struct ConsumableReader {
    // Option allows us to take ownership when first used
    inner: Arc<Mutex<Option<Box<dyn AsyncRead + Send + Unpin>>>>,
}

#[derive(Clone, Default)]
pub struct PathReader {
    paths: HashMap<PathBuf, ConsumableReader>,
}

impl PathReader {
    fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.paths.keys()
    }

    pub fn add(&mut self, path: PathBuf, reader: impl AsyncRead + Send + Unpin + 'static) {
        self.paths.insert(
            path,
            ConsumableReader {
                inner: Arc::new(Mutex::new(Some(Box::new(reader)))),
            },
        );
    }

    async fn open(&mut self, path: &Path) -> anyhow::Result<Box<dyn AsyncRead + Send + Unpin>> {
        let entry = self.paths.remove(path).context("File not found")?;
        let reader = entry.inner.lock().await.take();
        reader.context("Missing reader")
    }
}

#[derive(Clone)]
pub enum BrushVfs {
    Zip(ZipArchive<Cursor<ZipData>>),
    Manual(PathReader),
    Directory(PathBuf, Vec<PathBuf>),
}

// TODO: This is all awfully ad-hoc.
impl BrushVfs {
    pub async fn from_zip_reader(reader: impl AsyncRead + Unpin) -> ZipResult<Self> {
        let mut bytes = vec![];
        let mut reader = reader;
        reader.read_to_end(&mut bytes).await?;

        let zip_data = ZipData::from(bytes);
        let archive = ZipArchive::new(zip_data.open_for_read())?;
        Ok(BrushVfs::Zip(archive))
    }

    pub fn from_paths(paths: PathReader) -> Self {
        BrushVfs::Manual(paths)
    }

    pub async fn from_directory(dir: &Path) -> anyhow::Result<Self> {
        let mut read = ::tokio::fs::read_dir(dir).await?;
        let mut paths = vec![];
        while let Some(entry) = read.next_entry().await? {
            paths.push(entry.path());
        }
        Ok(BrushVfs::Directory(dir.to_path_buf(), paths))
    }

    pub(crate) fn file_names(&self) -> impl Iterator<Item = &str> + '_ {
        let iterator: Box<dyn Iterator<Item = &str>> = match self {
            BrushVfs::Zip(archive) => Box::new(archive.file_names()),
            BrushVfs::Manual(map) => Box::new(map.paths().filter_map(|k| k.to_str())),
            BrushVfs::Directory(_, paths) => Box::new(paths.iter().filter_map(|k| k.to_str())),
        };
        // stupic macOS.
        iterator.filter(|p| !p.contains("__MACOSX"))
    }

    pub(crate) fn find_with_extension(
        &self,
        extension: &str,
        contains: &str,
    ) -> anyhow::Result<PathBuf> {
        let names: Vec<_> = self
            .file_names()
            .filter(|name| name.ends_with(extension))
            .collect();

        if names.len() == 1 {
            return Ok(Path::new(names[0]).to_owned());
        }

        let names: Vec<_> = names
            .iter()
            .filter(|name| name.contains(contains))
            .collect();

        if names.len() == 1 {
            return Ok(Path::new(names[0]).to_owned());
        }

        anyhow::bail!("Failed to find file ending in {extension} maybe containing {contains}.");
    }

    pub(crate) async fn open_reader_at_path(
        &mut self,
        path: &Path,
    ) -> anyhow::Result<Box<dyn AsyncRead + Send + Unpin>> {
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
            BrushVfs::Directory(path_buf, _) => {
                let file = tokio::fs::File::open(path_buf).await?;
                Ok(Box::new(file))
            }
        }
    }

    pub(crate) fn find_base_path(&self, search_path: &str) -> Option<PathBuf> {
        for file in self.file_names() {
            let path = normalized_path(Path::new(file));
            if path.ends_with(search_path) {
                return path
                    .ancestors()
                    .nth(Path::new(search_path).components().count())
                    .map(|x| x.to_owned());
            }
        }
        None
    }
}
