use bytes::Bytes;
use futures_channel;
use futures_util::StreamExt;
use js_sys::Uint8Array;
use std::io;
use std::path::PathBuf;
use tokio::io::AsyncRead;
use tokio_util::io::StreamReader;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

use wasm_streams::ReadableStream as WasmReadableStream;
use web_sys::{Blob, Event, HtmlAnchorElement, HtmlInputElement, ReadableStream};

use crate::PickFileError;

/// A handle to a directory picked by the user via the File System Access API.
/// Can be used to read files on demand by path.
#[derive(Clone)]
pub struct DirectoryHandle {
    handle: web_sys::FileSystemDirectoryHandle,
}

impl DirectoryHandle {
    /// Get a file handle for the given path within this directory.
    /// The path can contain subdirectories (e.g., "subdir/file.txt").
    pub async fn get_file(&self, path: &std::path::Path) -> Result<web_sys::File, PickFileError> {
        let components: Vec<_> = path.components().collect();
        if components.is_empty() {
            return Err(PickFileError::NoFileSelected);
        }

        // Navigate through subdirectories
        let mut current_dir = self.handle.clone();
        for component in &components[..components.len() - 1] {
            let dir_name = component
                .as_os_str()
                .to_str()
                .ok_or(PickFileError::NoFileSelected)?;
            let promise = current_dir.get_directory_handle(dir_name);
            let result = wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|_| PickFileError::NoFileSelected)?;
            current_dir = result.unchecked_into();
        }

        // Get the file from the final directory
        let file_name = components
            .last()
            .unwrap()
            .as_os_str()
            .to_str()
            .ok_or(PickFileError::NoFileSelected)?;

        let promise = current_dir.get_file_handle(file_name);
        let file_handle: web_sys::FileSystemFileHandle =
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|_| PickFileError::NoFileSelected)?
                .unchecked_into();

        let promise = file_handle.get_file();
        let file: web_sys::File = wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|_| PickFileError::NoFileSelected)?
            .unchecked_into();

        Ok(file)
    }

    /// List all files in the directory recursively.
    pub async fn list_files(&self) -> Result<Vec<PathBuf>, PickFileError> {
        let mut files = Vec::new();
        self.list_files_recursive(&self.handle, PathBuf::new(), &mut files)
            .await?;
        Ok(files)
    }

    fn list_files_recursive<'a>(
        &'a self,
        dir: &'a web_sys::FileSystemDirectoryHandle,
        prefix: PathBuf,
        files: &'a mut Vec<PathBuf>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PickFileError>> + 'a>> {
        Box::pin(async move {
            // Get async iterator from entries()
            let entries = js_sys::Reflect::get(dir, &"entries".into())
                .map_err(|_| PickFileError::NoDirectorySelected)?
                .dyn_into::<js_sys::Function>()
                .map_err(|_| PickFileError::NoDirectorySelected)?
                .call0(dir)
                .map_err(|_| PickFileError::NoDirectorySelected)?;

            loop {
                // Call next() on the async iterator
                let next_fn = js_sys::Reflect::get(&entries, &"next".into())
                    .map_err(|_| PickFileError::NoDirectorySelected)?
                    .dyn_into::<js_sys::Function>()
                    .map_err(|_| PickFileError::NoDirectorySelected)?;

                let promise = next_fn
                    .call0(&entries)
                    .map_err(|_| PickFileError::NoDirectorySelected)?
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| PickFileError::NoDirectorySelected)?;

                let result = wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|_| PickFileError::NoDirectorySelected)?;

                let done = js_sys::Reflect::get(&result, &"done".into())
                    .map_err(|_| PickFileError::NoDirectorySelected)?
                    .as_bool()
                    .unwrap_or(true);

                if done {
                    break;
                }

                let value = js_sys::Reflect::get(&result, &"value".into())
                    .map_err(|_| PickFileError::NoDirectorySelected)?;

                let array: js_sys::Array = value.unchecked_into();
                let name: String = array
                    .get(0)
                    .as_string()
                    .ok_or(PickFileError::NoDirectorySelected)?;
                let handle: web_sys::FileSystemHandle = array.get(1).unchecked_into();

                // Get kind as string property
                let kind = js_sys::Reflect::get(&handle, &"kind".into())
                    .map_err(|_| PickFileError::NoDirectorySelected)?
                    .as_string()
                    .ok_or(PickFileError::NoDirectorySelected)?;

                let path = prefix.join(&name);

                if kind == "file" {
                    files.push(path);
                } else if kind == "directory" {
                    let subdir: web_sys::FileSystemDirectoryHandle = handle.unchecked_into();
                    self.list_files_recursive(&subdir, path, files).await?;
                }
            }
            Ok(())
        })
    }
}

/// Pick a directory using the File System Access API (showDirectoryPicker).
/// Returns a DirectoryHandle that can be used to read files on demand.
pub async fn pick_directory_handle() -> Result<DirectoryHandle, PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoDirectorySelected)?;

    // Call showDirectoryPicker()
    let promise = js_sys::Reflect::get(&window, &"showDirectoryPicker".into())
        .map_err(|_| PickFileError::NoDirectorySelected)?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| PickFileError::NoDirectorySelected)?
        .call0(&window)
        .map_err(|_| PickFileError::NoDirectorySelected)?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| PickFileError::NoDirectorySelected)?;

    let handle = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| PickFileError::NoDirectorySelected)?
        .dyn_into::<web_sys::FileSystemDirectoryHandle>()
        .map_err(|_| PickFileError::NoDirectorySelected)?;

    Ok(DirectoryHandle { handle })
}

pub async fn save_file(default_name: &str, data: &[u8]) -> Result<(), PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoFileSelected)?;
    let document = window.document().ok_or(PickFileError::NoFileSelected)?;

    let array = Uint8Array::from(data);
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&array);

    let blob =
        Blob::new_with_u8_array_sequence(&blob_parts).map_err(|_| PickFileError::NoFileSelected)?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|_| PickFileError::NoFileSelected)?;

    let anchor = document
        .create_element("a")
        .map_err(|_| PickFileError::NoFileSelected)?
        .dyn_into::<HtmlAnchorElement>()
        .map_err(|_| PickFileError::NoFileSelected)?;

    anchor.set_href(&url);
    anchor.set_download(default_name);
    anchor.click();

    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

pub async fn pick_file() -> Result<crate::PickedFile<impl AsyncRead + Unpin>, PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoFileSelected)?;
    let document = window.document().ok_or(PickFileError::NoFileSelected)?;

    let input = document
        .create_element("input")
        .map_err(|_| PickFileError::NoFileSelected)?
        .dyn_into::<HtmlInputElement>()
        .map_err(|_| PickFileError::NoFileSelected)?;

    input.set_type("file");

    let (sender, receiver) = futures_channel::oneshot::channel();
    let sender = std::rc::Rc::new(std::cell::RefCell::new(Some(sender)));

    let onchange = {
        let sender = sender.clone();
        let input = input.clone();
        Closure::wrap(Box::new(move |_: Event| {
            if let Some(sender) = sender.borrow_mut().take() {
                let files = input.files();
                let _ = sender.send(files);
            }
        }) as Box<dyn FnMut(_)>)
    };

    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    input.click();

    let files = receiver
        .await
        .map_err(|_| PickFileError::NoFileSelected)?
        .ok_or(PickFileError::NoFileSelected)?;

    let file = files.get(0).ok_or(PickFileError::NoFileSelected)?;
    let name = file.name();

    let readable_stream: ReadableStream = file.stream();
    let wasm_stream = WasmReadableStream::from_raw(readable_stream);

    let byte_stream = wasm_stream.into_stream().map(|result| {
        result
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Stream error: {:?}", e)))
            .and_then(|chunk| {
                if let Ok(uint8_array) = chunk.dyn_into::<Uint8Array>() {
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

    Ok(crate::PickedFile {
        name,
        reader: StreamReader::new(byte_stream),
    })
}
