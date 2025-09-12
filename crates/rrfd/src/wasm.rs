use bytes::Bytes;
use futures_channel;
use futures_util::StreamExt;
use js_sys::Uint8Array;
use std::{collections::HashMap, io, path::PathBuf};
use tokio::io::AsyncRead;
use tokio_util::io::StreamReader;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

use wasm_streams::ReadableStream as WasmReadableStream;
use web_sys::{Blob, Event, HtmlAnchorElement, HtmlInputElement, ReadableStream};

use crate::PickFileError;

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

pub async fn pick_file() -> Result<impl AsyncRead + Unpin, PickFileError> {
    let files = pick_files(false).await?;
    let file = files.get(0).ok_or(PickFileError::NoFileSelected)?;

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

    Ok(StreamReader::new(byte_stream))
}

pub async fn pick_directory() -> Result<PathBuf, PickFileError> {
    let files = pick_files(true).await?;

    if files.length() == 0 {
        return Err(PickFileError::NoDirectorySelected);
    }

    let first_file = files.get(0).ok_or(PickFileError::NoDirectorySelected)?;
    let path_value = js_sys::Reflect::get(&first_file, &"webkitRelativePath".into())
        .map_err(|_| PickFileError::NoDirectorySelected)?;
    let path_str = path_value
        .as_string()
        .ok_or(PickFileError::NoDirectorySelected)?;

    let path = PathBuf::from(path_str);
    Ok(path.parent().unwrap_or(&path).to_path_buf())
}

pub async fn pick_directory_files() -> Result<HashMap<PathBuf, web_sys::File>, PickFileError> {
    let files = pick_files(true).await?;
    let mut file_map = HashMap::new();

    for i in 0..files.length() {
        if let Some(file) = files.get(i) {
            // Get the relative path
            let path_value = js_sys::Reflect::get(&file, &"webkitRelativePath".into())
                .map_err(|_| PickFileError::NoDirectorySelected)?;
            let path_str = path_value
                .as_string()
                .ok_or(PickFileError::NoDirectorySelected)?;

            let path = PathBuf::from(path_str);
            file_map.insert(path, file);
        }
    }

    Ok(file_map)
}

async fn pick_files(directory: bool) -> Result<web_sys::FileList, PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoFileSelected)?;
    let document = window.document().ok_or(PickFileError::NoFileSelected)?;

    let input = document
        .create_element("input")
        .map_err(|_| PickFileError::NoFileSelected)?
        .dyn_into::<HtmlInputElement>()
        .map_err(|_| PickFileError::NoFileSelected)?;

    input.set_type("file");
    input.set_multiple(true);

    if directory {
        input
            .set_attribute("webkitdirectory", "")
            .map_err(|_| PickFileError::NoDirectorySelected)?;
    }

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

    let files = receiver.await.map_err(|_| PickFileError::NoFileSelected)?;
    files.ok_or(if directory {
        PickFileError::NoDirectorySelected
    } else {
        PickFileError::NoFileSelected
    })
}
