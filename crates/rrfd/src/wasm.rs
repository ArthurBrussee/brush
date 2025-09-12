use bytes::Bytes;
use futures_util::StreamExt;
use js_sys::Uint8Array;
use std::{io, path::PathBuf};
use tokio::io::AsyncRead;
use tokio_util::io::StreamReader;
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::JsFuture;
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

async fn pick_files(directory: bool) -> Result<web_sys::FileList, PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoFileSelected)?;
    let document = window.document().ok_or(PickFileError::NoFileSelected)?;

    let input = document
        .create_element("input")
        .map_err(|_| PickFileError::NoFileSelected)?
        .dyn_into::<HtmlInputElement>()
        .map_err(|_| PickFileError::NoFileSelected)?;

    input.set_type("file");
    if directory {
        let _ = input.set_attribute("webkitdirectory", "");
    }

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let closure = Closure::once_into_js(move |event: Event| {
            if let Some(target) = event.target() {
                if let Ok(input) = target.dyn_into::<HtmlInputElement>() {
                    if let Some(files) = input.files() {
                        let _ = resolve.call1(&JsValue::UNDEFINED, &files);
                        return;
                    }
                }
            }
            let _ = reject.call1(&JsValue::UNDEFINED, &JsValue::NULL);
        });

        let _ = input.add_event_listener_with_callback("change", closure.as_ref().unchecked_ref());
        input.click();
    });

    let result = JsFuture::from(promise)
        .await
        .map_err(|_| PickFileError::NoFileSelected)?;
    result
        .dyn_into::<web_sys::FileList>()
        .map_err(|_| PickFileError::NoFileSelected)
}
