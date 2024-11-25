use anyhow::Result;
use std::path::PathBuf;
use std::sync::RwLock;
use tokio::sync::mpsc::Sender;
use lazy_static::lazy_static;
use objc::runtime::{YES, Object, Class};
use objc::{msg_send, sel, sel_impl};
use std::ptr::null_mut;

lazy_static! {
    static ref CHANNEL: RwLock<Option<Sender<Result<PickedFile>>>> = RwLock::new(None);
}

#[derive(Clone)]
pub struct PickedFile {
    pub path: PathBuf,
    pub file_name: String,
    pub data: Vec<u8>,
}

impl PickedFile {
    pub fn new(path: PathBuf, file_name: String, data: Vec<u8>) -> Self {
        Self { path, file_name, data }
    }

    pub async fn read(&self) -> Result<Vec<u8>> {
        Ok(self.data.clone())
    }
}

#[repr(C)]
pub struct NSArray;

#[cfg(target_os = "ios")]
pub async fn pick_file() -> Result<PickedFile> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    {
        let mut channel = CHANNEL.write().unwrap();
        *channel = Some(sender);
    }

    unsafe {
        let picker: *mut Object = msg_send![Class::get("UIDocumentPickerViewController").unwrap(), alloc];
        let _: () = msg_send![picker, initForOpeningContentTypes:null_mut::<NSArray>()];
        let _: () = msg_send![picker, setModalPresentationStyle:0];
        let _: () = msg_send![picker, presentViewController:picker animated:YES completion:null_mut::<()>()];
    }

    receiver.recv().await.ok_or_else(|| anyhow::anyhow!("Failed to receive file"))?
}

#[no_mangle]
pub extern "C" fn ios_file_picker_callback(path: *const i8, data: *const u8, len: usize) {
    let path = unsafe { std::ffi::CStr::from_ptr(path) }
        .to_string_lossy()
        .into_owned();
    let data = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    
    let file_name = std::path::Path::new(&path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    
    let picked_file = PickedFile::new(
        PathBuf::from(path), 
        file_name,
        data
    );
    
    if let Ok(channel) = CHANNEL.read() {
        if let Some(sender) = channel.as_ref() {
            let _ = sender.try_send(Ok(picked_file));
        }
    }
}
