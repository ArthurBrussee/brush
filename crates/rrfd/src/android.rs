use jni::JNIEnv;
use jni::objects::{GlobalRef, JClass, JStaticMethodID, JString};
use jni::signature::Primitive;
use jni::sys::jint;
use lazy_static::lazy_static;
use std::os::fd::FromRawFd;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::fs::File;
use tokio::sync::mpsc::Sender;

lazy_static! {
    static ref VM: RwLock<Option<Arc<jni::JavaVM>>> = RwLock::new(None);

    // ✅ Now sends (File, filename)
    static ref CHANNEL: RwLock<Option<Sender<Option<(File, String)>>>> =
        RwLock::new(None);

    static ref START_FILE_PICKER: RwLock<Option<JStaticMethodID>> =
        RwLock::new(None);

    static ref FILE_PICKER_CLASS: RwLock<Option<GlobalRef>> =
        RwLock::new(None);
}

pub fn jni_initialize(vm: Arc<jni::JavaVM>) {
    let mut env = vm.get_env().expect("Cannot get JNIEnv");

    let class = env.find_class("com/splats/app/FilePicker").unwrap();

    let method = env
        .get_static_method_id(&class, "startFilePicker", "()V")
        .unwrap();

    *FILE_PICKER_CLASS.write().unwrap() =
        Some(env.new_global_ref(class).unwrap());

    *START_FILE_PICKER.write().unwrap() = Some(method);

    *VM.write().unwrap() = Some(vm);
}

pub(crate) async fn pick_file() -> std::io::Result<(File, String)> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);

    {
        let mut channel = CHANNEL.write().unwrap();
        *channel = Some(sender);
    }

    {
        let java_vm = VM
            .read()
            .unwrap()
            .clone()
            .expect("Java VM not initialized");

        let mut env = java_vm.attach_current_thread().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("JNI error: {:?}", e),
            )
        })?;

        let class = FILE_PICKER_CLASS.read().unwrap();
        let method = START_FILE_PICKER.read().unwrap();

        unsafe {
            env.call_static_method_unchecked(
                class.as_ref().unwrap(),
                method.as_ref().unwrap(),
                jni::signature::ReturnType::Primitive(Primitive::Void),
                &[],
            )
        }
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("JNI error: {:?}", e),
            )
        })?;
    }

    let result = receiver.recv().await.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to receive file picker result",
        )
    })?;

    match result {
        Some((file, name)) => Ok((file, name)),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No file selected",
        )),
    }
}

#[unsafe(no_mangle)]
extern "system" fn Java_com_splats_app_FilePicker_onFilePickerResult<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    fd: jint,
    name: JString<'local>,
) {
    let filename: String = match env.get_string(&name) {
        Ok(s) => s.into(),
        Err(_) => "file".to_string(),
    };

    let file = if fd < 0 {
        None
    } else {
        let std_file = unsafe { std::fs::File::from_raw_fd(fd) };
        Some((File::from_std(std_file), filename))
    };

    if let Ok(ch) = CHANNEL.read() {
        if let Some(ch) = ch.as_ref() {
            let _ = ch.try_send(file);
        }
    }
}

