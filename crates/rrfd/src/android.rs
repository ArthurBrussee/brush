use anyhow::{Context, Result, anyhow};
use jni::JNIEnv;
use jni::objects::{GlobalRef, JClass, JStaticMethodID};
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
    static ref CHANNEL: RwLock<Option<Sender<Option<File>>>> = RwLock::new(None);
    static ref START_FILE_PICKER: RwLock<Option<JStaticMethodID>> = RwLock::new(None);
    static ref FILE_PICKER_CLASS: RwLock<Option<GlobalRef>> = RwLock::new(None);
}

#[allow(unused)]
pub fn jni_initialize(vm: Arc<jni::JavaVM>) {
    let mut env = vm.get_env().expect("Cannot get reference to the JNIEnv");
    let class = env.find_class("com/splats/app/FilePicker").unwrap();
    let method = env
        .get_static_method_id(&class, "startFilePicker", "()V")
        .unwrap();
    *FILE_PICKER_CLASS
        .write()
        .expect("Failed to write JNI data.") = Some(env.new_global_ref(class).unwrap());
    *START_FILE_PICKER
        .write()
        .expect("Failed to write JNI data.") = Some(method);
    *VM.write().unwrap() = Some(vm);
}

#[allow(unused)]
pub(crate) async fn pick_file() -> Result<File> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    {
        let channel = CHANNEL.write();
        if let Ok(mut channel) = channel {
            *channel = Some(sender);
        } else {
            anyhow::bail!("Failed to initialize file picker");
        }
    }

    // Call method. Be sure this is scoped so we drop all guards before waiting.
    {
        let java_vm = VM
            .read()
            .unwrap()
            .clone()
            .expect("Failed to initialize Java VM");
        let mut env = java_vm.attach_current_thread()?;

        let class = FILE_PICKER_CLASS
            .read()
            .expect("Failed to initialize FilePicker class");
        let method = START_FILE_PICKER
            .read()
            .expect("Failed to initialize FilePicker method");

        // SAFETY: This is safe as long as we cached the method in the right way, and
        // this matches the Java side. Not much more we can do here.
        let _ = unsafe {
            env.call_static_method_unchecked(
                class.as_ref().expect("Failed to get class reference"),
                method.as_ref().expect("Failed to get method reference"),
                jni::signature::ReturnType::Primitive(Primitive::Void),
                &[],
            )
        }?;
    }

    let file = receiver
        .recv()
        .await
        .ok_or(anyhow!("Failed to receive anything"));

    let file = file?;
    file.context("No file selected")
}

#[no_mangle]
extern "system" fn Java_com_splats_app_FilePicker_onFilePickerResult<'local>(
    _env: JNIEnv<'local>,
    _class: JClass<'local>,
    fd: jint,
) {
    let file = if fd < 0 {
        None
    } else {
        // Convert the raw file descriptor into a Rust File
        // SAFETY: Pray that JNI gets us a valid file. It will be open
        // when passed to us.
        Some(unsafe { tokio::fs::File::from_raw_fd(fd) })
    };

    // Channel can be gone before the callback if other parts of pick_file fail.
    if let Ok(ch) = CHANNEL.read() {
        if let Some(ch) = ch.as_ref() {
            ch.try_send(file)
                .expect("Failed to send file picking result");
        }
    }
}
