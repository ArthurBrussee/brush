#![cfg(target_os = "android")]

use jni::sys::{JNI_VERSION_1_6, jint};
use std::os::raw::c_void;
use std::sync::Arc;

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(vm: jni::JavaVM, _: *mut c_void) -> jint {
    let vm_ref = Arc::new(vm);
    rrfd::android::jni_initialize(vm_ref);
    JNI_VERSION_1_6
}

#[unsafe(no_mangle)]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let wgpu_options = brush_ui::create_egui_options();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    // Unused.
    #[allow(unused)]
    let (send, rec) = tokio::sync::oneshot::channel();

    runtime.block_on(async {
        android_logger::init_once(
            android_logger::Config::default().with_max_level(log::LevelFilter::Info),
        );

        eframe::run_native(
            "Brush",
            eframe::NativeOptions {
                // Build app display.
                viewport: egui::ViewportBuilder::default(),
                android_app: Some(app),
                wgpu_options,
                ..Default::default()
            },
            Box::new(|cc| Ok(Box::new(brush_app::App::new(cc, send, None)))),
        )
        .unwrap();
    });
}
