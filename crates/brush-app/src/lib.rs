#![recursion_limit = "256"]
#![cfg(target_os = "android")]

mod ui_process;

use brush_ui::app::App;
use jni::sys::{JNI_VERSION_1_6, jint};
use std::os::raw::c_void;
use std::sync::Arc;
use ui_process::UiProcess;

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(vm: jni::JavaVM, _: *mut c_void) -> jint {
    let vm_ref = Arc::new(vm);
    rrfd::android::jni_initialize(vm_ref);
    JNI_VERSION_1_6
}

#[unsafe(no_mangle)]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    let context = Arc::new(UiProcess::new());

    let wgpu_options = brush_ui::create_egui_options();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

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
            Box::new(|cc| Ok(Box::new(App::new(cc, None, context)))),
        )
        .unwrap();
    });
}
