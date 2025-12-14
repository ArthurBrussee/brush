#![recursion_limit = "256"]

#[cfg(feature = "training")]
mod ffi;

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
#[cfg(target_os = "android")]
pub extern "system" fn JNI_OnLoad(vm: jni::JavaVM, _: *mut c_void) -> jni::sys::jint {
    let vm_ref = Arc::new(vm);
    rrfd::android::jni_initialize(vm_ref);
    jni::sys::JNI_VERSION_1_6
}

#[unsafe(no_mangle)]
#[cfg(target_os = "android")]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use brush_ui::app::App;
    use brush_ui::ui_process::UiProcess;
    use std::os::raw::c_void;
    use std::sync::Arc;

    let wgpu_options = brush_ui::create_egui_options();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let context = Arc::new(UiProcess::new());

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
            Box::new(|cc| Ok(Box::new(App::new(cc, context)))),
        )
        .unwrap();
    });
}
