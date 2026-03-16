// =============================================================================
// crates/brush-sfm/build.rs
// =============================================================================
// Cargo build script — runs before the crate is compiled.
//
// When `--features opencv-sfm` is set:
//   1. Compiles cpp/opencv_sfm_wrapper.cpp → libopencv_sfm_wrapper.a
//   2. Emits cargo:rustc-link-lib directives for the 5 minimal OpenCV .a files
//      produced by scripts/build_opencv_arm64.sh
//
// ENVIRONMENT VARIABLES:
//   OPENCV_ANDROID_DIR   Path to opencv_android_arm64/ (from build_opencv_arm64.sh)
//                        Defaults to ../../opencv_android_arm64 relative to crate root.
//
// TARGETS SUPPORTED:
//   aarch64-linux-android   Pixel 9a via cargo-ndk
//   x86_64-unknown-linux-gnu  Desktop (CI, development)
//   aarch64-apple-darwin      M-series Mac development
//   WASM: feature is disabled, build.rs is a no-op
// =============================================================================

fn main() {
    // Only link OpenCV when the feature is requested
    if std::env::var("CARGO_FEATURE_OPENCV_SFM").is_ok() {
        build_opencv_wrapper();
    }
}

fn build_opencv_wrapper() {
    use std::{env, path::PathBuf};

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os   = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let manifest    = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // ── Locate pre-built OpenCV ARM64 directory ────────────────────────────
    // Set OPENCV_ANDROID_DIR to the output of scripts/build_opencv_arm64.sh
    let opencv_root = env::var("OPENCV_ANDROID_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default: look relative to the workspace root
            manifest.parent()          // crates/
                .and_then(|p| p.parent())  // workspace root
                .map(|r| r.join("opencv_android_arm64"))
                .unwrap_or_else(|| PathBuf::from("opencv_android_arm64"))
        });

    // ── Header include directory ───────────────────────────────────────────
    // build_opencv_arm64.sh installs headers at:
    //   sdk/native/jni/include/     (Android NDK layout)
    //   include/opencv4/            (desktop install layout)
    let include_dir = {
        let jni = opencv_root.join("sdk/native/jni/include");
        let alt = opencv_root.join("include/opencv4");
        if jni.exists() { jni } else { alt }
    };

    // ── Static library directory ───────────────────────────────────────────
    let lib_dir = {
        let android_arm = opencv_root.join("sdk/native/staticlibs/arm64-v8a");
        let android_x86 = opencv_root.join("sdk/native/staticlibs/x86_64");
        let desktop     = opencv_root.join("lib");
        if android_arm.exists() {
            android_arm
        } else if android_x86.exists() {
            android_x86
        } else {
            desktop
        }
    };

    // Tell Cargo to re-run if these change
    println!("cargo:rerun-if-changed=cpp/opencv_sfm_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/opencv_sfm_wrapper.h");
    println!("cargo:rerun-if-env-changed=OPENCV_ANDROID_DIR");

    // ── Compile the C++ wrapper ────────────────────────────────────────────
    // The `cc` crate handles cross-compilation flags automatically when
    // invoked via cargo-ndk for the Android target.
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("cpp/opencv_sfm_wrapper.cpp")
        .include("cpp/")              // opencv_sfm_wrapper.h
        .include(&include_dir)        // opencv2/ headers
        // ARM64 NEON is already baked into the OpenCV we built — no extra flags
        .flag_if_supported("-fPIC")
        // Suppress noisy warnings from OpenCV's own headers
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers");

    build.compile("opencv_sfm_wrapper");
    // `compile()` automatically emits:
    //   cargo:rustc-link-lib=static=opencv_sfm_wrapper

    // ── Link the 5 minimal OpenCV static libraries (in dependency order) ──
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    for lib in &[
        "opencv_calib3d",    // findEssentialMat, recoverPose, triangulatePoints
        "opencv_features2d", // SIFT, FlannBasedMatcher
        "opencv_flann",      // KDTree index (needed by features2d)
        "opencv_imgproc",    // cvtColor, etc. (needed by calib3d/features2d)
        "opencv_core",       // Mat, Vector — everything depends on this
    ] {
        println!("cargo:rustc-link-lib=static={}", lib);
    }

    // ── Platform-specific system libraries ────────────────────────────────
    match target_os.as_str() {
        "android" => {
            println!("cargo:rustc-link-lib=log");   // Android logging
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=c++_static");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=c++");
        }
        _ => {}
    }

    println!(
        "cargo:warning=brush-sfm: OpenCV wrapper compiled. \
         arch={} os={} include={:?} libs={:?}",
        target_arch, target_os, include_dir, lib_dir
    );
}
