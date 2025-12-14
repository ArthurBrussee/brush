//! Build script to clean up old generated shader files.
//!
//! Previously, shaders were generated via build.rs into src/shaders/mod.rs files.
//! Now we use proc macros instead. This script removes any leftover generated files
//! to avoid confusion or compilation issues.

use std::path::Path;

fn main() {
    // Only rerun if this build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Clean up old generated shader files in the workspace
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crates_dir = Path::new(&manifest_dir).parent().unwrap();

    // List of crates that used to have generated shaders/mod.rs
    let crates_with_old_shaders = [
        "brush-kernel",
        "brush-prefix-sum",
        "brush-sort",
        "brush-render",
        "brush-render-bwd",
    ];

    for crate_name in crates_with_old_shaders {
        let old_file = crates_dir.join(crate_name).join("src/shaders/mod.rs");
        if old_file.exists() {
            println!(
                "cargo:warning=Removing old generated file: {}",
                old_file.display()
            );
            let _ = std::fs::remove_file(&old_file);
        }
    }
}
