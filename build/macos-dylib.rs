fn main() {
    println!("cargo::rerun-if-changed=../../build/macos-dylib.rs");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // CubeCL uses buildid 1.0.5, whose Mach-O reader unconditionally
        // references the executable header. A cdylib has a dylib header instead.
        // Alias that image's header so buildid reads the library's own LC_UUID,
        // preserving cache invalidation when the library is rebuilt.
        // Scope this to cdylibs: executables already define their own header.
        println!("cargo::rustc-link-arg-cdylib=-Wl,-alias,__mh_dylib_header,__mh_execute_header");
    }
}
