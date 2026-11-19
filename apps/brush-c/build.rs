fn main() {
    // cubecl's build stamp reads the main executable's Mach-O header
    // (`_mh_execute_header`), which only exists when linking an executable.
    // Resolve it when the dylib is loaded instead, so it finds the host app's.
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-U,__mh_execute_header");
    }
}
