use std::env;
use std::fs;
use std::path::PathBuf;

const COPY_NAME: &str = "burn_mapped.mpk.gz";

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR environment variable not set");
    // Navigate up from OUT_DIR to the target directory
    let mut target_dir = PathBuf::from(&out_dir);
    target_dir.pop(); // out
    target_dir.pop(); // hash
    target_dir.pop(); // hash
    let dest_path = target_dir.join(COPY_NAME);
    let in_dir = std::env::current_dir().expect("No cwd").join(COPY_NAME);
    fs::copy(&in_dir, &dest_path).expect("Failed to copy file");
}
