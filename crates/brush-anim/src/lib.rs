//! Shared animation model for Brush: a serializable keyframe [`AnimationConfig`]
//! with camera interpolation (all platforms), plus the native render/encode
//! pipeline used by both the viewer's export and the headless CLI.

mod config;
pub use config::{AnimationConfig, Interpolation, Keyframe};

#[cfg(not(target_family = "wasm"))]
mod render;
#[cfg(not(target_family = "wasm"))]
pub use render::render_to_mp4;
