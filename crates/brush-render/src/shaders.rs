//! Host-side mirrors of constants the kernels share with the host. The
//! cube `ProjectUniforms` launch arg is built directly from `Camera` via
//! [`crate::camera::Camera::to_project_uniforms_launch`]; per-render
//! state travels across the burn backward boundary as
//! [`crate::render_aux::RenderState`].

pub const SH_C0: f32 = 0.282_094_8;

pub mod helpers {
    pub const TILE_WIDTH: u32 = 16;
    pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;
}
