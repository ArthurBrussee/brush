//! Shared `CubeType` aggregates used by the render kernels. Generic
//! cube-side math (`Vec3A`, `Quat`, `Mat3`, `Sym2`, …) lives in
//! [`brush_cube`]; this module hosts the render-specific aggregates and
//! re-exports the math types for convenience.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

use crate::kernels::camera_model::CameraParams;
pub use brush_cube::{Mat3, Mat2x3, PixelRect, Quat, Sym2, TileBbox, Vec3A};

/// One projected splat as the kernel sees it. The on-device storage is
/// a flat `Tensor<f32>` of `9 * num_visible` lanes (see
/// `helpers::PROJECTED_LANES`); the load helper packages the lanes into
/// this struct so consumers don't carry nine independent locals.
#[derive(CubeType, CubeTypeMut, Copy, Clone)]
pub struct Splat {
    pub xy_x: f32,
    pub xy_y: f32,
    pub conic_x: f32,
    pub conic_y: f32,
    pub conic_z: f32,
    pub color_a: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
}

#[cube]
impl Splat {
    pub fn zero() -> Splat {
        Splat {
            xy_x: 0.0f32,
            xy_y: 0.0f32,
            conic_x: 0.0f32,
            conic_y: 0.0f32,
            conic_z: 0.0f32,
            color_a: 0.0f32,
            color_r: 0.0f32,
            color_g: 0.0f32,
            color_b: 0.0f32,
        }
    }
}

/// Project & visible-pass uniforms. The kernel only needs the top 3
/// rows of the world-to-camera viewmat (the bottom row is `(0, 0, 0, 1)`),
/// so we ship 12 scalars instead of 16.
#[derive(CubeLaunch, CubeType, Clone, Copy)]
pub struct ProjectUniforms {
    // 3x4 view matrix, column-major. `vm{i}_*` is column i, fields are
    // (x, y, z) of that column.
    pub vm0_x: f32,
    pub vm0_y: f32,
    pub vm0_z: f32,
    pub vm1_x: f32,
    pub vm1_y: f32,
    pub vm1_z: f32,
    pub vm2_x: f32,
    pub vm2_y: f32,
    pub vm2_z: f32,
    pub vm3_x: f32,
    pub vm3_y: f32,
    pub vm3_z: f32,
    pub camera: CameraParams,
    pub camera_x: f32,
    pub camera_y: f32,
    pub camera_z: f32,
    pub img_w: u32,
    pub img_h: u32,
    pub tile_bw: u32,
    pub tile_bh: u32,
    pub sh_degree: u32,
    pub total_splats: u32,
    pub num_visible: u32,
}

#[cube]
impl ProjectUniforms {
    /// Top-left 3x3 of the world-to-cam viewmat.
    pub fn view_rotation(self) -> Mat3 {
        Mat3 {
            c0_x: self.vm0_x,
            c0_y: self.vm0_y,
            c0_z: self.vm0_z,
            c1_x: self.vm1_x,
            c1_y: self.vm1_y,
            c1_z: self.vm1_z,
            c2_x: self.vm2_x,
            c2_y: self.vm2_y,
            c2_z: self.vm2_z,
        }
    }

    /// Translation column of the world-to-cam viewmat.
    pub fn view_translation(self) -> Vec3A {
        Vec3A::new(self.vm3_x, self.vm3_y, self.vm3_z)
    }

    pub fn camera_pos(self) -> Vec3A {
        Vec3A::new(self.camera_x, self.camera_y, self.camera_z)
    }
}

/// Rasterize-pass uniforms.
#[derive(CubeLaunch, CubeType, Clone, Copy)]
pub struct RasterizeUniforms {
    pub tile_bw: u32,
    pub img_w: u32,
    pub img_h: u32,
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
}
