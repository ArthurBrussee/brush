//! Host-side mirrors of structs/constants the kernels share with the
//! host. The `CubeCL` kernels take scalars individually via
//! `*Launch::new(...)`; these structs travel across the burn backward
//! boundary as a single argument.

pub const SH_C0: f32 = 0.282_094_8;

pub mod helpers {
    pub const TILE_WIDTH: u32 = 16;
    pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

    #[derive(Debug, Clone, Copy)]
    pub struct ProjectUniforms {
        pub viewmat: [[f32; 4]; 4],
        pub focal: [f32; 2],
        pub img_size: [u32; 2],
        pub tile_bounds: [u32; 2],
        pub pixel_center: [f32; 2],
        pub camera_position: [f32; 4],
        pub sh_degree: u32,
        pub total_splats: u32,
        pub num_visible: u32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RasterizeUniforms {
        pub tile_bounds: [u32; 2],
        pub img_size: [u32; 2],
        pub background: [f32; 4],
    }
}
