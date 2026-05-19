//! Host-side mirrors of structs/constants the kernels share with the
//! host. The `CubeCL` kernels take scalars individually via
//! `*Launch::new(...)`; these structs travel across the burn backward
//! boundary as a single argument.

pub const SH_C0: f32 = 0.282_094_8;

pub mod helpers {
    use crate::kernels::camera_model::pinhole::PinholeParams;
    use crate::kernels::camera_model::{CameraModel, JacobianClampLimits};
    use crate::kernels::types::ProjectUniformsLaunch;
    use burn_cubecl::cubecl::wgpu::WgpuRuntime;

    pub const TILE_WIDTH: u32 = 16;
    pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

    #[derive(Debug, Clone, Copy)]
    pub struct ProjectUniforms {
        pub viewmat: [[f32; 4]; 4],
        pub camera_model: CameraModel,
        pub pinhole_params: PinholeParams,
        pub img_size: [u32; 2],
        pub tile_bounds: [u32; 2],
        pub camera_position: [f32; 4],
        pub sh_degree: u32,
        pub total_splats: u32,
        pub num_visible: u32,

        // precomputed limits used for clamping the projection Jacobian
        pub jacobian_clamp_limits: JacobianClampLimits,
    }

    impl ProjectUniforms {
        /// Build the cube-side `ProjectUniforms` launch arg from the camera + img
        /// dims. Shared by the forward and backward projection passes.
        pub fn to_launch_object(&self) -> ProjectUniformsLaunch<WgpuRuntime> {
            ProjectUniformsLaunch::new(
                self.viewmat[0][0],
                self.viewmat[0][1],
                self.viewmat[0][2],
                self.viewmat[1][0],
                self.viewmat[1][1],
                self.viewmat[1][2],
                self.viewmat[2][0],
                self.viewmat[2][1],
                self.viewmat[2][2],
                self.viewmat[3][0],
                self.viewmat[3][1],
                self.viewmat[3][2],
                self.pinhole_params.to_launch_object(),
                self.jacobian_clamp_limits.to_launch_object(),
                self.camera_position[0],
                self.camera_position[1],
                self.camera_position[2],
                self.img_size[0],
                self.img_size[1],
                self.tile_bounds[0],
                self.tile_bounds[1],
                self.sh_degree,
                self.total_splats,
                self.num_visible,
            )
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RasterizeUniforms {
        pub tile_bounds: [u32; 2],
        pub img_size: [u32; 2],
        pub background: [f32; 4],
    }
}
