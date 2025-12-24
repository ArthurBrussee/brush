use brush_kernel::wgsl_kernel;

// Define kernels using proc macro

#[wgsl_kernel(source = "src/shaders/project_forward.wgsl")]
pub struct ProjectSplats {
    mip_splatting: bool,
}

#[wgsl_kernel(source = "src/shaders/project_visible.wgsl")]
pub struct ProjectVisible {
    mip_splatting: bool,
}

#[wgsl_kernel(source = "src/shaders/map_gaussian_to_intersects.wgsl")]
pub struct MapGaussiansToIntersect {
    pub prepass: bool,
}

#[wgsl_kernel(source = "src/shaders/rasterize.wgsl")]
pub struct Rasterize {
    pub bwd_info: bool,
    pub webgpu: bool,
}

// Re-export helper types and constants from the kernel modules that use them
pub mod helpers {
    // Types used by multiple shaders - available from project_visible
    pub use super::project_visible::PackedVec3;
    pub use super::project_visible::ProjectedSplat;
    pub use super::project_visible::RenderUniforms;

    // Constants are now associated with the kernel structs
    pub const COV_BLUR: f32 = super::ProjectVisible::COV_BLUR;
    pub const TILE_SIZE: u32 = super::Rasterize::TILE_SIZE;
    pub const TILE_WIDTH: u32 = super::Rasterize::TILE_WIDTH;
}

// Re-export module-specific constants
pub const SH_C0: f32 = ProjectVisible::SH_C0;
