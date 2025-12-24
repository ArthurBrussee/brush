use brush_kernel::wgsl_kernel;

// Generate shared types and constants from helpers.wgsl (no entry point)
#[wgsl_kernel(source = "src/shaders/helpers.wgsl")]
pub struct Helpers;

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

// Re-export module-specific constants
pub const SH_C0: f32 = ProjectVisible::SH_C0;
