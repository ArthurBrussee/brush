use super::shaders::{map_gaussian_to_intersects, project_forward, project_visible, rasterize};
use brush_kernel::kernel_source_gen;

kernel_source_gen!(ProjectSplats {}, project_forward);
kernel_source_gen!(ProjectVisible {}, project_visible);
kernel_source_gen!(
    MapGaussiansToIntersect { prepass },
    map_gaussian_to_intersects
);
kernel_source_gen!(Rasterize { bwd_info, webgpu }, rasterize);
