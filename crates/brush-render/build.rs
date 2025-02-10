use miette::IntoDiagnostic;

fn main() -> miette::Result<()> {
    brush_wgsl::build_modules(
        &[
            "src/shaders/project_forward.wgsl",
            "src/shaders/project_visible.wgsl",
            "src/shaders/map_gaussian_to_intersects.wgsl",
            "src/shaders/rasterize.wgsl",
        ],
        &["src/shaders/helpers.wgsl"],
        "src/shaders/mod.rs",
    )
    .into_diagnostic()
}
