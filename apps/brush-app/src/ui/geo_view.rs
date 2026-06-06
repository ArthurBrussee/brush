//! Viewer-side RaDe-GS geometry visualization: render the geometry channels and
//! colormap them for display (absolute jet depth / hemisphere normals).

use brush_render::{
    burn_glue::geo_colormap_pack,
    camera::Camera,
    gaussian_splats::{Splats, TextureMode, render_splats},
    geo::{rendered_depth, rendered_normal},
};
use burn::tensor::{Tensor, s};
use glam::{UVec2, Vec3};

use crate::ui::app::ViewChannel;

/// Colormap a geometry render into a packed RGBA8 image for the given viewer
/// channel. `img` is the full `[H, W, 8]` render (`rgba` then `Nx,Ny,Nz,depth`).
/// Depth uses an absolute `[0, max_meters]` jet ramp; normals are mapped to
/// the RGB hemisphere.
pub fn visualize_geo(img: Tensor<3>, channel: ViewChannel) -> Tensor<3> {
    let alpha = img.clone().slice(s![.., .., 3..4]);
    let geo = img.slice(s![.., .., 4..8]);
    let normal = rendered_normal(geo.clone());
    let depth = rendered_depth(geo, alpha.clone());
    let (normal_mode, range) = match channel {
        ViewChannel::Normal => (true, 1.0),
        ViewChannel::Depth { max_meters } => (false, max_meters),
        ViewChannel::Rgb => (false, ViewChannel::DEFAULT_DEPTH_METERS),
    };
    geo_colormap_pack(depth, normal, alpha, 0.0, range.max(1e-6), normal_mode)
}

/// Render the splats' geometry and colormap it for the given viewer channel.
pub async fn render_geo_channel(
    splats: Splats,
    camera: &Camera,
    img_size: UVec2,
    background: Vec3,
    splat_scale: Option<f32>,
    channel: ViewChannel,
) -> Tensor<3> {
    let (geo, _aux) = render_splats(
        splats,
        camera,
        img_size,
        background,
        splat_scale,
        TextureMode::Float,
        true,
    )
    .await;
    visualize_geo(geo, channel)
}
