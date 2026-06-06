//! `CubeCL` ports of the render kernels.

#![allow(
    clippy::doc_markdown,
    clippy::manual_div_ceil,
    clippy::manual_range_contains,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::excessive_precision,
    clippy::should_implement_trait,
    clippy::similar_names
)]

pub mod camera_model;
pub mod helpers;
pub mod integrate_alpha;
pub mod map_gaussians;
pub mod project_forward;
pub mod project_visible;
pub mod rasterize;
pub mod sh;
pub mod types;
