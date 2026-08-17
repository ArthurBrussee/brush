//! Backward (differentiable) render path.
//!
//! Lives in `brush-render` rather than a separate crate because the
//! `#[backend_extension]`-generated `Dispatch` impl calls the `Autodiff` arm,
//! so `impl SplatOps for Autodiff<..>` must be visible from the crate that
//! defines `SplatOps`.
pub mod burn_glue;
mod kernels;
mod render_bwd;

pub use burn_glue::{
    SplatOutputDiff, render_splats, render_splats_with_features, render_splats_with_pass,
};
