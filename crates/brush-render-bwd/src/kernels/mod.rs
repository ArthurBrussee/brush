//! `CubeCL` ports of the backward render kernels.

#![allow(
    clippy::doc_markdown,
    clippy::manual_div_ceil,
    clippy::doc_lazy_continuation,
    clippy::large_stack_frames,
    clippy::needless_pass_by_ref_mut,
    clippy::similar_names
)]

pub mod project_backwards;
pub mod rasterize_backwards;
