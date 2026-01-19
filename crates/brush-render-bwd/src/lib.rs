pub mod burn_glue;
mod render_bwd;

pub use burn_glue::{
    ProjectBwdState, RasterizeBwdState, RasterizeGrads, SplatBwdOps, SplatGrads, SplatOutputDiff,
    render_splats,
};
