#![recursion_limit = "256"]

pub mod config;
pub mod eval;
pub mod msg;
pub mod train;

mod adam_scaled;
mod multinomial;
mod quat_vec;
mod ssim;
mod stats;

mod splat_init;

pub use splat_init::{RandomSplatsConfig, create_random_splats, to_init_splats};

use brush_render::gaussian_splats::Splats;
use burn::{
    module::Param,
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};

// TODO: This should probably exist in Burn. Maybe make a PR.
pub fn splats_into_autodiff<B: Backend, BDiff: AutodiffBackend<InnerBackend = B>>(
    splats: Splats<B>,
) -> Splats<BDiff> {
    let mode = splats.render_mode;
    let (means_id, means, _) = splats.means.consume();
    let (rotation_id, rotation, _) = splats.rotations.consume();
    let (log_scales_id, log_scales, _) = splats.log_scales.consume();
    let (sh_coeffs_id, sh_coeffs, _) = splats.sh_coeffs.consume();
    let (raw_opacity_id, raw_opacity, _) = splats.raw_opacities.consume();

    Splats::<BDiff> {
        means: Param::initialized(means_id, Tensor::from_inner(means).require_grad()),
        rotations: Param::initialized(rotation_id, Tensor::from_inner(rotation).require_grad()),
        log_scales: Param::initialized(
            log_scales_id,
            Tensor::from_inner(log_scales).require_grad(),
        ),
        sh_coeffs: Param::initialized(sh_coeffs_id, Tensor::from_inner(sh_coeffs).require_grad()),
        raw_opacities: Param::initialized(
            raw_opacity_id,
            Tensor::from_inner(raw_opacity).require_grad(),
        ),
        render_mode: mode,
    }
}
