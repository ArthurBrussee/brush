#![recursion_limit = "256"]

pub mod config;
pub mod eval;
pub mod lod;
pub mod msg;
pub mod train;

mod adam_scaled;
mod multinomial;
mod quat_vec;
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
    let mip = splats.render_mip;
    let (transforms_id, transforms, _) = splats.transforms.consume();
    let (sh_coeffs_dc_id, sh_coeffs_dc, _) = splats.sh_coeffs_dc.consume();
    let (sh_coeffs_rest_id, sh_coeffs_rest, _) = splats.sh_coeffs_rest.consume();
    let (raw_opacity_id, raw_opacity, _) = splats.raw_opacities.consume();

    Splats::<BDiff> {
        transforms: Param::initialized(
            transforms_id,
            Tensor::from_inner(transforms).require_grad(),
        ),
        sh_coeffs_dc: Param::initialized(
            sh_coeffs_dc_id,
            Tensor::from_inner(sh_coeffs_dc).require_grad(),
        ),
        sh_coeffs_rest: Param::initialized(
            sh_coeffs_rest_id,
            Tensor::from_inner(sh_coeffs_rest).require_grad(),
        ),
        raw_opacities: Param::initialized(
            raw_opacity_id,
            Tensor::from_inner(raw_opacity).require_grad(),
        ),
        render_mip: mip,
    }
}
