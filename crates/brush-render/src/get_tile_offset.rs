use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::prelude::{ABSOLUTE_POS, Tensor};

pub(crate) const CHECKS_PER_ITER: u32 = 8;

#[cube]
fn check_tile_boundary(
    tile_id_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    isect_id: u32,
    inter: u32,
) {
    if isect_id < inter {
        let prev_tid = tile_id_from_isect[isect_id - 1];
        let tid = tile_id_from_isect[isect_id];

        if isect_id == inter - 1 {
            // Write the end of the last tile.
            tile_offsets[tid * 2 + 1] = isect_id + 1;
        }
        if tid != prev_tid {
            // Write the end of the previous tile.
            tile_offsets[prev_tid * 2 + 1] = isect_id;
            // Write start of this tile.
            tile_offsets[tid * 2] = isect_id;
        }
    }
}

#[cube(launch_unchecked)]
pub fn get_tile_offsets(
    tile_id_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    num_inter: &Tensor<u32>,
) {
    let inter = num_inter[0];
    let base_id = ABSOLUTE_POS * CHECKS_PER_ITER;

    #[unroll]
    for i in 0..CHECKS_PER_ITER {
        check_tile_boundary(tile_id_from_isect, tile_offsets, base_id + i, inter);
    }
}
