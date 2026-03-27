use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::frontend::CubeIndexMutExpand;
use burn_cubecl::cubecl::prelude::*;

pub(crate) const CHECKS_PER_ITER: u32 = 8;

#[cube(launch_unchecked)]
pub fn get_tile_offsets(
    tile_id_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    num_inter: &Tensor<u32>,
) {
    let inter = num_inter[0] as usize;
    // Compute linear position from 2D dispatch (for large dispatches that exceed 65535 workgroups)
    let workgroup_id = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let absolute_pos = workgroup_id * CUBE_DIM_X + UNIT_POS;
    let base_id = absolute_pos * CHECKS_PER_ITER;

    #[unroll]
    for i in 0..CHECKS_PER_ITER {
        let isect_id = (base_id + i) as usize;

        if isect_id < inter {
            let tid = tile_id_from_isect[isect_id] as usize;

            if isect_id == inter - 1 {
                // Write the end of the last tile.
                tile_offsets[tid * 2 + 1] = isect_id as u32 + 1;
            }

            if isect_id == 0 {
                // First intersection: always write the start of its tile.
                tile_offsets[tid * 2] = 0;
            } else {
                let prev_tid = tile_id_from_isect[isect_id - 1] as usize;
                if tid != prev_tid {
                    // Write the end of the previous tile.
                    tile_offsets[prev_tid * 2 + 1] = isect_id as u32;
                    // Write start of this tile.
                    tile_offsets[tid * 2] = isect_id as u32;
                }
            }
        }
    }
}
