use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::frontend::IndexMutExpand;
use burn_cubecl::cubecl::prelude::*;

#[doc(hidden)]
pub const CHECKS_PER_ITER: u32 = 8;

#[cube(launch)]
pub fn get_tile_offsets(
    num_inter: u32,
    num_tiles: u32,
    tile_id_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
) {
    // Compute linear position from 2D dispatch (for large dispatches that exceed 65535 workgroups)
    let workgroup_id = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let absolute_pos = workgroup_id * CUBE_DIM_X + UNIT_POS;
    let base_id = absolute_pos * CHECKS_PER_ITER;

    // `tile_id_from_isect` can contain the sentinel `num_tiles` produced by
    // `map_gaussians_to_intersect` whenever its `count_contributing_tiles`
    // disagrees with PF (separate optimisation passes). `tile_offsets` is
    // sized for valid tiles only, so we must gate every write on `tid <
    // num_tiles` to avoid stomping the slot one past the end.
    #[unroll]
    for i in 0..CHECKS_PER_ITER {
        let isect_id = base_id + i;

        if isect_id < num_inter {
            let tid = tile_id_from_isect[isect_id as usize];

            if tid < num_tiles {
                let tid_us = tid as usize;
                if isect_id == num_inter - 1 {
                    // Write the end of the last tile.
                    tile_offsets[tid_us * 2 + 1] = isect_id + 1;
                }

                if isect_id == 0 {
                    // First intersection: always write the start of its tile.
                    tile_offsets[tid_us * 2] = 0;
                } else {
                    let prev_tid = tile_id_from_isect[isect_id as usize - 1];
                    if tid != prev_tid {
                        if prev_tid < num_tiles {
                            // Write the end of the previous tile.
                            tile_offsets[prev_tid as usize * 2 + 1] = isect_id;
                        }
                        // Write start of this tile.
                        tile_offsets[tid_us * 2] = isect_id;
                    }
                }
            }
        }
    }
}
