use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::frontend::CubeIndexMutExpand;
use burn_cubecl::cubecl::prelude::*;

pub const WG: u32 = 256;
pub const WG_USIZE: usize = WG as usize;
pub const BITS_PER_PASS: u32 = 4;
pub const BIN_COUNT: u32 = 1 << BITS_PER_PASS;
pub const BIN_COUNT_USIZE: usize = BIN_COUNT as usize;
pub const ELEMENTS_PER_THREAD: u32 = 4;
pub const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;

// Upper bound on the number of subgroups inside a workgroup of size WG.
// Subgroup size varies by hardware: 8/16 on some Intel, 32 on Apple/most Intel/
// NVIDIA, 64 on AMD wave64. With WG=256 the worst case is SG=8, which gives
// 32 subgroups. We pad `partials` arrays to 32 so they are correctly sized
// for any subgroup size in [8, 64].
pub const MAX_SUBGROUPS: u32 = 32;

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1u32) / b
}

#[cube(launch_unchecked)]
pub fn sort_count_kernel(
    num_keys_arr: &Tensor<u32>,
    src: &Tensor<u32>,
    counts: &mut Tensor<u32>,
    shift: u32,
) {
    let num_keys = num_keys_arr[0];

    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let group_id = CUBE_POS as u32;

    if group_id >= num_wgs {
        terminate!();
    }

    let histogram = SharedMemory::<Atomic<u32>>::new(BIN_COUNT_USIZE);
    if UNIT_POS < BIN_COUNT {
        Atomic::store(&histogram[UNIT_POS as usize], 0u32);
    }
    sync_cube();

    let wg_block_start = BLOCK_SIZE * group_id;
    let mut data_index = wg_block_start + UNIT_POS;

    for _ in 0u32..ELEMENTS_PER_THREAD {
        if data_index < num_keys {
            let local_key = (src[data_index as usize] >> shift) & 0xfu32;
            Atomic::fetch_add(&histogram[local_key as usize], 1u32);
        }
        data_index += WG;
    }
    sync_cube();
    if UNIT_POS < BIN_COUNT {
        let nw = div_ceil(num_keys, BLOCK_SIZE);
        counts[(UNIT_POS * nw + group_id) as usize] = Atomic::load(&histogram[UNIT_POS as usize]);
    }
}

#[cube(launch_unchecked)]
pub fn sort_reduce_kernel(
    num_keys_arr: &Tensor<u32>,
    counts: &Tensor<u32>,
    reduced: &mut Tensor<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let group_id = CUBE_POS as u32;
    if group_id >= num_reduce_wgs {
        terminate!();
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;
    let bin_id = group_id / num_reduce_wg_per_bin;

    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * BLOCK_SIZE;

    let mut sum = 0u32;
    for i in 0u32..ELEMENTS_PER_THREAD {
        let data_index = base_index + i * WG + UNIT_POS;
        if data_index < num_wgs {
            sum += counts[(bin_offset + data_index) as usize];
        }
    }

    let subgroup_sum = plane_sum(sum);

    let mut partials = SharedMemory::<u32>::new(MAX_SUBGROUPS as usize);
    let subgroup_id = UNIT_POS / PLANE_DIM;
    let num_subgroups = WG / PLANE_DIM;

    if UNIT_POS_PLANE == 0u32 {
        partials[subgroup_id as usize] = subgroup_sum;
    }
    sync_cube();

    if num_subgroups <= PLANE_DIM {
        let v = select(
            UNIT_POS_PLANE < num_subgroups,
            partials[UNIT_POS_PLANE as usize],
            0u32,
        );
        let total = plane_sum(v);
        if subgroup_id == 0u32 && UNIT_POS_PLANE == 0u32 {
            reduced[group_id as usize] = total;
        }
    } else if UNIT_POS == 0u32 {
        let mut total = 0u32;
        for i in 0u32..num_subgroups {
            total += partials[i as usize];
        }
        reduced[group_id as usize] = total;
    }
}

#[cube(launch_unchecked)]
pub fn sort_scan_kernel(num_keys_arr: &Tensor<u32>, reduced: &mut Tensor<u32>) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let subgroup_id = UNIT_POS / PLANE_DIM;
    let num_subgroups = WG / PLANE_DIM;

    let mut partials = SharedMemory::<u32>::new(MAX_SUBGROUPS as usize);
    let mut lds = SharedMemory::<u32>::new((WG * ELEMENTS_PER_THREAD) as usize);
    let mut chunk_total = SharedMemory::<u32>::new(1usize);

    let mut carry = 0u32;
    let mut chunk_start = 0u32;
    while chunk_start < num_reduce_wgs {
        for i in 0u32..ELEMENTS_PER_THREAD {
            let data_index = chunk_start + i * WG + UNIT_POS;
            let col = (i * WG + UNIT_POS) / ELEMENTS_PER_THREAD;
            let row = (i * WG + UNIT_POS) % ELEMENTS_PER_THREAD;
            let mut v = 0u32;
            if data_index < num_reduce_wgs {
                v = reduced[data_index as usize];
            }
            lds[(row * WG + col) as usize] = v;
        }
        sync_cube();

        let mut thread_sum = 0u32;
        for i in 0u32..ELEMENTS_PER_THREAD {
            let tmp = lds[(i * WG + UNIT_POS) as usize];
            lds[(i * WG + UNIT_POS) as usize] = thread_sum;
            thread_sum += tmp;
        }

        let sg_inclusive = plane_inclusive_sum(thread_sum);
        if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
            partials[subgroup_id as usize] = sg_inclusive;
        }
        sync_cube();
        if num_subgroups <= PLANE_DIM {
            let v = select(
                UNIT_POS_PLANE < num_subgroups,
                partials[UNIT_POS_PLANE as usize],
                0u32,
            );
            let scanned = plane_exclusive_sum(v);
            if subgroup_id == 0u32 {
                if UNIT_POS_PLANE < num_subgroups {
                    partials[UNIT_POS_PLANE as usize] = scanned;
                }
                if UNIT_POS_PLANE == num_subgroups - 1u32 {
                    chunk_total[0_usize] = scanned + v;
                }
            }
        } else if UNIT_POS == 0u32 {
            let mut acc = 0u32;
            for i in 0u32..num_subgroups {
                let v = partials[i as usize];
                partials[i as usize] = acc;
                acc += v;
            }
            chunk_total[0_usize] = acc;
        }
        sync_cube();

        let workgroup_exclusive = partials[subgroup_id as usize] + sg_inclusive - thread_sum;
        let base = carry + workgroup_exclusive;
        for i in 0u32..ELEMENTS_PER_THREAD {
            lds[(i * WG + UNIT_POS) as usize] += base;
        }
        sync_cube();

        for i in 0u32..ELEMENTS_PER_THREAD {
            let data_index = chunk_start + i * WG + UNIT_POS;
            let col = (i * WG + UNIT_POS) / ELEMENTS_PER_THREAD;
            let row = (i * WG + UNIT_POS) % ELEMENTS_PER_THREAD;
            if data_index < num_reduce_wgs {
                reduced[data_index as usize] = lds[(row * WG + col) as usize];
            }
        }
        sync_cube();

        carry += chunk_total[0_usize];
        chunk_start += BLOCK_SIZE;
    }
}

#[cube(launch_unchecked)]
pub fn sort_scan_add_kernel(
    num_keys_arr: &Tensor<u32>,
    reduced: &Tensor<u32>,
    counts: &mut Tensor<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let group_id = CUBE_POS as u32;
    if group_id >= num_reduce_wgs {
        terminate!();
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;
    let bin_id = group_id / num_reduce_wg_per_bin;
    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * ELEMENTS_PER_THREAD * WG;

    let mut partials = SharedMemory::<u32>::new(MAX_SUBGROUPS as usize);
    let mut lds = SharedMemory::<u32>::new((WG * ELEMENTS_PER_THREAD) as usize);

    for i in 0u32..ELEMENTS_PER_THREAD {
        let data_index = base_index + i * WG + UNIT_POS;
        let col = (i * WG + UNIT_POS) / ELEMENTS_PER_THREAD;
        let row = (i * WG + UNIT_POS) % ELEMENTS_PER_THREAD;
        // Gate explicitly: the prior version relied on WebGPU robustness
        // clamping OOB reads to zero, which isn't a guarantee under
        // `launch_unchecked` on non-WebGPU backends.
        let mut v = 0u32;
        if data_index < num_wgs {
            v = counts[(bin_offset + data_index) as usize];
        }
        lds[(row * WG + col) as usize] = v;
    }
    sync_cube();

    let mut thread_sum = 0u32;
    for i in 0u32..ELEMENTS_PER_THREAD {
        let tmp = lds[(i * WG + UNIT_POS) as usize];
        lds[(i * WG + UNIT_POS) as usize] = thread_sum;
        thread_sum += tmp;
    }

    let subgroup_id = UNIT_POS / PLANE_DIM;
    let num_subgroups = WG / PLANE_DIM;

    let sg_inclusive = plane_inclusive_sum(thread_sum);
    if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
        partials[subgroup_id as usize] = sg_inclusive;
    }
    sync_cube();
    if num_subgroups <= PLANE_DIM {
        let v = select(
            UNIT_POS_PLANE < num_subgroups,
            partials[UNIT_POS_PLANE as usize],
            0u32,
        );
        let scanned = plane_exclusive_sum(v);
        if subgroup_id == 0u32 && UNIT_POS_PLANE < num_subgroups {
            partials[UNIT_POS_PLANE as usize] = scanned;
        }
    } else if UNIT_POS == 0u32 {
        let mut acc = 0u32;
        for i in 0u32..num_subgroups {
            let v = partials[i as usize];
            partials[i as usize] = acc;
            acc += v;
        }
    }
    sync_cube();

    let workgroup_exclusive = partials[subgroup_id as usize] + sg_inclusive - thread_sum;
    let total_base = reduced[group_id as usize] + workgroup_exclusive;
    for i in 0u32..ELEMENTS_PER_THREAD {
        lds[(i * WG + UNIT_POS) as usize] += total_base;
    }
    sync_cube();
    for i in 0u32..ELEMENTS_PER_THREAD {
        let data_index = base_index + i * WG + UNIT_POS;
        let col = (i * WG + UNIT_POS) / ELEMENTS_PER_THREAD;
        let row = (i * WG + UNIT_POS) % ELEMENTS_PER_THREAD;
        if data_index < num_wgs {
            counts[(bin_offset + data_index) as usize] = lds[(row * WG + col) as usize];
        }
    }
}

#[cube(launch_unchecked)]
pub fn sort_scatter_kernel(
    num_keys_arr: &Tensor<u32>,
    src: &Tensor<u32>,
    values: &Tensor<u32>,
    counts: &Tensor<u32>,
    out: &mut Tensor<u32>,
    out_values: &mut Tensor<u32>,
    shift: u32,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);

    let group_id = CUBE_POS as u32;
    if group_id >= num_wgs {
        terminate!();
    }

    let subgroup_id = UNIT_POS / PLANE_DIM;
    let num_subgroups = WG / PLANE_DIM;

    let mut lds_keys = SharedMemory::<u32>::new(WG_USIZE);
    let mut lds_values = SharedMemory::<u32>::new(WG_USIZE);
    let mut lds_scratch = SharedMemory::<u32>::new(WG_USIZE);
    let mut bin_offset_cache = SharedMemory::<u32>::new(WG_USIZE);
    let local_histogram = SharedMemory::<Atomic<u32>>::new(BIN_COUNT_USIZE);
    let mut partials = SharedMemory::<u32>::new(MAX_SUBGROUPS as usize);
    let mut chunk_total = SharedMemory::<u32>::new(1usize);

    if UNIT_POS < BIN_COUNT {
        bin_offset_cache[UNIT_POS as usize] = counts[(UNIT_POS * num_wgs + group_id) as usize];
    }
    sync_cube();
    let wg_block_start = BLOCK_SIZE * group_id;
    let block_index = wg_block_start + UNIT_POS;
    let mut data_index = block_index;
    for _ in 0u32..ELEMENTS_PER_THREAD {
        if UNIT_POS < BIN_COUNT {
            Atomic::store(&local_histogram[UNIT_POS as usize], 0u32);
        }

        let mut local_key = 0xFFFFFFFFu32;
        let mut local_value = 0u32;

        if data_index < num_keys {
            local_key = src[data_index as usize];
            local_value = values[data_index as usize];
        }

        let mut bit_shift = 0u32;
        while bit_shift < BITS_PER_PASS {
            let key_index = (local_key >> shift) & 0xfu32;
            let bit_key = (key_index >> bit_shift) & 3u32;
            let packed_input = 1u32 << (bit_key * 8u32);

            let sg_inclusive = plane_inclusive_sum(packed_input);
            if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
                partials[subgroup_id as usize] = sg_inclusive;
            }
            sync_cube();
            if num_subgroups <= PLANE_DIM {
                let v = select(
                    UNIT_POS_PLANE < num_subgroups,
                    partials[UNIT_POS_PLANE as usize],
                    0u32,
                );
                let scanned = plane_exclusive_sum(v);
                if subgroup_id == 0u32 {
                    if UNIT_POS_PLANE < num_subgroups {
                        partials[UNIT_POS_PLANE as usize] = scanned;
                    }
                    if UNIT_POS_PLANE == num_subgroups - 1u32 {
                        chunk_total[0_usize] = scanned + v;
                    }
                }
            } else if UNIT_POS == 0u32 {
                let mut acc = 0u32;
                for i in 0u32..num_subgroups {
                    let v = partials[i as usize];
                    partials[i as usize] = acc;
                    acc += v;
                }
                chunk_total[0_usize] = acc;
            }
            sync_cube();

            let total = chunk_total[0_usize];
            let bin_offsets = (total << 8u32) + (total << 16u32) + (total << 24u32);
            let exclusive_at_thread = partials[subgroup_id as usize] + sg_inclusive - packed_input;
            let local_sum = bin_offsets + exclusive_at_thread;
            let key_offset = (local_sum >> (bit_key * 8u32)) & 0xffu32;

            lds_keys[key_offset as usize] = local_key;
            lds_values[key_offset as usize] = local_value;
            sync_cube();
            local_key = lds_keys[UNIT_POS as usize];
            local_value = lds_values[UNIT_POS as usize];

            bit_shift += 2u32;
        }

        let key_index = (local_key >> shift) & 0xfu32;
        Atomic::fetch_add(&local_histogram[key_index as usize], 1u32);
        sync_cube();

        if PLANE_DIM >= BIN_COUNT {
            let v = select(
                UNIT_POS_PLANE < BIN_COUNT,
                Atomic::load(&local_histogram[UNIT_POS_PLANE as usize]),
                0u32,
            );
            let inclusive = plane_inclusive_sum(v);
            if subgroup_id == 0u32 && UNIT_POS_PLANE < BIN_COUNT {
                lds_scratch[UNIT_POS_PLANE as usize] = inclusive;
            }
        } else if UNIT_POS == 0u32 {
            let mut acc = 0u32;
            for b in 0u32..BIN_COUNT {
                acc += Atomic::load(&local_histogram[b as usize]);
                lds_scratch[b as usize] = acc;
            }
        }
        sync_cube();
        let global_offset = bin_offset_cache[key_index as usize];
        sync_cube();
        let mut local_offset = UNIT_POS;
        if key_index > 0u32 {
            local_offset -= lds_scratch[(key_index - 1u32) as usize];
        }
        let total_offset = global_offset + local_offset;
        if total_offset < num_keys {
            out[total_offset as usize] = local_key;
            out_values[total_offset as usize] = local_value;
        }
        if UNIT_POS < BIN_COUNT {
            bin_offset_cache[UNIT_POS as usize] +=
                Atomic::load(&local_histogram[UNIT_POS as usize]);
        }
        sync_cube();
        data_index += WG;
    }
}
