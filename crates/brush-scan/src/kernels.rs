use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::frontend::CompilationArg;
use burn::cubecl::frontend::IndexMutExpand;
use burn::cubecl::prelude::*;

/// Threads per cube for the scan kernels. WebGPU guarantees at least 256.
pub const WG: u32 = 256;
/// Elements each thread scans locally before the cube-wide combine.
pub const ELEMENTS_PER_THREAD: u32 = 4;
/// Elements scanned by one cube.
pub const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
pub const BLOCK_SIZE_USIZE: usize = BLOCK_SIZE as usize;

/// Upper bound on the number of planes (subgroups) inside a cube. Plane size
/// varies by hardware: 8/16 on some Intel, 32 on Apple/most Intel/NVIDIA, 64
/// on AMD wave64. With 256 threads the worst case is 32 planes of 8, so the
/// per-plane partials below are padded to 32. Cubes wider than 256 threads
/// need this raised.
pub const MAX_PLANES: usize = 32;

/// Exclusive prefix sum of `value` over the whole cube. Returns the exclusive
/// prefix for this thread and the cube total (uniform across the cube).
///
/// Plane-scan first, then one plane scans the per-plane totals, with a serial
/// fallback when there are more planes than one plane has lanes. Uses shared
/// memory internally, so a caller running this in a loop must `sync_cube()`
/// between calls to keep one call's reads ahead of the next call's writes.
#[cube]
pub fn cube_exclusive_sum(value: u32) -> (u32, u32) {
    let mut partials = Shared::new_slice(MAX_PLANES);
    let mut total_shared = Shared::new_slice(1usize);

    let plane_id = UNIT_POS / PLANE_DIM;
    let num_planes = CUBE_DIM / PLANE_DIM;

    let plane_inclusive = plane_inclusive_sum(value);
    if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
        partials[plane_id as usize] = plane_inclusive;
    }
    sync_cube();

    if num_planes <= PLANE_DIM {
        let v = select(
            UNIT_POS_PLANE < num_planes,
            partials[UNIT_POS_PLANE as usize],
            0u32,
        );
        let scanned = plane_exclusive_sum(v);
        if plane_id == 0u32 {
            if UNIT_POS_PLANE < num_planes {
                partials[UNIT_POS_PLANE as usize] = scanned;
            }
            if UNIT_POS_PLANE == num_planes - 1u32 {
                total_shared[0_usize] = scanned + v;
            }
        }
    } else if UNIT_POS == 0u32 {
        let mut acc = 0u32;
        for i in 0u32..num_planes {
            let v = partials[i as usize];
            partials[i as usize] = acc;
            acc += v;
        }
        total_shared[0_usize] = acc;
    }
    sync_cube();

    let exclusive = partials[plane_id as usize] + plane_inclusive - value;
    (exclusive, total_shared[0_usize])
}

/// Sum of `value` over the whole cube, returned to every thread.
#[cube]
pub fn cube_sum(value: u32) -> u32 {
    let (_, total) = cube_exclusive_sum(value);
    total
}

/// Where the `lin`-th element of a block lives in a block-scan buffer.
/// Consecutive threads load consecutive elements (coalesced), stored
/// transposed so each thread then owns [`ELEMENTS_PER_THREAD`] consecutive
/// values at a conflict-free stride.
#[cube]
pub fn lds_index(lin: u32) -> u32 {
    (lin % ELEMENTS_PER_THREAD) * WG + lin / ELEMENTS_PER_THREAD
}

/// Scan the [`BLOCK_SIZE`] values this cube holds in `lds` (laid out by
/// [`lds_index()`], so thread `UNIT_POS` owns `lds[j * WG + UNIT_POS]`), add
/// `base` to every result, and return the cube's total. `inclusive` picks
/// whether an element counts itself.
///
/// The caller syncs: once after filling `lds`, and again before reading the
/// scanned values back out.
#[cube]
pub fn block_scan(lds: &mut Shared<[u32]>, base: u32, #[comptime] inclusive: bool) -> u32 {
    let mut thread_sum = 0u32;
    for j in 0u32..ELEMENTS_PER_THREAD {
        let idx = (j * WG + UNIT_POS) as usize;
        let v = lds[idx];
        if inclusive {
            thread_sum += v;
            lds[idx] = thread_sum;
        } else {
            lds[idx] = thread_sum;
            thread_sum += v;
        }
    }

    let (exclusive, total) = cube_exclusive_sum(thread_sum);

    let offset = base + exclusive;
    for j in 0u32..ELEMENTS_PER_THREAD {
        lds[(j * WG + UNIT_POS) as usize] += offset;
    }
    total
}

/// Inclusive scan within each block of [`BLOCK_SIZE`] elements. Each cube
/// writes its block's total to `block_sums[CUBE_POS]` for the next level.
#[cube(launch)]
pub fn scan_blocks_kernel(
    input: &Tensor<u32>,
    output: &mut Tensor<u32>,
    block_sums: &mut Tensor<u32>,
) {
    let n = input.len() as u32;
    let block = CUBE_POS as u32;
    let base = block * BLOCK_SIZE;

    let mut lds = Shared::new_slice(BLOCK_SIZE_USIZE);
    for i in 0u32..ELEMENTS_PER_THREAD {
        let lin = i * WG + UNIT_POS;
        let idx = base + lin;
        let mut v = 0u32;
        if idx < n {
            v = input[idx as usize];
        }
        lds[lds_index(lin) as usize] = v;
    }
    sync_cube();

    let total = block_scan(&mut lds, 0u32, true);
    sync_cube();

    for i in 0u32..ELEMENTS_PER_THREAD {
        let lin = i * WG + UNIT_POS;
        let idx = base + lin;
        if idx < n {
            output[idx as usize] = lds[lds_index(lin) as usize];
        }
    }
    if UNIT_POS == 0u32 {
        block_sums[block as usize] = total;
    }
}

/// Add each block's exclusive offset, taken from the inclusive scan of the
/// block sums, to every element of `output`.
#[cube(launch)]
pub fn add_block_offsets_kernel(scanned_sums: &Tensor<u32>, output: &mut Tensor<u32>) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= output.len() as u32 {
        terminate!();
    }
    let block = idx / BLOCK_SIZE;
    if block > 0u32 {
        output[idx as usize] += scanned_sums[(block - 1u32) as usize];
    }
}
