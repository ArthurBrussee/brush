use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::frontend::CubeIndexMutExpand;
use burn_cubecl::cubecl::prelude::*;

pub const THREADS_PER_GROUP: u32 = 512;
const TPG_USIZE: usize = THREADS_PER_GROUP as usize;

#[cube]
fn linear_workgroup_id() -> u32 {
    CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X
}

#[cube]
fn linear_global_id() -> u32 {
    linear_workgroup_id() * THREADS_PER_GROUP + UNIT_POS
}

#[cube]
fn group_scan(id: u32, gi: u32, x: u32, output: &mut Tensor<u32>) {
    let mut bucket = SharedMemory::<u32>::new(TPG_USIZE);
    bucket[gi as usize] = x;

    let mut t = 1u32;
    while t < THREADS_PER_GROUP {
        sync_cube();
        let mut temp = bucket[gi as usize];
        if gi >= t {
            temp += bucket[(gi - t) as usize];
        }
        sync_cube();
        bucket[gi as usize] = temp;
        t *= 2u32;
    }
    if (id as usize) < output.len() {
        output[id as usize] = bucket[gi as usize];
    }
}

#[cube(launch_unchecked)]
pub fn prefix_sum_scan_kernel(input: &Tensor<u32>, output: &mut Tensor<u32>) {
    let id = linear_global_id();
    let gi = UNIT_POS;

    let mut x = 0u32;
    if (id as usize) < input.len() {
        x = input[id as usize];
    }

    group_scan(id, gi, x, output);
}

#[cube(launch_unchecked)]
pub fn prefix_sum_scan_sums_kernel(input: &Tensor<u32>, output: &mut Tensor<u32>) {
    let id = linear_global_id();
    let gi = UNIT_POS;
    // id * THREADS_PER_GROUP - 1, gated on id != 0 to avoid underflow.
    let mut x = 0u32;
    if id != 0u32 {
        let idx = id * THREADS_PER_GROUP - 1u32;
        if (idx as usize) < input.len() {
            x = input[idx as usize];
        }
    }

    group_scan(id, gi, x, output);
}

#[cube(launch_unchecked)]
pub fn prefix_sum_add_scanned_sums_kernel(input: &Tensor<u32>, output: &mut Tensor<u32>) {
    let id = linear_global_id();
    let workgroup_id = linear_workgroup_id();

    if (id as usize) < output.len() {
        output[id as usize] += input[workgroup_id as usize];
    }
}
