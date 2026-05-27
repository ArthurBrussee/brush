use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::frontend::IndexMutExpand;
use burn_cubecl::cubecl::prelude::*;

pub const THREADS_PER_GROUP: usize = 512;

#[cube]
fn linear_workgroup_id() -> usize {
    CUBE_POS
}

#[cube]
fn linear_global_id() -> usize {
    ABSOLUTE_POS
}

#[cube]
fn group_scan(id: usize, gi: usize, x: u32, output: &mut Tensor<u32>) {
    let mut bucket = Shared::new_slice(THREADS_PER_GROUP);
    bucket[gi] = x;

    let mut t = 1;
    while t < THREADS_PER_GROUP {
        sync_cube();
        let mut temp = bucket[gi];
        if gi >= t {
            temp += bucket[gi - t];
        }
        sync_cube();
        bucket[gi] = temp;
        t *= 2;
    }
    if id < output.len() {
        output[id] = bucket[gi];
    }
}

#[cube(launch)]
pub fn prefix_sum_scan_kernel(input: &Tensor<u32>, output: &mut Tensor<u32>) {
    let id = linear_global_id();

    let mut x = 0u32;
    if id < input.len() {
        x = input[id];
    }

    group_scan(id, UNIT_POS as usize, x, output);
}

#[cube(launch)]
pub fn prefix_sum_scan_sums_kernel(input: &Tensor<u32>, output: &mut Tensor<u32>) {
    let id = linear_global_id();
    // id * THREADS_PER_GROUP - 1, gated on id != 0 to avoid underflow.
    let mut x = 0u32;
    if id != 0 {
        let idx = id * THREADS_PER_GROUP - 1;
        if idx < input.len() {
            x = input[idx];
        }
    }
    group_scan(id, UNIT_POS as usize, x, output);
}

#[cube(launch)]
pub fn prefix_sum_add_scanned_sums_kernel(input: &Tensor<u32>, output: &mut Tensor<u32>) {
    let id = linear_global_id();
    let workgroup_id = linear_workgroup_id();

    if id < output.len() {
        output[id] += input[workgroup_id];
    }
}
