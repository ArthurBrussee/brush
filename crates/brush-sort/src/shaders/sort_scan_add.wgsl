#import sorting

@group(0) @binding(0) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read> reduced: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;

// Per-subgroup partial sums for the workgroup-wide scan step.
var<workgroup> partials: array<u32, sorting::MAX_SUBGROUPS>;
var<workgroup> lds: array<array<u32, sorting::WG>, sorting::ELEMENTS_PER_THREAD>;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    let num_keys = num_keys_arr[0];
    // let num_keys = num_keys_arr[0];
    let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
    let num_reduce_wgs = sorting::BIN_COUNT * sorting::div_ceil(num_wgs, sorting::BLOCK_SIZE);

    let group_id = sorting::get_workgroup_id(wid, num_workgroups);

    if group_id >= num_reduce_wgs {
        return;
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / sorting::BIN_COUNT;

    let bin_id = group_id / num_reduce_wg_per_bin;
    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * sorting::ELEMENTS_PER_THREAD * sorting::WG;

    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * sorting::WG + local_id.x;
        let col = (i * sorting::WG + local_id.x) / sorting::ELEMENTS_PER_THREAD;
        let row = (i * sorting::WG + local_id.x) % sorting::ELEMENTS_PER_THREAD;
        // This is not gated, we let robustness do it for us
        lds[row][col] = counts[bin_offset + data_index];
    }
    workgroupBarrier();
    // Per-thread serial exclusive scan. After this, lds[i][local_id.x] holds
    // exclusive prefix sums and `thread_sum` holds the inclusive total of the
    // thread's column.
    var thread_sum = 0u;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let tmp = lds[i][local_id.x];
        lds[i][local_id.x] = thread_sum;
        thread_sum += tmp;
    }

    // Workgroup-wide exclusive scan with hybrid combine: subgroupExclusiveAdd
    // in subgroup 0 if num_subgroups fits in a single subgroup (true for SG ≥
    // 16), serial fallback in thread 0 otherwise.
    let subgroup_id = local_id.x / subgroup_size;
    let num_subgroups = sorting::WG / subgroup_size;

    let sg_inclusive = subgroupInclusiveAdd(thread_sum);
    if subgroup_invocation_id == subgroup_size - 1u {
        partials[subgroup_id] = sg_inclusive;
    }
    workgroupBarrier();
    if num_subgroups <= subgroup_size {
        if subgroup_id == 0u {
            let v = select(0u, partials[subgroup_invocation_id], subgroup_invocation_id < num_subgroups);
            let scanned = subgroupExclusiveAdd(v);
            if subgroup_invocation_id < num_subgroups {
                partials[subgroup_invocation_id] = scanned;
            }
        }
    } else {
        if local_id.x == 0u {
            var acc = 0u;
            for (var i = 0u; i < num_subgroups; i++) {
                let v = partials[i];
                partials[i] = acc;
                acc += v;
            }
        }
    }
    workgroupBarrier();

    // Add the global base for this chunk (`reduced[group_id]`) plus this
    // thread's exclusive prefix within the workgroup, then push it down to
    // every entry in this thread's column.
    let workgroup_exclusive = partials[subgroup_id] + sg_inclusive - thread_sum;
    let total_base = reduced[group_id] + workgroup_exclusive;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        lds[i][local_id.x] += total_base;
    }
    // lds now contains exclusive prefix sum
    // Note: storing inclusive might be slightly cheaper here
    workgroupBarrier();
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * sorting::WG + local_id.x;
        let col = (i * sorting::WG + local_id.x) / sorting::ELEMENTS_PER_THREAD;
        let row = (i * sorting::WG + local_id.x) % sorting::ELEMENTS_PER_THREAD;
        if data_index < num_wgs {
            counts[bin_offset + data_index] = lds[row][col];
        }
    }
}
