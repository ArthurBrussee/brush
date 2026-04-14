#import sorting

@group(0) @binding(0) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> reduced: array<u32>;

// Per-subgroup partial sums. MAX_SUBGROUPS is the worst-case number of
// subgroups in a workgroup of size WG (32 when SG=8 and WG=256).
var<workgroup> partials: array<u32, sorting::MAX_SUBGROUPS>;

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
    let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
    let num_reduce_wgs = sorting::BIN_COUNT * sorting::div_ceil(num_wgs, sorting::BLOCK_SIZE);

    let group_id = sorting::get_workgroup_id(wid, num_workgroups);

    if group_id >= num_reduce_wgs {
        return;
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / sorting::BIN_COUNT;
    let bin_id = group_id / num_reduce_wg_per_bin;

    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * sorting::BLOCK_SIZE;

    // Each thread reads ELEMENTS_PER_THREAD entries and serially sums them.
    // Out-of-range entries (when this is the bin's last partial chunk) are
    // skipped via the `< num_wgs` gate.
    var sum = 0u;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * sorting::WG + local_id.x;
        if data_index < num_wgs {
            sum += counts[bin_offset + data_index];
        }
    }

    // Two-level reduction: each subgroup reduces internally with subgroupAdd,
    // then the per-subgroup totals are combined. Fast path: subgroup 0 does a
    // second subgroupAdd over all partials (works when num_subgroups <=
    // subgroup_size, which is the common case on every modern GPU). Slow
    // path: thread 0 sums serially. Both branches are gated on a uniform
    // condition so the subgroup op stays in uniform control flow.
    let subgroup_sum = subgroupAdd(sum);

    let subgroup_id = local_id.x / subgroup_size;
    let num_subgroups = sorting::WG / subgroup_size;

    if subgroup_invocation_id == 0u {
        partials[subgroup_id] = subgroup_sum;
    }
    workgroupBarrier();

    if num_subgroups <= subgroup_size {
        if subgroup_id == 0u {
            let v = select(0u, partials[subgroup_invocation_id], subgroup_invocation_id < num_subgroups);
            let total = subgroupAdd(v);
            if subgroup_invocation_id == 0u {
                reduced[group_id] = total;
            }
        }
    } else {
        if local_id.x == 0u {
            var total = 0u;
            for (var i = 0u; i < num_subgroups; i++) {
                total += partials[i];
            }
            reduced[group_id] = total;
        }
    }
}
