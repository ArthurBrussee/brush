#import sorting

struct Uniforms {
    shift: u32,
}

@group(0) @binding(0) var<storage, read> config: Uniforms;
@group(0) @binding(1) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<u32>;
@group(0) @binding(4) var<storage, read> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_values: array<u32>;

// Dual lds buffers for the in-workgroup shuffle of (key, value) pairs. Using
// two separate arrays lets us write both fields in one barrier window and
// read them in another, instead of write-key/barrier/read-key/barrier/
// write-value/barrier/read-value/barrier (4 → 2 barriers per inner step).
var<workgroup> lds_keys: array<u32, sorting::WG>;
var<workgroup> lds_values: array<u32, sorting::WG>;
var<workgroup> lds_scratch: array<u32, sorting::WG>;
var<workgroup> bin_offset_cache: array<u32, sorting::WG>;
var<workgroup> local_histogram: array<atomic<u32>, sorting::BIN_COUNT>;
// Per-subgroup partial sums for the workgroup-wide inclusive scan over
// `packed_histogram`. Used by the inner 2-bit pass.
var<workgroup> partials: array<u32, sorting::MAX_SUBGROUPS>;
// Workgroup-wide total of the packed_histogram scan, broadcast to every
// thread via a barrier after the cross-subgroup combine.
var<workgroup> chunk_total: u32;

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

    let group_id = sorting::get_workgroup_id(wid, num_workgroups);

    if group_id >= num_wgs {
        return;
    }

    let subgroup_id = local_id.x / subgroup_size;
    let num_subgroups = sorting::WG / subgroup_size;

    if local_id.x < sorting::BIN_COUNT {
        bin_offset_cache[local_id.x] = counts[local_id.x * num_wgs + group_id];
    }
    workgroupBarrier();
    let wg_block_start = sorting::BLOCK_SIZE * group_id;
    let block_index = wg_block_start + local_id.x;
    var data_index = block_index;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        if local_id.x < sorting::BIN_COUNT {
            atomicStore(&local_histogram[local_id.x], 0u);
        }
        var local_key = ~0u;
        var local_value = 0u;

        if data_index < num_keys {
            local_key = src[data_index];
            local_value = values[data_index];
        }

        for (var bit_shift = 0u; bit_shift < sorting::BITS_PER_PASS; bit_shift += 2u) {
            let key_index = (local_key >> config.shift) & 0xfu;
            let bit_key = (key_index >> bit_shift) & 3u;
            let packed_input = 1u << (bit_key * 8u);

            // Workgroup-wide exclusive scan of `packed_input`. The packed u32
            // carries 4 sub-counters in its bytes; integer add is associative
            // so the same subgroup-then-combine pattern works on the packed
            // form. Hybrid combine: subgroupExclusiveAdd if num_subgroups
            // fits in a single subgroup, serial fallback otherwise.
            let sg_inclusive = subgroupInclusiveAdd(packed_input);
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
                    if subgroup_invocation_id == num_subgroups - 1u {
                        chunk_total = scanned + v;
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
                    chunk_total = acc;
                }
            }
            workgroupBarrier();

            let total = chunk_total;
            let bin_offsets = (total << 8u) + (total << 16u) + (total << 24u);
            let exclusive_at_thread = partials[subgroup_id] + sg_inclusive - packed_input;
            let local_sum = bin_offsets + exclusive_at_thread;
            let key_offset = (local_sum >> (bit_key * 8u)) & 0xffu;

            // In-workgroup permutation. Dual lds buffers let us write both
            // key and value before a single barrier and read both back after
            // it — one barrier per inner step instead of four. The previous
            // inner step's reads are already flushed by the scan barriers
            // above, so no separate "post-read" barrier is needed here either.
            lds_keys[key_offset] = local_key;
            lds_values[key_offset] = local_value;
            workgroupBarrier();
            local_key = lds_keys[local_id.x];
            local_value = lds_values[local_id.x];
        }
        // The reads above are flushed before the next outer iteration writes
        // anything to lds_keys/lds_values by the workgroupBarrier on line
        // following the histogram scan (`workgroupBarrier()` after the
        // bin_offset_cache update).
        let key_index = (local_key >> config.shift) & 0xfu;
        atomicAdd(&local_histogram[key_index], 1u);
        workgroupBarrier();

        // Inclusive prefix sum of the BIN_COUNT-sized histogram.
        //
        // Fast path (subgroup_size >= BIN_COUNT, i.e. SG ≥ 16): subgroup 0
        // has at least 16 lanes, so a single subgroupInclusiveAdd gated by
        // `lane < BIN_COUNT` does the whole scan. This replaces a 4-iteration
        // Hillis-Steele scan that previously did 8 workgroupBarriers per
        // outer iteration.
        //
        // Slow path (SG=8): fall back to a serial scan in thread 0. Same
        // hybrid pattern as the cross-subgroup combine elsewhere.
        if subgroup_size >= sorting::BIN_COUNT {
            if subgroup_id == 0u {
                let v = select(
                    0u,
                    atomicLoad(&local_histogram[subgroup_invocation_id]),
                    subgroup_invocation_id < sorting::BIN_COUNT,
                );
                let inclusive = subgroupInclusiveAdd(v);
                if subgroup_invocation_id < sorting::BIN_COUNT {
                    lds_scratch[subgroup_invocation_id] = inclusive;
                }
            }
        } else {
            if local_id.x == 0u {
                var acc = 0u;
                for (var b = 0u; b < sorting::BIN_COUNT; b++) {
                    acc += atomicLoad(&local_histogram[b]);
                    lds_scratch[b] = acc;
                }
            }
        }
        workgroupBarrier();
        let global_offset = bin_offset_cache[key_index];
        workgroupBarrier();
        var local_offset = local_id.x;
        if key_index > 0u {
            local_offset -= lds_scratch[key_index - 1u];
        }
        let total_offset = global_offset + local_offset;
        if total_offset < num_keys {
            out[total_offset] = local_key;
            out_values[total_offset] = local_value;
        }
        if local_id.x < sorting::BIN_COUNT {
            bin_offset_cache[local_id.x] += atomicLoad(&local_histogram[local_id.x]);
        }
        workgroupBarrier();
        data_index += sorting::WG;
    }
}
