#import sorting

@group(0) @binding(0) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> reduced: array<u32>;

// Per-subgroup partial sums for the workgroup-wide scan step.
var<workgroup> partials: array<u32, sorting::MAX_SUBGROUPS>;
var<workgroup> lds: array<array<u32, sorting::WG>, sorting::ELEMENTS_PER_THREAD>;
// Workgroup-wide running carry between scan chunks. Written by exactly one
// thread inside the cross-subgroup combine and read by every thread on the
// next iteration via `carry += chunk_total`. MUST be workgroup storage —
// before the rebase fix this was a per-thread local, which left every other
// thread reading 0 and silently broke the multi-chunk scan.
var<workgroup> chunk_total: u32;

// Exclusive prefix sum over `reduced[0..num_reduce_wgs]`, in-place.
//
// Runs as a single workgroup. Historically this kernel only covered
// BLOCK_SIZE entries, which silently broke sorts above ~67M keys. It now
// walks `reduced` in BLOCK_SIZE-sized chunks with a running carry between
// chunks, so it scans arbitrary-length input.
//
// The per-chunk workgroup-wide scan uses subgroupInclusiveAdd for the in-
// subgroup pass, then a hybrid cross-subgroup combine: a parallel
// subgroupExclusiveAdd in subgroup 0 when num_subgroups fits in one subgroup
// (the common case on SG ≥ 16), or a serial scan in thread 0 otherwise
// (only kicks in on SG=8 hardware where WG=256 produces 32 subgroups).
@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    let num_keys = num_keys_arr[0];
    // let num_keys = num_keys_arr[0];
    let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
    let num_reduce_wgs = sorting::BIN_COUNT * sorting::div_ceil(num_wgs, sorting::BLOCK_SIZE);

    let subgroup_id = local_id.x / subgroup_size;
    let num_subgroups = sorting::WG / subgroup_size;

    var carry = 0u;
    var chunk_start = 0u;
    loop {
        if chunk_start >= num_reduce_wgs { break; }

        // Load. Adjacent threads pick up adjacent `reduced[]` indices for
        // coalesced global access (thread N at step i reads
        // `reduced[chunk_start + i*WG + N]`). The col/row swizzle into `lds`
        // packs each run of E_P_T consecutive `reduced[]` indices into one
        // thread's column, so the per-thread serial scan below operates on
        // contiguous values.
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            let data_index = chunk_start + i * sorting::WG + local_id.x;
            let col = (i * sorting::WG + local_id.x) / sorting::ELEMENTS_PER_THREAD;
            let row = (i * sorting::WG + local_id.x) % sorting::ELEMENTS_PER_THREAD;
            var v = 0u;
            if data_index < num_reduce_wgs {
                v = reduced[data_index];
            }
            lds[row][col] = v;
        }
        workgroupBarrier();

        // Per-thread serial exclusive scan. After this, lds[*][local_id.x]
        // holds the exclusive prefix sums and `thread_sum` holds the column
        // total.
        var thread_sum = 0u;
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            let tmp = lds[i][local_id.x];
            lds[i][local_id.x] = thread_sum;
            thread_sum += tmp;
        }

        // Workgroup-wide exclusive scan: subgroup inclusive scan, then a
        // cross-subgroup combine that picks fast or slow path based on
        // subgroup_size at runtime (uniform branch, see sorting.wgsl).
        let sg_inclusive = subgroupInclusiveAdd(thread_sum);
        if subgroup_invocation_id == subgroup_size - 1u {
            partials[subgroup_id] = sg_inclusive;
        }
        workgroupBarrier();
        if num_subgroups <= subgroup_size {
            // Every subgroup runs the scan so the call stays in uniform
            // control flow — Chrome/Tint can't prove `subgroup_id == 0u` is
            // plane-uniform and rejects subgroup ops gated on it. `partials`
            // is workgroup storage, so every subgroup reads the same inputs
            // and produces the same scan result; only subgroup 0 writes back.
            let v = select(0u, partials[subgroup_invocation_id], subgroup_invocation_id < num_subgroups);
            let scanned = subgroupExclusiveAdd(v);
            if subgroup_id == 0u {
                if subgroup_invocation_id < num_subgroups {
                    partials[subgroup_invocation_id] = scanned;
                }
                // Capture the inclusive total of the entire combine — last
                // active lane has it.
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

        // Each thread's exclusive prefix across the workgroup = exclusive
        // base of its subgroup + (its inclusive scan - its own contribution).
        let workgroup_exclusive = partials[subgroup_id] + sg_inclusive - thread_sum;
        let base = carry + workgroup_exclusive;
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            lds[i][local_id.x] += base;
        }
        workgroupBarrier();

        // Write back, mirroring the swizzled load so the global write is
        // also coalesced.
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            let data_index = chunk_start + i * sorting::WG + local_id.x;
            let col = (i * sorting::WG + local_id.x) / sorting::ELEMENTS_PER_THREAD;
            let row = (i * sorting::WG + local_id.x) % sorting::ELEMENTS_PER_THREAD;
            if data_index < num_reduce_wgs {
                reduced[data_index] = lds[row][col];
            }
        }
        workgroupBarrier();

        carry += chunk_total;
        chunk_start += sorting::BLOCK_SIZE;
    }
}
