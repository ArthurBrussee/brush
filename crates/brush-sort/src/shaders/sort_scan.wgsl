#import sorting

@group(0) @binding(0) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> reduced: array<u32>;

var<workgroup> sums: array<u32, sorting::WG>;
var<workgroup> lds: array<array<u32, sorting::WG>, sorting::ELEMENTS_PER_THREAD>;
var<workgroup> chunk_total: u32;

// Exclusive prefix sum over `reduced[0..num_reduce_wgs]`, in-place.
//
// Runs as a single workgroup. Historically this kernel only covered
// BLOCK_SIZE entries, which silently broke sorts where num_reduce_wgs exceeded
// BLOCK_SIZE (i.e. sorts of more than ~67M keys). It now loops over chunks of
// BLOCK_SIZE and propagates a running `carry` between chunks, so it scans an
// arbitrary-length `reduced` buffer.
@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
    let num_reduce_wgs = sorting::BIN_COUNT * sorting::div_ceil(num_wgs, sorting::BLOCK_SIZE);

    var carry = 0u;
    var chunk_start = 0u;
    loop {
        if chunk_start >= num_reduce_wgs { break; }

        // Load chunk into lds, zero-padding anything past num_reduce_wgs so the
        // scan of the final short chunk produces the right total.
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

        // Per-thread serial exclusive scan over its ELEMENTS_PER_THREAD column.
        // After this, `sum` holds the inclusive total of the thread's column.
        var sum = 0u;
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            let tmp = lds[i][local_id.x];
            lds[i][local_id.x] = sum;
            sum += tmp;
        }

        // Hillis-Steele inclusive scan over the per-thread totals.
        sums[local_id.x] = sum;
        for (var i = 0u; i < 8u; i++) {
            workgroupBarrier();
            if local_id.x >= (1u << i) {
                sum += sums[local_id.x - (1u << i)];
            }
            workgroupBarrier();
            sums[local_id.x] = sum;
        }
        workgroupBarrier();

        // Capture the total of this chunk for the next iteration's carry.
        if local_id.x == sorting::WG - 1u {
            chunk_total = sums[sorting::WG - 1u];
        }

        // Per-thread exclusive base = carry + (inclusive prefix of prior threads).
        var base = carry;
        if local_id.x > 0u {
            base += sums[local_id.x - 1u];
        }
        for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
            lds[i][local_id.x] += base;
        }
        workgroupBarrier();

        // Write back.
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
