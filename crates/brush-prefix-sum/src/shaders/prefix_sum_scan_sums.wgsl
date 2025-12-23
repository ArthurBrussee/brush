#import prefix_sum_helpers as helpers

@compute
@workgroup_size(helpers::THREADS_PER_GROUP, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let id = helpers::get_global_id(gid, wid, lid);
    let idx = id * helpers::THREADS_PER_GROUP - 1u;

    var x = 0u;
    if (idx >= 0u && idx < arrayLength(&helpers::input)) {
        x = helpers::input[idx];
    }

    helpers::groupScan(id, lid, x);
}
