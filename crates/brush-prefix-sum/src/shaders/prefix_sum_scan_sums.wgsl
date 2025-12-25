#import prefix_sum_helpers as helpers

@compute
@workgroup_size(helpers::THREADS_PER_GROUP, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let id = helpers::get_global_id(wid, num_wgs, lid);
    let len = helpers::get_length();
    let idx = id * helpers::THREADS_PER_GROUP - 1u;

    // Number of groups needed for `len` elements
    let num_groups = (len + helpers::THREADS_PER_GROUP - 1u) / helpers::THREADS_PER_GROUP;

    var x = 0u;
    if (idx >= 0u && idx < len) {
        x = helpers::input[idx];
    }

    helpers::groupScan(id, lid, x, num_groups);
}
