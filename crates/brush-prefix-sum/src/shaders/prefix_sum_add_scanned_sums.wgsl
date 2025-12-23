#import prefix_sum_helpers as helpers

@compute
@workgroup_size(helpers::THREADS_PER_GROUP, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let id = helpers::get_global_id(wid, num_wgs, lid);
    let workgroup_id = helpers::get_workgroup_id(wid, num_wgs);

    if (id < arrayLength(&helpers::output)) {
        helpers::output[id] += helpers::input[workgroup_id];
    }
}
