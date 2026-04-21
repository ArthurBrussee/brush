#import helpers;

@group(0) @binding(0) var<storage, read> transforms: array<f32>;
@group(0) @binding(1) var<storage, read> raw_opacities: array<f32>;
@group(0) @binding(2) var<storage, read_write> global_from_compact_gid: array<u32>;
@group(0) @binding(3) var<storage, read_write> depths: array<f32>;
@group(0) @binding(4) var<storage, read_write> num_visible: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> intersect_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> num_intersections: atomic<u32>;
@group(0) @binding(7) var<storage, read> uniforms: helpers::ProjectUniforms;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) num_wgs: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let global_gid = helpers::get_global_id(wid, num_wgs, lid, WG_SIZE);

    if global_gid >= uniforms.total_splats {
        return;
    }

    // Read transform data: means(3) + quats(4) + log_scales(3)
    let base = global_gid * 10u;
    let mean = vec3f(transforms[base], transforms[base + 1u], transforms[base + 2u]);

    let img_size = uniforms.img_size;
    let viewmat = uniforms.viewmat;
    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let mean_c = R * mean + viewmat[3].xyz;

    // project_forward is the sole visibility gate. All guards are
    // positive-phrased so NaN reliably fails them (NaN compares
    // unordered -> every comparison returns false -> the `!ok`
    // branch bails). Finite out-of-distribution values
    // (huge log_scales, huge positions) pass. calc_cov2d handles
    // overflow by clamping.
    let mean_c_ok =
        helpers::is_finite_f32(mean_c.x) &&
        helpers::is_finite_f32(mean_c.y) &&
        mean_c.z >= 0.01 && mean_c.z <= 1e10;
    if !mean_c_ok {
        return;
    }

    let log_scale_raw = vec3f(transforms[base + 7u], transforms[base + 8u], transforms[base + 9u]);
    let scale = exp(log_scale_raw);
    let scale_ok =
        helpers::is_finite_f32(scale.x) && scale.x >= 0.0 &&
        helpers::is_finite_f32(scale.y) && scale.y >= 0.0 &&
        helpers::is_finite_f32(scale.z) && scale.z >= 0.0;
    if !scale_ok {
        return;
    }

    let quat_unorm = vec4f(transforms[base + 3u], transforms[base + 4u], transforms[base + 5u], transforms[base + 6u]);
    let quat_norm_sqr = dot(quat_unorm, quat_unorm);
    let quat_ok = quat_norm_sqr >= 1e-6 && helpers::is_finite_f32(quat_norm_sqr);
    if !quat_ok {
        return;
    }

    let raw_opac = raw_opacities[global_gid];
    if !helpers::is_finite_f32(raw_opac) {
        return;
    }

    // project_visible uses `normalize` — we do too, so both dispatches
    // produce bit-identical quaternions. (The old `x * inverseSqrt(dot)`
    // phrasing was mathematically equivalent but the compiler was free to
    // pick a different implementation in each kernel.)
    let quat = normalize(quat_unorm);

    var opac = helpers::sigmoid(raw_opac);
    var cov2d = helpers::calc_cov2d(scale, quat, mean_c, uniforms.focal, uniforms.img_size, uniforms.pixel_center, viewmat);
    opac *= helpers::compensate_cov2d(&cov2d);

    // Last gate: non-finite cov2d (Inf scale overflow, NaN math) — both
    // kernels agree on this splat being invisible.
    if !helpers::is_finite_cov2d(cov2d) {
        return;
    }

    let mean2d = uniforms.focal * mean_c.xy * (1.0 / mean_c.z) + uniforms.pixel_center;

    if !(opac >= 1.0 / 255.0) {
        return;
    }

    let power_threshold = log(255.0f * opac);

    // Same conic + compute_bbox_extent that MG runs on the stored conic,
    // so the two dispatches agree on tile count.
    let conic = helpers::inverse(cov2d);
    let conic_packed = vec3f(conic[0][0], conic[0][1], conic[1][1]);
    let extent = helpers::compute_bbox_extent(conic_packed, power_threshold);
    if !(extent.x >= 0.0 && extent.y >= 0.0) {
        return;
    }

    let bbox_on_screen = mean2d.x + extent.x > 0.0
        && mean2d.x - extent.x < f32(uniforms.img_size.x)
        && mean2d.y + extent.y > 0.0
        && mean2d.y - extent.y < f32(uniforms.img_size.y);
    if !bbox_on_screen {
        return;
    }
    let tile_bbox = helpers::get_tile_bbox(mean2d, extent, uniforms.tile_bounds);
    let tile_bbox_min = tile_bbox.xy;
    let tile_bbox_max = tile_bbox.zw;
    let tile_bbox_width = tile_bbox_max.x - tile_bbox_min.x;
    let num_tiles_bbox = (tile_bbox_max.y - tile_bbox_min.y) * tile_bbox_width;

    var num_tiles_hit = 0u;
    for (var tile_idx = 0u; tile_idx < num_tiles_bbox; tile_idx++) {
        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min.x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min.y;
        let rect = helpers::tile_rect(vec2u(tx, ty));
        if helpers::will_primitive_contribute(rect, mean2d, conic_packed, power_threshold) {
            num_tiles_hit += 1u;
        }
    }

    intersect_counts[global_gid] = num_tiles_hit;
    atomicAdd(&num_intersections, num_tiles_hit);

    let write_id = atomicAdd(&num_visible, 1u);
    global_from_compact_gid[write_id] = global_gid;
    depths[write_id] = mean_c.z;
}
