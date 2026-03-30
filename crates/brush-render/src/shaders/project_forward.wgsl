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

    // Check if this splat is 'valid' (aka visible). Phrase as positive to bail on NaN.
    if mean_c.z < 0.01 || mean_c.z > 1e10 {
        return;
    }

    let scale = exp(vec3f(transforms[base + 7u], transforms[base + 8u], transforms[base + 9u]));
    var quat = vec4f(transforms[base + 3u], transforms[base + 4u], transforms[base + 5u], transforms[base + 6u]);

    // Skip any invalid rotations. This will mean overtime
    // these gaussians just die off while optimizing. For the viewer, the importer
    // atm always normalizes the quaternions.
    // Phrase as positive to bail on NaN.
    let quat_norm_sqr = dot(quat, quat);
    if quat_norm_sqr < 1e-6 {
        return;
    }

    quat *= inverseSqrt(quat_norm_sqr);

    var opac = helpers::sigmoid(raw_opacities[global_gid]);
    let cov3d = helpers::calc_cov3d(scale, quat);
    var cov2d = helpers::calc_cov2d(cov3d, mean_c, uniforms.focal, uniforms.img_size, uniforms.pixel_center, viewmat);
    opac *= helpers::compensate_cov2d(&cov2d);

    // compute the projected mean
    let mean2d = uniforms.focal * mean_c.xy * (1.0 / mean_c.z) + uniforms.pixel_center;

    if opac < 1.0 / 255.0 {
        return;
    }

    let power_threshold = log(255.0f * opac);
    let extent = helpers::compute_bbox_extent(cov2d, power_threshold);
    if extent.x < 0.0 || extent.y < 0.0 {
        return;
    }

    if mean2d.x + extent.x <= 0 || mean2d.x - extent.x >= f32(uniforms.img_size.x) ||
       mean2d.y + extent.y <= 0 || mean2d.y - extent.y >= f32(uniforms.img_size.y) {
        return;
    }

    // Count tile intersections for this splat.
    let conic = helpers::inverse(cov2d);
    let conic_packed = vec3f(conic[0][0], conic[0][1], conic[1][1]);
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
