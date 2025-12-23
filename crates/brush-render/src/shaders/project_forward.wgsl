#define UNIFORM_WRITE

#import helpers;

// Unfiroms contains the splat count which we're writing to.
@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read> means: array<helpers::PackedVec3>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;
@group(0) @binding(3) var<storage, read> log_scales: array<helpers::PackedVec3>;
@group(0) @binding(4) var<storage, read> raw_opacities: array<f32>;

@group(0) @binding(5) var<storage, read_write> global_from_compact_gid: array<u32>;
@group(0) @binding(6) var<storage, read_write> depths: array<f32>;

const WG_SIZE: u32 = 256u;

@compute
@workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let global_gid = helpers::get_global_id(wid, lid, WG_SIZE);

    if global_gid >= uniforms.total_splats {
        return;
    }

    // Project world space to camera space.
    let mean = helpers::as_vec(means[global_gid]);

    let img_size = uniforms.img_size;
    let viewmat = uniforms.viewmat;
    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let mean_c = R * mean + viewmat[3].xyz;

    // Check if this splat is 'valid' (aka visible). Phrase as positive to bail on NaN.
    if mean_c.z < 0.01 || mean_c.z > 1e10 {
        return;
    }

    let scale = exp(helpers::as_vec(log_scales[global_gid]));
    var quat = quats[global_gid];

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

    let extent = helpers::compute_bbox_extent(cov2d, log(255.0f * opac));
    if extent.x < 0.0 || extent.y < 0.0 {
        return;
    }

    if mean2d.x + extent.x <= 0 || mean2d.x - extent.x >= f32(uniforms.img_size.x) ||
       mean2d.y + extent.y <= 0 || mean2d.y - extent.y >= f32(uniforms.img_size.y) {
        return;
    }
    // Now write all the data to the buffers.
    let write_id = atomicAdd(&uniforms.num_visible, 1u);
    global_from_compact_gid[write_id] = global_gid;
    depths[write_id] = mean_c.z;
}
