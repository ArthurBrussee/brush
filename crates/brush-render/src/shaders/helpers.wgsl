const TILE_WIDTH: u32 = 16u;
const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

// Helper function to compact bits for 2D z-order decoding
fn compact_bits_16(v: u32) -> u32 {
    var x = v & 0x55555555u;
    x = (x | (x >> 1u)) & 0x33333333u;
    x = (x | (x >> 2u)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4u)) & 0x00FF00FFu;
    x = (x | (x >> 8u)) & 0x0000FFFFu;
    return x;
}

// Decode z-order to 2D coordinates
fn decode_morton_2d(morton: u32) -> vec2<u32> {
    let x = compact_bits_16(morton);
    let y = compact_bits_16(morton >> 1u);
    return vec2<u32>(x, y);
}

fn map_1d_to_2d(id: u32, tiles_per_row: u32) -> vec2<u32> {
    let tile_id = id / TILE_SIZE;
    let within_tile_id = id % TILE_SIZE;

    let tile_x = tile_id % tiles_per_row;
    let tile_y = tile_id / tiles_per_row;

    return vec2u(tile_x * TILE_WIDTH, tile_y * TILE_WIDTH) + decode_morton_2d(within_tile_id);
}

struct RenderUniforms {
    // View matrix transform world to view position.
    viewmat: mat4x4f,

    // Focal of camera (fx, fy)
    focal: vec2f,
    // Img resolution (w, h)
    img_size: vec2u,

    tile_bounds: vec2u,
    // Camera center (cx, cy).
    pixel_center: vec2f,

    // Position of camera (xyz + pad)
    camera_position: vec4f,

    // Degree of sh coeffecients used.
    sh_degree: u32,

#ifdef UNIFORM_WRITE
    // Number of visible gaussians, written by project_forward.
    // This needs to be non-atomic for other kernels as you can't have
    // read-only atomic data.
    num_visible: atomic<u32>,
#else
    // Number of visible gaussians.
    num_visible: u32,
#endif

    total_splats: u32,
    max_intersects: u32,

    // Nb: Alpha is ignored atm.
    background: vec4f,
}

struct ProjectedSplat {
    xy_x: f32,
    xy_y: f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
}

fn create_projected_splat(xy: vec2f, conic: vec3f, color: vec4f) -> ProjectedSplat {
    return ProjectedSplat(xy.x, xy.y, conic.x, conic.y, conic.z, color.r, color.g, color.b, color.a);
}

struct PackedVec3 {
    x: f32,
    y: f32,
    z: f32,
}

fn get_bbox(center: vec2f, dims: vec2f, bounds: vec2u) -> vec4u {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    let min = vec2u(clamp(center - dims, vec2f(0.0), vec2f(bounds)));
    let max = vec2u(clamp(center + dims + vec2f(1.0), vec2f(0.0), vec2f(bounds)));
    return vec4u(min, max);
}

fn get_tile_bbox(pix_center: vec2f, pix_extent: vec2f, tile_bounds: vec2u) -> vec4u {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    let tile_center = pix_center / f32(TILE_WIDTH);
    let tile_extent = pix_extent / f32(TILE_WIDTH);
    return get_bbox(tile_center, tile_extent, tile_bounds);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0f / (1.0f + exp(-x));
}

// device helper to get 3D covariance from scale and quat parameters
fn quat_to_mat(quat: vec4f) -> mat3x3f {
    // quat to rotation matrix
    let w = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    // See https://www.songho.ca/opengl/gl_quaternion.html
    return mat3x3f(
        vec3f(
            (1.0 - 2.0 * (y2 + z2)),
            (2.0 * (xy + wz)),
            (2.0 * (xz - wy)), // 1st col
        ),
        vec3f(
            (2.0 * (xy - wz)),
            (1.0 - 2.0 * (x2 + z2)),
            (2.0 * (yz + wx)), // 2nd col
        ),
        vec3f(
            (2.0 * (xz + wy)),
            (2.0 * (yz - wx)),
            (1.0 - 2.0 * (x2 + y2)) // 3rd col
        ),
    );
}

fn scale_to_mat(scale: vec3f) -> mat3x3f {
    return mat3x3(
        vec3f(scale.x, 0.0, 0.0),
        vec3f(0.0, scale.y, 0.0),
        vec3f(0.0, 0.0, scale.z)
    );
}

fn calc_cov3d(scale: vec3f, quat: vec4f) -> mat3x3f {
    let M = quat_to_mat(quat) * scale_to_mat(scale);
    return M * transpose(M);
}

fn calc_cam_J(mean_c: vec3f, focal: vec2f, img_size: vec2u, pixel_center: vec2f) -> mat3x2f {
    let lims_pos = (1.15f * vec2f(img_size.xy) - pixel_center) / focal;
    let lims_neg = (-0.15f * vec2f(img_size.xy) - pixel_center) / focal;
    let rz = 1.0 / mean_c.z;

    // Get normalized image coords +- clipped to the frustum.
    let uv_clipped = clamp(mean_c.xy * rz, lims_neg, lims_pos);

    let duv_dxy = focal * rz;
    let J = mat3x2f(
        vec2f(duv_dxy.x, 0.0),
        vec2f(0.0, duv_dxy.y),
        -duv_dxy * uv_clipped
    );

    return J;
}

fn calc_cov2d(cov3d: mat3x3f, mean_c: vec3f, focal: vec2f, img_size: vec2u, pixel_center: vec2f, viewmat: mat4x4f) -> mat2x2f {
    let R = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let covar_cam = R * cov3d * transpose(R);

    let J = calc_cam_J(mean_c, focal, img_size, pixel_center);

    var cov2d = J * covar_cam * transpose(J);

    // add a little blur along axes and save upper triangular elements
    cov2d[0][0] += COV_BLUR;
    cov2d[1][1] += COV_BLUR;
    return cov2d;
}

fn inverse(m: mat2x2f) -> mat2x2f {
    let det = determinant(m);
    if (det <= 0.0f) {
        return mat2x2f(vec2f(0.0), vec2f(0.0));
    }
    let inv_det = 1.0f / det;
    return mat2x2f(vec2f(m[1][1] * inv_det, -m[0][1] * inv_det), vec2f(-m[0][1] * inv_det, m[0][0] * inv_det));
}

const COV_BLUR: f32 = 0.3;

fn cov_compensation(cov2d: vec3f) -> f32 {
    let cov_orig = cov2d - vec3f(COV_BLUR, 0.0, COV_BLUR);
    let det_orig = cov_orig.x * cov_orig.z - cov_orig.y * cov_orig.y;
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    return sqrt(max(0.0, det_orig / det));
}

fn calc_sigma(pixel_coord: vec2f, conic: vec3f, xy: vec2f) -> f32 {
    let delta = pixel_coord - xy;
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
}

fn calc_vis(pixel_coord: vec2f, conic: vec3f, xy: vec2f) -> f32 {
    return exp(-calc_sigma(pixel_coord, conic, xy));
}

fn compute_bbox_extent(cov2d: mat2x2f, power_threshold: f32) -> vec2f {
    return vec2f(
        sqrt(2.0f * power_threshold * cov2d[0][0]),
        sqrt(2.0f * power_threshold * cov2d[1][1]),
    );
}

fn tile_rect(tile: vec2u) -> vec4f {
    let rect_min = vec2f(tile * TILE_WIDTH);
    let rect_max = rect_min + f32(TILE_WIDTH);
    return vec4f(rect_min.x, rect_min.y, rect_max.x, rect_max.y);
}

// Based on method from StopThePop: https://arxiv.org/pdf/2402.00525.
fn will_primitive_contribute(rect: vec4f, mean: vec2f, conic: vec3f, power_threshold: f32) -> bool {
    let x_left = mean.x < rect.x;
    let x_right = mean.x > rect.z;
    let in_x_range = !(x_left || x_right);

    let y_above = mean.y < rect.y;
    let y_below = mean.y > rect.w;
    let in_y_range = !(y_above || y_below);

    if (in_x_range && in_y_range) {
        return true;
    }

    let closest_corner = vec2f(
        select(rect.z, rect.x, x_left),
        select(rect.w, rect.y, y_above)
    );

    let width = rect.z - rect.x;
    let height = rect.w - rect.y;
    let d = vec2f(
        select(-f32(width), f32(width), x_left),
        select(-f32(height), f32(height), y_above)
    );

    let diff = mean - closest_corner;
    let t_max = vec2f(
        select(clamp((d.x * conic.x * diff.x + d.x * conic.y * diff.y) / (d.x * conic.x * d.x), 0.0f, 1.0f), 0.0f, in_y_range),
        select(clamp((d.y * conic.y * diff.x + d.y * conic.z * diff.y) / (d.y * conic.z * d.y), 0.0f, 1.0f), 0.0f, in_x_range)
    );

    let max_contribution_point = closest_corner + t_max * d;
    let max_power_in_tile = calc_sigma(mean, conic, max_contribution_point);

    return max_power_in_tile <= power_threshold;
}

fn ceil_div(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}

fn as_vec(packed: PackedVec3) -> vec3f {
    return vec3f(packed.x, packed.y, packed.z);
}

fn as_packed(vec: vec3f) -> PackedVec3 {
    return PackedVec3(vec.x, vec.y, vec.z);
}
