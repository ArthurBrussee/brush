//! Monocular normal-map supervision.
//!
//! The per-splat pseudo-normal (shortest-scale axis, oriented to face the
//! camera, rotated into camera space) is computed with plain tensor ops
//! here, then handed to the renderer as its generic `[N, 3]` per-splat
//! feature input (`brush_render::bwd::render_splats_with_features`). The
//! rasterizer alpha-composites it into three extra output channels in the
//! *same* pass as the color render — projection, sorting and tile mapping
//! are shared, so the marginal cost is just the extra blend lanes.
//! Gradients flow back through the compositing into the feature tensor
//! (and from there through the tensor-op derivation below into the shared
//! `transforms` param), and into position/rotation/scale/opacity via the
//! blend weights. See `docs/normal-supervision.md` for the full writeup.

use brush_render::{camera::Camera, gaussian_splats::Splats};
use burn::tensor::{Device, Int, Tensor, TensorData, s};

use crate::quat_vec::quaternion_vec_multiply;

/// Broadcast a small constant vector to `[n, K]`. Built on the inner
/// (non-autodiff) device and lifted via `Tensor::from_inner` — mirroring how
/// GT data is lifted in [`normal_loss`] below — so it can combine with
/// splat-derived tensors that may be on the autodiff graph without tripping
/// burn-dispatch's cross-backend assert (a plain `Tensor::from_floats(..,
/// &device)` does not automatically pick up the autodiff-ness of `device`).
fn broadcast_row<const K: usize>(vals: [f32; K], n: usize, device: &Device) -> Tensor<2> {
    let inner: Tensor<1> = Tensor::from_floats(vals, &device.clone().inner());
    let lifted: Tensor<1> = Tensor::from_inner(inner);
    lifted.reshape([1, K]).expand([n as i32, K as i32])
}

/// Axis-sign correction between brush's internal camera space (OpenCV-style:
/// +X right, +Y down, +Z forward into the scene — see
/// `brush_dataset::formats::opengl_c2w_to_pose`, which performs the same
/// Y/Z flip in the opposite direction to convert *into* this convention) and
/// the Sapiens2 normal-map encoding, which empirical inspection of
/// `~/Documents/circ_colmap/normals/` showed follows the opposite
/// (OpenGL-style: +Y up, +Z toward the viewer) convention: the blue channel
/// (Z) reads high at the center of a camera-facing surface, and red (X)
/// increases left-to-right, matching a shared X axis with Y and Z flipped.
///
/// Applied to the *GT* at the loss boundary (the sign flips are involutive,
/// so the same constant maps either direction) — the rendered feature stays
/// a plain brush-convention camera-space normal.
///
/// Calibrated with the `normal_calib` example
/// (`crates/brush-bench-test/examples/normal_calib.rs`): masked mean
/// cosine over all 8 sign combinations against a model trained with this
/// loss for 15k steps — `[+X, -Y, -Z]` won decisively at 0.980 vs 0.762
/// for the runner-up. Rerun that example to recalibrate for a different
/// normal predictor; if it disagrees, flip signs here — nowhere else.
const GT_AXIS_SIGN: [f32; 3] = [1.0, -1.0, -1.0];

/// Per-splat world-space unit pseudo-normal: the splat's local axis with the
/// smallest scale (standard proxy for vanilla anisotropic 3D Gaussians —
/// splats naturally flatten against the surfaces they represent as training
/// progresses, so this axis converges toward the true surface normal),
/// oriented to face the camera.
fn world_space_normals(splats: &Splats, camera: &Camera) -> Tensor<2> {
    let quats = splats.rotations(); // [N, 4] (w, x, y, z), unnormalized
    let means = splats.means(); // [N, 3]
    // Which local axis (x/y/z) has the smallest scale, per splat, from the
    // *raw* log-scales: `exp` is monotone, and the min-scale fold (when
    // set) maps every axis of a splat through `s -> sqrt(s^2 + f^2)` with
    // one shared `f` — also monotone — so the argmin is identical to the
    // folded `scales()` while skipping the whole fold op-chain. `argmin`
    // returns an Int tensor, which is inherently outside the float autodiff
    // graph — this index selection is a free stop-gradient, same idea as
    // the `Tensor::sign()` use below.
    let log_scales = splats.log_scales(); // [N, 3]
    let n = log_scales.dims()[0];
    let device = log_scales.device();
    let axis_idx = log_scales.argmin(1).squeeze_dim::<1>(1); // [N] Int
    let local_axis = axis_idx.float().one_hot::<2>(3); // [N, 3] one-hot on the shortest axis

    // Rotate that local axis into world space. The quats are unnormalized
    // (`rotations()` is a raw slice), so the rotated vector's length is
    // |q|^2 — normalize the *output*, which is equivalent to normalizing
    // the quat and cheaper.
    let world_axis = quaternion_vec_multiply(quats, local_axis); // [N, 3]
    let axis_len = world_axis
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12);
    let world_axis = world_axis.div(axis_len);

    // Orient to face the camera: flip any normal pointing away from it.
    // `Tensor::sign()` has a zero backward gradient (confirmed against
    // burn's vendored autodiff ops), so this is a proper stop-gradient, not
    // a hand-rolled detach.
    let cam_pos = broadcast_row(
        [camera.position.x, camera.position.y, camera.position.z],
        n,
        &device,
    );
    let to_cam = cam_pos.sub(means);
    let sign = to_cam.mul(world_axis.clone()).sum_dim(1).sign(); // [N, 1]
    world_axis.mul(sign)
}

/// Rotate a `[N, 3]` world-space normal tensor into camera space using the
/// camera's fixed (non-learned) rotation.
fn rotate_to_camera_space(world_normal: Tensor<2>, camera: &Camera) -> Tensor<2> {
    let n = world_normal.dims()[0];
    let device = world_normal.device();
    // `camera.rotation` is the local(camera)-to-world rotation
    // (`Camera::local_to_world`); we need the inverse to go world -> camera.
    let world_to_cam = camera.rotation.inverse();
    let q = broadcast_row(
        [
            world_to_cam.w,
            world_to_cam.x,
            world_to_cam.y,
            world_to_cam.z,
        ],
        n,
        &device,
    );
    quaternion_vec_multiply(q, world_normal)
}

/// Per-splat camera-space unit pseudo-normal in brush's camera convention,
/// `[N, 3]`, on the autodiff graph of the splats' params. This is the
/// feature tensor handed to `render_splats_with_features`.
pub fn splat_camera_normals(splats: &Splats, camera: &Camera) -> Tensor<2> {
    rotate_to_camera_space(world_space_normals(splats, camera), camera)
}

/// Masked `L1 + (1 - cosine similarity)` normal loss (the MonoSDF-style
/// combined term also used by DN-Splatter's normal supervision), averaged
/// over pixels where the GT foreground mask (`gt[..,..,3]`) is set. `pred`
/// and `gt` are both `[H, W, 3]`; `mask` is `[H, W, 1]`.
fn masked_l1_cosine_loss(pred: Tensor<3>, gt: Tensor<3>, mask: Tensor<3>) -> Tensor<1> {
    let diff_l1 = (pred.clone() - gt.clone()).abs().sum_dim(2); // [H, W, 1]

    let dot = pred.clone().mul(gt.clone()).sum_dim(2); // [H, W, 1]
    let pred_norm = pred.powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);
    let gt_norm = gt.powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);
    let cosine = dot.div(pred_norm.mul(gt_norm));
    let cosine_loss = cosine.neg().add_scalar(1.0); // 1 - cosine

    let per_pixel = (diff_l1 + cosine_loss).mul(mask.clone()); // [H, W, 1]
    let mask_count = mask.sum().clamp_min(1.0);
    per_pixel.sum() / mask_count
}

/// Compute the masked L1 + cosine loss between a rendered normal map and
/// the GT monocular prior.
///
/// `pred_normal` is the `[H, W, 3]` composited feature image from
/// `SplatOutputDiff::features` (raw `[-1, 1]` camera-space normals, brush
/// convention, on the autodiff graph). `gt_normal_data` is `[H, W, 4]` u8
/// (RGB = `(n+1)/2 * 255` encoded normal in the predictor's convention,
/// A = foreground mask — see `brush_dataset::scene::normal_sample_to_data`);
/// it's decoded to `[-1, 1]` and converted to brush's camera convention via
/// [`GT_AXIS_SIGN`] here.
pub fn normal_loss(pred_normal: Tensor<3>, gt_normal_data: TensorData, device: &Device) -> Tensor<1> {
    // GT is pure data — decode it on the inner (non-autodiff) device, then
    // lift it onto the autodiff backend to combine with `pred` (mirrors
    // `train.rs`'s handling of `gt_rgb_diff` for the LPIPS loss).
    let inner_device = device.clone().inner();
    let gt_int: Tensor<3, Int> = Tensor::from_data(gt_normal_data, &inner_device);
    let gt_rgba = gt_int.float().div_scalar(255.0);
    let gt_encoded = gt_rgba.clone().slice(s![.., .., 0..3]);
    let sign: Tensor<1> = Tensor::from_floats(GT_AXIS_SIGN, &inner_device);
    let gt_normal_inner = (gt_encoded.mul_scalar(2.0).add_scalar(-1.0)).mul(sign.reshape([1, 1, 3]));
    let mask_inner = gt_rgba.slice(s![.., .., 3..4]).greater_elem(0.5).float();

    let gt_normal: Tensor<3> = Tensor::from_inner(gt_normal_inner);
    let mask: Tensor<3> = Tensor::from_inner(mask_inner);

    masked_l1_cosine_loss(pred_normal, gt_normal, mask)
}
