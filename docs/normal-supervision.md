# Monocular normal-map supervision

Brush can optionally supervise training with per-frame monocular normal maps
(e.g. produced by [Sapiens2](https://github.com/facebookresearch/sapiens2)),
in addition to the usual photometric loss. This is off by default; enable it
from the CLI flags below or the viewer's training settings (Losses → "Normal
map supervision").

## Dataset format

Alongside a dataset's RGB frames, drop a sibling `normals/` directory
with same-named PNGs:

```
dataset/
  cameras.txt
  images.txt
  points3D.txt
  frame_00001_.png       # RGBA: alpha is a foreground mask
  normals/
    frame_00001_.png     # RGBA, same size/mask, RGB = encoded normal
```

The normal PNG's RGB channels encode a **camera-space unit normal** via the
standard `(n + 1) / 2 * 255` mapping. The alpha channel is the same
foreground mask as the RGB frame's — pixels outside it are ignored by the
loss. Detection is automatic (mirrors the existing `masks/` convention) and
works for the COLMAP, Nerfstudio and RealityCapture loaders: if `normals/`
isn't present, the feature is simply inactive. A normal map that fails to
decode degrades that view to photometric-only supervision (with a warning)
rather than aborting training.

Enable it with:

```
brush-cli <dataset> \
  --normal-loss-weight 0.05 \
  --normal-loss-start-iter 5000
```

- `--normal-loss-weight` (default `0.0`, disabled): weight in the total loss.
  Kept low deliberately — normal supervision is a regularizer on geometry,
  not a primary reconstruction signal, and monocular predictions are only
  locally, not globally, consistent. `0.01`–`0.1` matches values used in the
  literature (DN-Splatter, MonoGSDF).
- `--normal-loss-start-iter` (default `5000`): delays supervision until
  rough geometry has formed from the photometric loss alone. Turning it on
  too early fights still-forming, noisy splats.

Once active, the loss runs **every step** — normals composite in the same
render pass as color (see below), so there is no extra pipeline cost to
amortize and no throttling flag.

## The architecture: a generic per-splat feature channel

The rasterizer (`crates/brush-render`) supports an optional per-splat
`[N, 3]` **feature** input alongside the usual splat parameters, gated at
kernel-compile time (CubeCL `#[comptime]` monomorphization — the
feature-free configuration compiles to exactly the kernels a non-feature
build would produce, so the viewer/eval paths and feature-less training pay
nothing):

1. `project_visible` gathers the feature into 3 extra lanes of the packed
   per-splat `projected_splats` buffer (9 → 12 lanes).
2. `rasterize` alpha-composites those lanes into 3 extra output channels
   (`out_img` becomes `[H, W, 7]`: RGBA + feature), exactly like color but
   **signed** (no `max(.., 0)` clamp) and with no background contribution.
   This is the "alpha-composite normals according to the rendering
   equation" formulation from DN-Splatter/2DGS/GOF.
3. The hand-written backward extends in lockstep: `rasterize_backwards`
   emits 3 extra gradient lanes (and feeds the features into the alpha
   gradient's remainder dot-product, mirroring the color terms), and
   `project_backwards` scatters them to a dense `[N, 3]` feature gradient
   that burn's autodiff hands back to whatever tensor graph produced the
   feature input.

The correctness of that backward extension is pinned by finite-difference
tests (`crates/brush-bench-test/tests/finite_diff.rs`): gradients w.r.t.
the feature input itself, cross-terms (a feature-only loss driving
position/scale/opacity gradients through the blend weights), and an
equivalence test asserting the RGBA output is unchanged when features are
enabled.

Normal supervision (`crates/brush-train/src/normals.rs`) is then a small
consumer of that mechanism, per step once active:

1. **Per-splat world-space pseudo-normal.** Vanilla 3D Gaussians don't carry
   an explicit normal, but as training flattens a splat against a surface,
   its shortest-scale local axis converges toward the true surface normal —
   a standard proxy used in several 3DGS-plus-normal-regularization papers.
   `world_space_normals` finds that axis via `argmin` over the raw
   log-scales (monotone in the true scales, including under the Mip
   3D-filter fold, so the fold op-chain is skipped; the Int `argmin` is a
   free stop-gradient), builds a one-hot local-axis vector, rotates it into
   world space with the existing `quaternion_vec_multiply` helper, and
   normalizes the result (the stored quats are unnormalized). The result is
   sign-oriented to face the camera using `Tensor::sign()`, which has a zero
   backward gradient in burn — a proper stop-gradient.
2. **Rotate into camera space** (`rotate_to_camera_space`) using the
   camera's fixed rotation.
3. **Render.** The `[N, 3]` camera-space normal is passed to
   `render_splats_with_features` as the feature input of the *main* render —
   one pass produces the RGBA image and the composited normal map.
   Gradients flow into position/rotation/scale/opacity (via the blend
   weights, same as color) *and* through the derivation above into the
   shared `transforms` param; summing the photometric and normal losses
   into one scalar before the single `backward()` accumulates both
   correctly — ordinary reverse-mode diamond-graph accumulation.
4. **Loss** (`masked_l1_cosine_loss`): `L1 + (1 − cosine similarity)`
   between the rendered and GT normal — both in raw `[-1, 1]` — masked to
   the GT foreground alpha; the same combined form used by MonoSDF and
   DN-Splatter's normal term. The GT is stored as u8 and decoded on-GPU.

## Camera-space axis convention

Sapiens2's documentation states its normals are "3-channel (x, y, z) unit
vectors in the camera coordinate frame," without pinning down the exact
axis directions. Brush's internal camera space is OpenCV-style (+X right,
+Y down, +Z forward into the scene — see `world_to_local()` in
`crates/brush-render/src/camera.rs`); Sapiens2 empirically uses the
OpenGL-style convention (+Y up, +Z toward the viewer).

The correction lives as a single named constant, `GT_AXIS_SIGN` in
`crates/brush-train/src/normals.rs`, applied to the **GT** at the loss
boundary (the sign flips are involutive, so one constant maps either
direction) — the rendered feature stays a plain brush-convention normal.
It was calibrated by scoring all 8 axis-sign combinations by masked mean
cosine similarity against a sample dataset's GT maps; `[+X, −Y, −Z]` won
decisively (0.980 vs 0.762 for the runner-up, on a model trained with this
loss). If a different normal predictor or dataset ever needs a different
convention, rerun the calibration harness —

```
cargo run -p brush-bench-test --example normal_calib --release -- \
  <dataset_dir> <trained.ply>
```

— and change that one constant.

## What's deliberately out of scope here

- **Normal-consistency regularizers** (à la 2DGS's depth-normal
  consistency): the feature channel gives a rendered normal map, but no
  rendered depth to check it against — that would be a further (now
  straightforward) extension of the same feature-lane mechanism.
