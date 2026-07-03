# DIG in Brush -- usage

End-to-end workflow for training DINO-feature-embedded Gaussians (DiG) in Brush. A port of the DiG model from [Robot See Robot Do](https://arxiv.org/abs/2409.18121) (reference implementation: [kerrj/dig](https://github.com/kerrj/dig), MIT), running on any Brush backend including Metal.

## 1. One-off: extract DINOv2 features (Python)

The script carries inline dependency metadata (PEP 723), so with [uv](https://docs.astral.sh/uv/) no environment setup is needed — uv resolves torch/torchvision/numpy/pillow into a cached ephemeral env on first run:

```bash
uv run scripts/extract_dino_features.py --data /path/to/dataset
```

(Or run it with any Python env that has those four packages: `python scripts/extract_dino_features.py ...`. MPS is used automatically on Apple silicon.)

`/path/to/dataset` is a normal Brush dataset (COLMAP layout with an `images/` folder, or images directly in the folder). The reference recipe is the default; `--model` (any `facebookresearch/dinov2` hub model, patch size derived automatically), `--max-size` (default 1260), and `--pca-dim` (default 96) are tunable and recorded in `meta.json`. This writes:

```
dataset/dino_features/<image_stem>.npy   # [H/14, W/14, 96] f32 per view
dataset/dino_features/pca.npy            # [768, 96] PCA projection
dataset/dino_features/meta.json
```

Numerically identical recipe to the reference `kerrj/dig` cache (DINOv2 ViT-B/14, features ÷10, PCA→96). Takes a few minutes for ~100–200 images on an M-series GPU.

## 2. Train in Brush

```bash
cargo run --release --bin brush -- /path/to/dataset --dino
```

Feature training is explicit opt-in via `--dino` (a warning is emitted if the flag is set but no `dino_features/` cache is found; without the flag, any feature cache is ignored). Knobs:

- `--dino` — enable DiG feature training (required).
- `--dino-view` — start the viewer in the DINO feature view instead of RGB (it can also be toggled live, see below).
- `--dino-loss-weight <w>` — weight of the feature MSE (default `1.0`).
- `--features-dir-name <name>` — feature folder name (default `dino_features`).
- `--dino-feature-dim <d>` — per-Gaussian stored feature dimension (default `64`).
- `--dino-rescale-factor <r>` — rendered-feature upscale vs. the GT feature-map resolution (default `5`).
- `--dino-lr <lr>` / `--dino-lr-end <lr>` — feature/MLP LR schedule (defaults `1e-2` → `1e-3` over 6k steps, then held).
- `--dino-nn-reg-weight <w>` — 3-NN feature-variance regularizer weight, active after step 1000 (default `0.01`; `0` disables).

All defaults match the reference recipe (`crates/brush-train/src/dig.rs`); the decoder shape (hidden width 64, no bias, output dim = the cache's channel count) is fixed by the architecture. The reference trains DiG for 8000 iterations, so `--total-train-iters 8000` is a reasonable match for object captures.

Note: LOD generation (`--lod-levels > 0`) resets the trainer between levels and is not supported together with feature training — leave it at the default `0`.

## 3. Outputs

At every checkpoint export (and at the end of training), next to the exported PLY:

```
export_dir/<name>.ply                 # the splat, as usual
export_dir/<name>_dig_features.npy    # [N, 64] f32 — rows match PLY splat order
export_dir/<name>_dig_mlp.json        # decoder weights: 4 layers, row-major [d_in, d_out], ReLU, no bias
```

To map a Gaussian's stored feature into the 96-d PCA'd DINO space (e.g. for clustering or comparing against DINOv2 features of a new image):

```python
import json, numpy as np
feats = np.load("point_cloud_dig_features.npy")           # [N, 64]
mlp = json.load(open("point_cloud_dig_mlp.json"))
x = feats
for i, layer in enumerate(mlp["layers"]):
    w = np.array(layer["weight"], dtype=np.float32).reshape(layer["shape"])
    x = x @ w
    if i < len(mlp["layers"]) - 1:
        x = np.maximum(x, 0.0)
# x: [N, 96] — same space as dataset/dino_features/*.npy (compare via pca.npy)
```

## Verifying a run

- **Live, in the viewer:** when `--dino` training is active, a **"DINO feature view"** checkbox appears in the scene controls, and `--dino-view` starts with it enabled. It recolors the splats by their learned features — each Gaussian's feature decoded through the MLP, with the top-3 PCA channels mapped to RGB — and refreshes every 50 training steps, so you can watch semantic structure emerge alongside RGB training. Early on it's noise; parts/regions should separate into coherent colors as the DINO MSE drops.
- The training loss should drop noticeably in the first ~500 steps beyond what RGB-only training shows (the DINO MSE dominates early).
- Offline check on exported features: decode with the snippet above, take 3 PCA dims, normalize to [0,1], and use as splat colors — same picture as the viewer toggle.

## Not included (future work)

GARField-style hierarchical grouping, the interactive click-to-segment viewer, camera-pose optimization, and RSRD part tracking.
