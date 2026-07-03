#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "numpy",
#   "pillow",
# ]
# ///
"""Extract DINOv2 feature maps for a dataset (offline preprocessing for DIG-in-Brush).

For each image, computes a [h/14, w/14, 768] DINOv2 feature map (matching DIG's
dino_dataloader.py math: resize to max-dim 1260 rounded to /14, ImageNet
normalize, get_intermediate_layers, /10), fits a PCA over all images' features,
and writes per-view [h/14, w/14, pca_dim] f32 .npy files plus pca.npy and
meta.json into <data>/dino_features/.

Assumptions:
  - Runs on the host machine as a one-off preprocessing step, outside of
    Brush (which is pure Rust and never invokes Python).
  - PyTorch (+ torchvision) must be importable with a working backend
    (MPS on Apple silicon, CUDA, or CPU). With `uv run` this is handled
    automatically via the inline dependency metadata above; with plain
    `python` the active environment must already provide it.
  - Network access on first run: DINOv2 weights are fetched via torch.hub.

Usage:
    uv run scripts/extract_dino_features.py --data /path/to/dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DEFAULT_MAX_SIZE = 1260
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PCA_MAX_ROWS = 4_000_000
NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def get_img_resolution(H, W, max_size, p):
    # Matches DIG's dino_dataloader.py get_img_resolution.
    if H < W:
        new_W = (max_size // p) * p
        new_H = (int((H / W) * max_size) // p) * p
    else:
        new_H = (max_size // p) * p
        new_W = (int((W / H) * max_size) // p) * p
    return new_H, new_W


def find_images(data_dir: Path) -> list[Path]:
    img_dir = data_dir / "images"
    if not img_dir.is_dir():
        img_dir = data_dir
    paths = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def pick_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--pca-dim", type=int, default=96, help="PCA output dimension")
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vitb14",
        help="torch.hub facebookresearch/dinov2 model name (patch size is derived from it)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=DEFAULT_MAX_SIZE,
        help="Max image dimension before feature extraction (rounded down to the patch size)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/mps/cpu)")
    args = parser.parse_args()

    image_paths = find_images(args.data)
    if not image_paths:
        raise SystemExit(f"No images found in {args.data} or {args.data / 'images'}")

    device = pick_device(args.device)
    print(f"Found {len(image_paths)} images, using device {device}")

    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model = model.to(device).eval()
    patch = model.patch_embed.patch_size[0]

    out_dir = args.data / "dino_features"
    out_dir.mkdir(exist_ok=True)

    feats_per_image = []  # CPU tensors, [h/14, w/14, 768]
    image_shape = None
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        if image_shape is None:
            image_shape = [H, W]
        h, w = get_img_resolution(H, W, args.max_size, patch)
        tensor = transforms.functional.to_tensor(img)
        tensor = transforms.functional.resize(
            tensor, (h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        tensor = NORMALIZE(tensor).to(device)
        with torch.no_grad():
            desc = model.get_intermediate_layers(tensor[None], reshape=True)[0]
            desc = desc.squeeze().permute(1, 2, 0) / 10  # [h/14, w/14, 768]
        feats_per_image.append(desc.cpu())
        print(f"[{i + 1}/{len(image_paths)}] {path.name}: features {tuple(desc.shape)}")

    # Fit PCA over all images' features.
    feat_dim = feats_per_image[0].shape[-1]
    flat = torch.cat([f.reshape(-1, feat_dim) for f in feats_per_image], dim=0)
    fit_rows = flat
    if flat.shape[0] > PCA_MAX_ROWS:
        gen = torch.Generator().manual_seed(0)
        idx = torch.randperm(flat.shape[0], generator=gen)[:PCA_MAX_ROWS]
        fit_rows = flat[idx]
        print(f"Subsampled {PCA_MAX_ROWS} of {flat.shape[0]} rows for PCA fit")
    print(f"Fitting PCA {feat_dim} -> {args.pca_dim} on {fit_rows.shape[0]} rows...")
    pca_matrix = torch.pca_lowrank(fit_rows, q=args.pca_dim, niter=20)[2]  # [768, pca_dim]

    # Project and save per-view feature maps.
    for path, feats in zip(image_paths, feats_per_image):
        h, w, _ = feats.shape
        projected = (feats.reshape(-1, feat_dim) @ pca_matrix).reshape(h, w, args.pca_dim)
        out = np.ascontiguousarray(projected.numpy().astype(np.float32))
        np.save(out_dir / f"{path.stem}.npy", out)

    np.save(out_dir / "pca.npy", pca_matrix.numpy().astype(np.float32))
    meta = {
        "model": args.model,
        "patch_size": patch,
        "pca_dim": args.pca_dim,
        "scale_div": 10,
        "max_size": args.max_size,
        "image_shape": image_shape,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(image_paths)} feature maps, pca.npy, meta.json to {out_dir}")


if __name__ == "__main__":
    main()
