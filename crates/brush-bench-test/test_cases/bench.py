# %%
import math
import os
import gc
import time
from pathlib import Path
import mediapy
import numpy as np
import torch
from gsplat.rendering import rasterization
from gsplat.rendering import spherical_harmonics

import matplotlib.pyplot as plt

from PIL import Image

from safetensors.torch import save_file
from safetensors import safe_open

import dataclasses

# %%
# Helper functions
@dataclasses.dataclass(frozen=True)
class Camera:
    viewmat: torch.Tensor
    focal: float
    w: int
    h: int
    background: torch.Tensor

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor.to("cuda:0")

def fov_to_focal(fov: float, img_size: int) -> float: 
    return 0.5 * float(img_size) / math.tan(0.5 * fov)

def basic_camera(w: int, h: int) -> Camera:
    background = torch.zeros(3, device=g_device)
    fov_x = math.pi / 2.0


    pos = torch.tensor([0.123, 0.456, -8.0], device=g_device)

    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, -pos[0]],
            [0.0, 1.0, 0.0, -pos[1]],
            [0.0, 0.0, 1.0, -pos[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=g_device,
    )
    focal = fov_to_focal(fov_x, w)
    return Camera(viewmat=viewmat, w=w, h=h, focal=focal, background=background)

# %%
g_device = torch.device("cuda:0")
DEFAULT_TILE_SIZE = 16
SH_DEGREE = 3
SH_COUNT = (SH_DEGREE + 1) ** 2


# %%
def load_bench_tensors(point_frac: float, mean_mult: float, grad: bool):
    with safe_open("./bench_data.safetensors", framework="pt", device="cpu") as f:
        means: torch.Tensor = f.get_tensor("means")
        num_points = int(means.shape[0] * point_frac)
        means = means[0:num_points, ...] * mean_mult
        log_scales: torch.Tensor = f.get_tensor("scales")[0:num_points, ...]
        coeffs: torch.Tensor = f.get_tensor("coeffs")[0:num_points, ...]
        quats: torch.Tensor = f.get_tensor("quats")[0:num_points, ...]
        opacities: torch.Tensor = f.get_tensor("opacities")[0:num_points, ...]

    means = means.to(device=g_device).detach()
    log_scales = log_scales.to(device=g_device).detach()
    coeffs = coeffs.to(device=g_device).detach()
    quats = quats.to(device=g_device).detach()
    opacities = opacities.to(device=g_device).detach()

    means.requires_grad = grad
    log_scales.requires_grad = grad
    coeffs.requires_grad = grad
    quats.requires_grad = grad
    opacities.requires_grad = grad

    return means, log_scales, coeffs, quats, opacities

# %%

BENCH_DENSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DENSE_MULT = 0.25

TARGET_SAMPLE_COUNT = 40
INTERNAL_ITERS = 4

def bench_general(point_frac: float, mean_mult: float, w: int, h: int, calc_grads: bool):
    # Generate camera
    cam = basic_camera(w, h)
    means, log_scales, coeffs, quats, opacities = load_bench_tensors(point_frac, mean_mult, calc_grads)

    fx, fy, cx, cy = cam.focal, cam.focal, w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    viewmats = cam.viewmat[None, ...]
    K = K[None, ...]
    K = torch.tensor(K, device=g_device).float().detach()

    def internal_iter():
        camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
        # Do SH separately, as we do not want to gradients from the directions to the means.
        dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
        dirs = dirs.detach()
        
        colors = spherical_harmonics(0, dirs, coeffs[None], masks=None)  # [C, N, 3]
        colors = colors + 0.5
        colors = colors.clamp(min=0.0)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=log_scales.exp(),
            opacities=torch.sigmoid(opacities),
            colors=colors,
            viewmats=viewmats,  # [C, 4, 4]
            Ks=K,  # [C, 3, 3]
            width=w,
            height=h,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode='classic',
            distributed=False,
            tile_size=DEFAULT_TILE_SIZE,
            near_plane=0.01,
            far_plane=1e12,
            camera_model='pinhole',
        )

        img = torch.cat([render_colors, render_alphas], dim=3)

        if calc_grads:
            loss = img.mean()
            loss.backward()

    def measure_iter():
        start_time = time.perf_counter()
        for _ in range(INTERNAL_ITERS):
            internal_iter()
        torch.cuda.synchronize(g_device)
        bench_time = time.perf_counter() - start_time
        return bench_time
    
    times = [measure_iter() for _ in range(TARGET_SAMPLE_COUNT)]
    ret = np.median(times)
    print(f"bench_times: {ret}")
    return ret

# %%
gc.collect()

bench_times = {}

for grad in [False, True]:
    name = "fwd" if not grad else "bwd"
    print(f"{name} base")
    bench_times[f"{name}_base"] = [bench_general(dens, 1.0, 512, 512, grad) for dens in BENCH_DENSITIES]
    print(f"{name} dense")
    bench_times[f"{name}_dense"] = [bench_general(dens, DENSE_MULT, 512, 512, grad) for dens in BENCH_DENSITIES]
    print(f"{name} hd")
    bench_times[f"{name}_hd"] = [bench_general(dens, 1.0, 1024, 1024, grad) for dens in BENCH_DENSITIES]

# %%
rust_times = {
    "bwd_base": [12.04, 11.48, 15.75, 21.15, 27.11, 31.42, 37.59, 41.36, 45.13, 50.42],
    "bwd_dense": [14.86, 18.36, 20.12, 22.65, 26.06, 30.06, 34.31, 39.51, 43.65, 47.8],
    "bwd_hd": [19.29, 24.32, 31.02, 35.11, 41.33, 48.38, 55.74, 62.54, 69.79, 76.81],
    "fwd_base": [2.679, 3.485, 4.867, 6.565, 7.962, 9.237, 10.89, 11.96, 13.23, 14.91],
    "fwd_dense": [4.608, 5.745, 6.232, 7.115, 8.301, 9.588, 11.37, 12.49, 13.66, 14.75],
    "fwd_hd": [4.432, 5.395, 6.918, 9.053, 11.14, 13.23, 15.05, 15.83, 17.5, 19.22]
}

# %%
for bench_type in ['fwd', 'bwd']:
    # Colors for each type of line
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    # Plot lines
    for key, color in zip(bench_times.keys(), colors):
        if not bench_type in key:
            continue
        rust_values = rust_times[key]
        python_values = bench_times[key]

        counts = np.array(BENCH_DENSITIES) * (2 ** 21)

        plt.plot(counts, np.array(rust_values) / 1000.0, label=key, color=color, linestyle='-') 
        plt.plot(counts, np.array(python_values), color=color, linestyle='--')

    # Customize the plot
    plt.xlabel('Splat count')
    plt.ylabel('Time (seconds)')
    plt.title('Brush vs gsplat (dashed) Implementation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    legend1 = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(legend1)
    plt.legend(loc='upper left') 

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Show the plot
    plt.show()