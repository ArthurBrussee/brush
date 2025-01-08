# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
# Train a whole image, see how that progresses for reference.
# Maybe capture first training step as something... ?
import math
from pathlib import Path
import mediapy
import numpy as np
import torch
from gsplat.rendering import rasterization
from gsplat.rendering import spherical_harmonics

from PIL import Image

from safetensors.torch import save_file

import dataclasses

# %%
g_device = torch.device("cuda:0")
DEFAULT_TILE_SIZE = 16
SH_DEGREE = 3
SH_COUNT = (SH_DEGREE + 1) ** 2


# %%
# Helper functions
@dataclasses.dataclass(frozen=True)
class Camera:
    viewmat: torch.Tensor
    focal: np.ndarray
    w: np.ndarray
    h: np.ndarray
    background: torch.Tensor

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor.to("cuda:0")

def fov_to_focal(fov: np.ndarray, img_size: np.ndarray) -> np.ndarray: 
    return 0.5 * np.array(img_size, dtype=np.float32) / np.tan(0.5 * fov).astype(np.float32)

def basic_camera(w: int, h: int) -> Camera:
    background = torch.zeros(3, device=g_device)
    fov_x = np.array(math.pi / 2.0, dtype=np.float32)

    pos = torch.tensor([0.123, 0.456, -8.0], dtype=torch.float32, device=g_device)

    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, -pos[0]],
            [0.0, 1.0, 0.0, -pos[1]],
            [0.0, 0.0, 1.0, -pos[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=g_device,
        dtype=torch.float32,
    )
    w = np.array(w, dtype=np.float32)
    h = np.array(h, dtype=np.float32)
    focal = fov_to_focal(fov_x, w)
    return Camera(viewmat=viewmat, 
                  w=np.array(w, dtype=np.float32), 
                  h=np.array(h, dtype=np.float32), 
                  focal=focal, 
                  background=background)

# %%
def execute_test(means, log_scales, quats, coeffs, opacities, name: str):
    crab_img = image_path_to_tensor("./crab.png")
    crab_img = torch.concat([crab_img, torch.zeros_like(crab_img)[..., 0:1]], dim=2)

    means.requires_grad = True
    log_scales.requires_grad = True
    quats.requires_grad = True
    coeffs.requires_grad = True
    opacities.requires_grad = True

    H, W, _ = crab_img.shape
    cam = basic_camera(W, H)

    fx, fy = cam.focal, cam.focal
    cx, cy = np.array(W, dtype=np.float32) / 2.0, np.array(H, dtype=np.float32) / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
    viewmats = cam.viewmat[None, ...]
    K = K[None, ...]
    K = torch.tensor(K, device=g_device).float().detach()

    camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
    # Do SH separately, as we do not want to gradients from the directions to the means.
    dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
    dirs = dirs.detach()
    
    colors = spherical_harmonics(3, dirs, coeffs[None], masks=None)  # [C, N, 3]
    colors = colors + 0.5
    # colors = colors.clamp(min=0.0)

    quat_norms = torch.nn.functional.normalize(quats, dim=1)

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quat_norms,
        scales=log_scales.exp(),
        opacities=torch.sigmoid(opacities),
        colors=colors,
        viewmats=viewmats,  # [C, 4, 4]
        Ks=K,  # [C, 3, 3]
        width=W,
        height=H,
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

    depths = info["depths"]
    conics = info["conics"]
    xys = info["means2d"]

    xys.retain_grad()
    conics.retain_grad()

    out_img = torch.concat([render_colors[0], render_alphas[0]], dim=2).to(device="cuda")
    out_img.retain_grad()
    
    mediapy.show_image((out_img.detach().cpu().numpy() * 255.0).astype(np.uint8), width=W*4, height=H*4)

    loss = ((out_img - crab_img) ** 2).mean()
    loss.backward()

    tensors = {
        "means": means,
        "scales": log_scales,
        "coeffs": coeffs,
        "quats": quats,
        "opacities": opacities,

        "depths": depths[0],
        "xys": xys[0],
        "conics": conics[0],
        "v_xy": xys.grad[0],
        "v_conics": conics.grad[0],
        
        "v_means": means.grad,
        "v_scales": log_scales.grad,
        "v_coeffs": coeffs.grad,
        "v_quats": quats.grad,
        "v_opacities": opacities.grad,
        
        "out_img": out_img,
        "v_out_img": out_img.grad,
    }
    save_file(tensors, f"./{name}.safetensors")

# %%
# Super simple case: a few splats visible in a tiny image.
def tiny_case():
    torch.manual_seed(14)
    num_points = 4
    means = 10.5 * (torch.rand(num_points, 3, device=g_device) - 0.5)
    log_scales = (torch.rand(num_points, 3, device=g_device) * 2.5).log()
    coeffs = (torch.rand(num_points, SH_COUNT, 3, device=g_device) - 0.5) * 0.5
    
    u = torch.rand(num_points, 1, device=g_device)
    v = torch.rand(num_points, 1, device=g_device)
    w = torch.rand(num_points, 1, device=g_device)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ],
        -1,
    )
    opacities = torch.rand(num_points, device=g_device) * 0.5 + 0.5
    execute_test(means, log_scales, quats, coeffs, opacities, "tiny_case")

tiny_case()

# %%

# Basic case: a bunch of splats visible.
def gen_basic_case():
    torch.manual_seed(3)
    num_points = 16
    means = 10.0 * (torch.rand(num_points, 3, device=g_device) - 0.5)
    log_scales = torch.rand(num_points, 3, device=g_device).log() * 0.5
    coeffs = (torch.rand(num_points, SH_COUNT, 3, device=g_device) - 0.5) * 0.5

    u = torch.rand(num_points, 1, device=g_device)
    v = torch.rand(num_points, 1, device=g_device)
    w = torch.rand(num_points, 1, device=g_device)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ],
        -1,
    )
    opacities = torch.rand(num_points, device=g_device) * 0.5 + 0.5
    execute_test(means, log_scales, quats, coeffs, opacities, "basic_case")

gen_basic_case()


# %%

# Bigger test case: Lots of splats saturaing the image.
def gen_mix_case():
    torch.manual_seed(6)
    num_points = 76873
    means = 2000.0 * (torch.rand(num_points, 3, device=g_device) - 0.5)
    log_scales = (torch.rand(num_points, 3, device=g_device) * 15.0 + 0.05).log()
    coeffs = (torch.rand(num_points, SH_COUNT, 3, device=g_device) - 0.5) * 0.5
    
    u = torch.rand(num_points, 1, device=g_device)
    v = torch.rand(num_points, 1, device=g_device)
    w = torch.rand(num_points, 1, device=g_device)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ],
        -1,
    )
    opacities = torch.rand(num_points, device=g_device)
    execute_test(means, log_scales, quats, coeffs, opacities, "mix_case")

gen_mix_case()