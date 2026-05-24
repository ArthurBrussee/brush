# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate forward-render reference safetensors for the Rust test
suite. Forward only — gsplat detaches dirs before SH eval (missing the
viewdir→mean path), so its backward isn't a fair reference; Rust
validates its own backward via the finite-diff suite."""

import numpy as np
import torch
from gsplat.rendering import rasterization, spherical_harmonics
from safetensors.torch import save_file

DEVICE = torch.device("cuda:0")
SH_DEGREE = 3
SH_COUNT = (SH_DEGREE + 1) ** 2
W, H = 256, 256
CAM_POS = torch.tensor([0.123, 0.456, -8.0], dtype=torch.float32, device=DEVICE)

# Pinhole intrinsics: 90° fov_x, square pixels.
FOCAL = 0.5 * W / np.tan(np.pi / 4.0)
K = torch.tensor(
    [[[FOCAL, 0, W / 2.0], [0, FOCAL, H / 2.0], [0, 0, 1.0]]],
    device=DEVICE,
    dtype=torch.float32,
)
VIEWMAT = torch.eye(4, device=DEVICE, dtype=torch.float32)
VIEWMAT[:3, 3] = -CAM_POS
VIEWMATS = VIEWMAT[None]


@torch.no_grad()
def render_and_save(means, log_scales, quats, coeffs, opacities, name):
    dirs = means[None, :, :] - torch.inverse(VIEWMATS)[:, None, :3, 3]
    colors = (spherical_harmonics(SH_DEGREE, dirs, coeffs[None]) + 0.5).clamp(min=0.0)
    rgb, alpha, _ = rasterization(
        means=means,
        quats=torch.nn.functional.normalize(quats, dim=1),
        scales=log_scales.exp(),
        opacities=torch.sigmoid(opacities),
        colors=colors,
        viewmats=VIEWMATS,
        Ks=K,
        width=W,
        height=H,
        tile_size=16,
        near_plane=0.01,
        far_plane=1e12,
        camera_model="pinhole",
        rasterize_mode="classic",
    )
    save_file(
        {
            "means": means,
            "scales": log_scales,
            "quats": quats,
            "coeffs": coeffs,
            "opacities": opacities,
            "out_img": torch.cat([rgb[0], alpha[0]], dim=2),
        },
        f"./{name}.safetensors",
    )


def rand(*shape):
    return torch.rand(*shape, device=DEVICE)


# (seed, n, means_scale, log_scales_fn, opacities_fn, name)
CASES = [
    (14, 4, 10.5, lambda n: (rand(n, 3) * 2.5).log(), lambda n: rand(n) * 0.5 + 0.5, "tiny_case"),
    (3, 16, 10.0, lambda n: rand(n, 3).log() * 0.5, lambda n: rand(n) * 0.5 + 0.5, "basic_case"),
    (6, 76873, 2000.0, lambda n: (rand(n, 3) * 15.0 + 0.05).log(), lambda n: rand(n), "mix_case"),
]


if __name__ == "__main__":
    for seed, n, mean_scale, log_scales_fn, opacities_fn, name in CASES:
        torch.manual_seed(seed)
        render_and_save(
            means=mean_scale * (rand(n, 3) - 0.5),
            log_scales=log_scales_fn(n),
            quats=rand(n, 4),
            coeffs=(rand(n, SH_COUNT, 3) - 0.5) * 0.5,
            opacities=opacities_fn(n),
            name=name,
        )
