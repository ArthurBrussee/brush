# Brush - universal splats

https://github.com/user-attachments/assets/b7f55b9c-8632-49f9-b34b-d5de52a7a8b0

Brush is a 3D reconstruction engine, using [Gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). It aims to be highly portable, flexible and fast. 3D reconstruction should be accessible to everyone!

Brush works on a wide range of systems: **macOS/windows/linux**, **AMD/Nvidia** cards, **Android**, and in a **browser**. To achieve this, it uses WebGPU compatible tech, like the [Burn](https://github.com/tracel-ai/burn) machine learning framework, which has a portable [`wgpu`](https://github.com/gfx-rs/wgpu) backend. This project is currently still a proof of concept, and doesn't yet implement the many extensions to gaussian splatting that have been developed, nor is performance optimal yet.

[**Try the (experimental) web demo** <img src="https://cdn-icons-png.flaticon.com/256/888/888846.png" alt="chrome logo" width="24"/>
](https://arthurbrussee.github.io/brush-demo)

_NOTE: This only works on desktop Chrome 129+ currently (Oct 2024). Firefox and Safari are hopefully supported [soon](https://caniuse.com/webgpu), but currently even firefox nightly and safari technical preview do not work_

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/TbxJST2BbC)](https://discord.gg/TbxJST2BbC)

## Features

The demo can load pretrained ply splats, and can load datasets to train on. Currently only two formats are supported. A .zip file containing:
- An `images` & `sparse` folder with [`COLMAP`](https://github.com/colmap/colmap) data
- A .json and images, like the [nerfstudio format](https://docs.nerf.studio/quickstart/data_conventions.html).
  - You can specify a custom transforms_train.json and transforms_eval.json split.

While training you can interact with the scene and see the training dynamics live, and compare the current rendering to training / eval views as the training progresses.

## Web

https://github.com/user-attachments/assets/4c70f892-cfd2-419f-8098-b0e20dba23c7

## Rerun

https://github.com/user-attachments/assets/f679fec0-935d-4dd2-87e1-c301db9cdc2c

While training, additional data can be visualized with the excellent [rerun](https://rerun.io/). To install rerun on your machine, please follow their [instructions](https://rerun.io/docs/getting-started/installing-viewer). Open the ./brush_blueprint.rbl in the viewer for best results.

## Mobile

https://github.com/user-attachments/assets/d6751cb3-ff58-45a4-8321-77d3b0a7b051

Training on a pixel 7

# Why

Machine learning for real time rendering has a lot of potential, but most popular ML tools don't align well with it. Rendering requires low latency, usually involve dynamic shapes, and it's not pleasant to attempt to ship apps with large PyTorch/Jax/CUDA deps calling out to python in a rendering loop. The usual fix is to write a separate training and inference application. Brush on the other hand, written in rust using `wgpu` and `burn`, can produce simple dependency free binaries, and can run on nearly all devices.

# Getting started
Install rust 1.81+ and run `cargo run` or `cargo run --release`. You can run tests with `cargo test --all`. Brush uses the wonderful [rerun](https://rerun.io/) for additional visualizations while training.
It currently requires rerun 0.19 however, which isn't released yet.

### Windows/macOS/Linux
Simply `cargo run` or `cargo run --release` from the workspace root.

Note: Linux has not yet been tested but *should* work. Windows works well, but does currently only works on Vulkan.

### Web
This project uses [`trunk`](https://github.com/trunk-rs/trunk) to build for the web. Install trunk, and then run `trunk serve` or `trunk serve --release` to run a development server.

WebGPU is still a new standard, and as such, only the latest versions of Chrome work currently. Firefox nightly should work but unfortunately crashes currently.

The public web demo is registered for the [subgroups origin trial](https://chromestatus.com/feature/5126409856221184). To run the web demo for yourself, please enable the "Unsafe WebGPU support" flag in Chrome.

### Android
To build on Android, see the more detailed README instructions at crates/brush-android.

### iOS
Brush *should* work on iOs but there is currently no project setup to do so.

## Technical details

Brush is split into various crates. A quick overview of the different responsibilities are:

- `brush-render` is the main crate that pulls together the kernels into rendering functions.
- `brush-train` has code to actually train Gaussians, and handle larger scale optimizations like splitting/cloning gaussians etc.
- `brush-train-loop` default training loop using brush-train.
- `brush-app` handles the UI and integrating the training loop. This is also the binary target for the  web, and mac/Windows/Linux.
- `brush-android` handles running on android.
- `brush-wgsl` handles some kernel inspection for generating CPU-side structs and interacing with [naga-oil](https://github.com/bevyengine/naga_oil) to handle shader imports.
- `brush-dataset` handles importing different training data formats.
- `brush-prefix-sum` and `brush-sort` are only compute kernels and should be largely independent of Brush (other than `brush-wgsl`).
- `rrfd` is a small extension of [`rfd`](https://github.com/PolyMeilex/rfd)

### Kernels

The kernels are written in a "sparse" style, that is, only work for visible gaussians is done, though the final calculated gradients are dense. Brush uses a GPU radix sort based on [FidelityFX](https://www.amd.com/en/products/graphics/technologies/fidelityfx.html) (see `crates/brush-sort`). The sorting is done in two parts - first splats are sorted only by depth, then sorted by their tile ID, which saves some sorting time compared to sorting both depth and tile ids at the same time.

Compatibility with WebGPU does bring some challenges, even with (the excellent) [wgpu](https://github.com/gfx-rs/wgpu).
- WebGPU lacks native atomic floating point additions, and a software CAS loop has to be used.
- GPU readbacks have to be async on WebGPU. A rendering pass can't do this unless the whole rendering becomes async, which has its own perils, and isn't great for an UI. The reference tile renderer requires reading back the number of "intersections" (each visible tile of a gaussian is one intersection), but this is not feasible. This is worked around by assuming a worst case. To reduce the number of tiles the rasterizer culls away unused tiles by intersecting the gaussian ellipses with the screenspace tiles.

The WGSL kernels use [naga_oil](https://github.com/bevyengine/naga_oil) to manage imports. brush-wgsl additionally does some reflection to generate rust code to send uniform data to a kernel. In the future, it might be possible to port the kernels to Burns new [`CubeCL`](https://github.com/tracel-ai/cubecl) language, which is much more ergonomic and would allow generating CUDA / rocM kernels. It might also be possible to integrate with George Kopanos' [Slang kernels](https://github.com/google/slang-gaussian-rasterization).

### Benchmarks

Rendering performance is expected to be very competitive with gSplat, while training performance is still a bit slower. You can run some benchmarks using `cargo bench`. The performance of the splatting forward and backwards kernel are faster than the _legacy_ gSplat kernels as they use some new techniques for better performance, but they haven't been compared yet to the more recent gSplat kernels. End-to-end training performance is also still slower, due to other overheads.

For additional profiling, you can use [tracy](https://github.com/wolfpld/tracy) and run with `cargo run --release --feature=tracy`.

### Quality

Quality is similar, but for now still somewhat lagging behind the original GS implementation. This is likely due to some suboptimal splitting/cloning heuristics.

| Scene      | Brush   | GS paper|
|------------|---------|---------|
| Bicycle@7K | 23.4    | 23.604  |
| Garden@7k  | 26.1    | 26.245  |
| Stump@7k   | 25.3    | 25.709  |

# Acknowledgements

[**gSplat**](https://github.com/nerfstudio-project/gsplat), for their reference version of the kernels

**Peter Hedman, George Kopanas & Bernhard Kerbl**, for the many discussions & pointers.

**The Burn team**, for help & improvements to Burn along the way

**Raph Levien**, for the [original version](https://github.com/googlefonts/compute-shader-101/pull/31) of the GPU radix sort.

# Disclaimer

This is *not* an official Google product. This repository is a forked public version of [the google-research repository](https://github.com/google-research/google-research/tree/master/brush_splat)
