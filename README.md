# Brush — SeedeXR optimized fork

> A resource-aware fork of [**Brush**](https://github.com/ArthurBrussee/brush) (Arthur Brussee),
> maintained by **Seede XR Group Limited / Seede XR Studios**, focused on training 3D Gaussian
> Splats efficiently on commodity hardware — especially **16 GiB Apple-Silicon** machines.
> Fork home: **https://github.com/SeedeXR/brush** · upstream credit below.

Brush is a 3D reconstruction engine using [Gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
It runs on a wide range of systems: **macOS/Windows/Linux**, **AMD/Nvidia/Intel** cards,
**Android**, and in a **browser**, via WebGPU-compatible tech and the
[Burn](https://github.com/tracel-ai/burn) machine-learning framework — producing simple,
dependency-free binaries that run nearly anywhere without setup.

---

## ✨ What this fork adds (resource & efficiency work)

Measured on a 10-core, 16 GiB Apple-Silicon machine (release builds, seed-fixed; "noise band"
= PSNR ±0.08, SSIM ±0.002, peak RSS ±30 MB). Full study and data below.

| Improvement | Result |
|---|---|
| **Adaptive host-cache budget** | hardcoded 6 GiB → **min(6 GiB, ¼ of system RAM)** (auto **4 GiB** on 16 GiB), with `--max-cache-bytes` override |
| **Peak training memory** (`--sh-degree 1`, diffuse scenes) | **−30–40 %** at quality within the noise band |
| **Exported `.ply` model size** (`--sh-degree 1`) | **−60 %** |
| **LRU cache eviction** | hot views stay resident under a smaller budget (was refuse-on-full) |
| **Decode-concurrency cap** | bounds peak image-decode memory; throughput-neutral |
| **Observability** | `--log-resources N` (GPU mem + splat count) and a richer end-of-run summary |

Recommended memory-saving defaults: **`--sh-degree 2`** is quality-neutral even on specular scenes
at ~−20 % memory; **`--sh-degree 1`** is best for diffuse scenes or tight memory budgets. See
[`docs/optimization-results.md`](docs/optimization-results.md) for the before/after evidence and
the diffuse-vs-specular caveat.

The per-step rasterization kernels were found to be **already well-optimized** upstream; the one
remaining speed lever (importance-based primitive reduction) is documented as future work in
[`docs/speed-and-algorithms.md`](docs/speed-and-algorithms.md).

---

## 🚀 Quickstart (CLI)

```bash
# Build the headless trainer (always use --release for real runs)
cargo build --release -p brush-cli

# Train a COLMAP/Nerfstudio dataset and export a .ply
./target/release/brush-cli /path/to/dataset

# Low-memory profile (quality-neutral, ~-20% RAM)
./target/release/brush-cli /path/to/dataset --sh-degree 2

# Constrained machine: cap host cache + watch resources
./target/release/brush-cli /path/to/dataset --max-cache-bytes 2147483648 --log-resources-every 500
```

- **Full flag reference:** [`docs/cli-reference.md`](docs/cli-reference.md) · or the man page:
  `man ./docs/man/brush-cli.1`
- **`brush-cli --help`** is the authoritative, version-exact option list.

---

## 📚 Documentation

| Doc | What |
|---|---|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/cli-reference.md](docs/cli-reference.md) | Complete CLI manual (every flag, examples) |
| [docs/man/brush-cli.1](docs/man/brush-cli.1) | Installable roff man page (auto-generated from clap) |
| [docs/cli-internals.md](docs/cli-internals.md) | Binary internals (functions/helpers, control flow) |
| [docs/architecture.md](docs/architecture.md) | Engine architecture & training pipeline |
| [docs/data-flow.md](docs/data-flow.md) | Dataset → loader → train → export data flow |
| [docs/performance.md](docs/performance.md) | Memory/resource model + tuning knobs |
| [docs/speed-and-algorithms.md](docs/speed-and-algorithms.md) | Where training time goes; algorithmic levers |
| [docs/optimization-results.md](docs/optimization-results.md) | **Before/after results** of this fork's work |
| [docs/profiling.md](docs/profiling.md) | How to benchmark and profile |
| [research/](research/) | Research paper (`.md` + `.pdf`) on this study |
| [llms.txt](llms.txt) | LLM-friendly project overview |

---

# Features

## Training
Brush takes in COLMAP data or datasets in the Nerfstudio format. Training is fully supported
natively, on mobile, and in a browser. While training you can interact with the scene and see the
training dynamics live, and compare the current rendering to input views as the training
progresses. It also supports masking images:
- Images with transparency — forces the final splat to match the input transparency.
- A folder of images called `masks` — ignores masked-out parts of the image.

## Viewer
Brush also works well as a splat viewer, including on the web. It can load `.ply` &
`.compressed.ply` files, and stream from a URL (append `?url=`). It can load a `.zip` of splat
files to display as an animation, or a special ply with delta frames (see
[cat-4D](https://cat-4d.github.io/) and [Cap4D](https://felixtaubner.github.io/cap4d/)).

## CLI
The headless trainer is `brush-cli` (this fork's optimization focus). The GUI app (`brush`,
brush-app) additionally offers a viewer. See the [CLI reference](docs/cli-reference.md).

## Rerun
While training, additional data can be visualized with [rerun](https://rerun.io/) (`--rerun-enabled`).
Install with `cargo install rerun-cli`; open `./brush_blueprint.rbl` in the viewer for best results.

## Building Brush
First install Rust (this fork's pinned dependency graph currently needs **1.96+**; upstream lists
1.88+). Run tests with `cargo test --all`.

### Windows/macOS/Linux
`cargo run --release` from the workspace root for an optimized build of the app, or
`cargo build --release -p brush-cli` for just the headless trainer.

### Web
Brush compiles to WASM. See `apps/brush-app/web`; Brush uses
[`wasm-pack`](https://drager.github.io/wasm-pack/). WebGPU support: Chrome 134+ on Windows/macOS.

### Android
One-time: install the Android SDK & NDK, set `ANDROID_NDK_HOME`/`ANDROID_HOME`,
`rustup target add aarch64-linux-android`, `cargo install cargo-ndk`. Then:
```
cargo ndk -t arm64-v8a -o crates/brush-app/app/src/main/jniLibs/ build --release
./gradlew installDebug && adb shell am start -n com.splats.app/.MainActivity
```

## Benchmarks
Rendering and training are generally faster than gsplat. Run kernel benchmarks with `cargo bench`.
(Note: the `brush-sort` bench currently fails to compile against the pinned Burn `main` due to a
feature-unification issue — tracked; production code is unaffected.)

# Acknowledgements

This fork builds directly on **[Brush](https://github.com/ArthurBrussee/brush) by Arthur Brussee** —
all credit for the engine is theirs. Additional thanks (from upstream):

[**gSplat**](https://github.com/nerfstudio-project/gsplat), for their reference kernels ·
**Peter Hedman, George Kopanas & Bernhard Kerbl**, for discussions & pointers ·
**The Burn team**, for help & improvements ·
**Raph Levien**, for the [original GPU radix sort](https://github.com/googlefonts/compute-shader-101/pull/31) ·
**GradeEterna**, for feedback and scenes.

The optimization study in this fork was **AI-assisted research conducted with Claude (Anthropic)**;
see [`research/`](research/).

# Disclaimer

This is *not* an official Google product. Upstream Brush is a forked public version of
[the google-research repository](https://github.com/google-research/google-research/tree/master/brush_splat).
Apache-2.0 licensed (see `LICENSE`).
