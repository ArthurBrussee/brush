# brush-cli — Command Reference (man page)

Complete reference for the headless Brush trainer, `brush-cli`. Generated from and kept in sync
with `brush-cli --help` (run that for the authoritative, version-exact list). Flags added/affected
by the resource-optimization work are marked **★**.

> **Man page:** an installable roff man page lives at `docs/man/brush-cli.1` and is
> **auto-generated from the live clap command** via `cargo run -p brush-cli --example gen-man`
> (`clap_mangen`, a dev-dependency) — so it never drifts. View with `man ./docs/man/brush-cli.1`.

## Synopsis
```
brush-cli [OPTIONS] [PATH_OR_URL]
```

## Description
`brush-cli` trains a 3D Gaussian Splatting model from a dataset and exports `.ply` splats. It is
the lean, headless front end (the GUI app is `brush-app`). Build with `--release` for any real
run — debug is far slower (the CLI warns).

```bash
cargo run --release -p brush-cli -- <dataset_path_or_url> [flags]
# or the built binary:
./target/release/brush-cli <dataset_path_or_url> [flags]
```

The training pipeline (dataset → loader → train loop → export) and how data flows is described in
[data-flow.md](./data-flow.md) and [architecture.md](./architecture.md).

## Argument
| Arg | Meaning |
|---|---|
| `PATH_OR_URL` | Dataset or splat source. A **directory** (COLMAP `sparse/` or Nerfstudio `transforms.json`), a **`.zip`**, a **`.ply`**, or a **URL** (streamed; append `?url=` style sources). Omit only with `--with-viewer`. |

## Global options
| Flag | Default | Meaning |
|---|---|---|
| `--with-viewer` | off | Spawn a viewer to visualize training. (Headless training is the CLI's normal mode; a `PATH_OR_URL` makes this default off.) |
| `-h, --help` | | Print help. |
| `-V, --version` | | Print version. |

---

## Training options
| Flag | Default | Meaning |
|---|---|---|
| `--total-train-iters <N>` | 30000 | Total training steps. |
| `--render-mode <default\|mip>` | default | `mip` enables the Mip-Splatting 2D filter (anti-aliasing). |
| `--lr-mean <f>` | 2e-5 | Start LR for Gaussian means (positions). |
| `--lr-mean-end <f>` | 2e-7 | Final LR for means (exponential schedule). |
| `--mean-noise-weight <f>` | 50.0 | Noise added to means of low-opacity Gaussians (escapes bad minima). |
| `--lr-coeffs-dc <f>` | 2e-3 | LR for the base SH (RGB / DC) coefficients. |
| `--lr-coeffs-sh-scale <f>` | 10.0 | Divisor applied to the LR of higher SH orders. |
| `--lr-opac <f>` | 0.012 | LR for opacity. |
| `--lr-scale <f>` | 5e-3 | LR for scales. |
| `--lr-rotation <f>` | 2e-3 | LR for rotations. |
| `--ssim-weight <f>` | 0.2 | SSIM weight in the loss (L1 weight = 1 − this). |
| `--opac-decay <f>` | 0.004 | Opacity decay factor. |
| `--background-color <R> <G> <B>` | 0 0 0 | Base training background color. |
| `--background-noise-strength <f>` | 0.1 | Uniform background-color noise per step, in [−s,+s] clamped to [0,1]. |
| `--random-init-scene-scale <f>` | auto | Depth for random frustum init when no SfM points exist (auto-estimated from camera spacing otherwise). |

## Refine options (densification — controls splat count `N`)
| Flag | Default | Meaning |
|---|---|---|
| **★ `--max-splats <N>`** | 10000000 | Upper bound on splat count. Lower it to cap memory & speed (e.g. `300000`) — a *blunt* density cap; small quality cost. |
| `--refine-every <N>` | 200 | Steps between refine passes (prune/split/grow). ≈ images needed to cover the scene. |
| `--growth-grad-threshold <f>` | 0.0025 | Lower → faster growth. |
| `--growth-select-fraction <f>` | 0.25 | Fraction of growth-eligible splats that actually grow. |
| `--growth-stop-iter <N>` | 15000 | Stop growing after this iteration. |
| `--split-at-screen-size <f>` | 0.5 | Force-split splats whose screen extent exceeds this fraction (0 disables). |
| `--match-alpha-weight <f>` | 0.1 | L1 weight on alpha when input views have transparency. |
| `--lpips-loss-weight <f>` | 0.0 | LPIPS perceptual-loss weight (0 = off; >0 materializes an f32 GT image — more memory). |

## LOD options (post-training level-of-detail baking)
| Flag | Default | Meaning |
|---|---|---|
| `--lod-levels <N>` | 0 | LOD levels to generate after training (0 = off). Each exports a `_lodN.ply`. |
| `--lod-refine-steps <N>` | 5000 | Refinement steps per LOD level. |
| `--lod-decimation-keep <1-100>` | 50 | % of Gaussians kept per LOD level. |
| `--lod-image-scale <1-100>` | 50 | % image scale per LOD level. |

## Model options
| Flag | Default | Meaning |
|---|---|---|
| **★ `--sh-degree <0-4>`** | 3 | Spherical-harmonics degree (view-dependent color). **Dominates per-splat memory & `.ply` size.** `2` ≈ quality-neutral at −20–32% memory (safe default); `1` = −30–40% memory, small SSIM cost on glossy scenes (fine on diffuse). See [optimization-results.md](./optimization-results.md). |

## Dataset options
| Flag | Default | Meaning |
|---|---|---|
| `--max-frames <N>` | all | Cap the number of frames loaded. |
| `--max-resolution <px>` | 1920 | Downscale images above this (JPEG uses fast IDCT scale-on-decode). Bounds decode memory. |
| `--eval-split-every <N>` | none | Hold out every Nth image for evaluation (PSNR/SSIM). |
| `--subsample-frames <N>` | none | Load only every Nth frame. |
| `--subsample-points <N>` | none | Load only every Nth initial SfM point. |
| `--alpha-mode <masked\|transparent>` | auto | Interpret an alpha channel / `masks/` folder as masking vs transparency. |
| **★ `--max-cache-bytes <bytes>`** | min(6 GiB, ¼ RAM) | Host packed-batch cache budget. **Auto-set to ¼ of system RAM (cap 6 GiB)** — ~4 GiB on a 16 GiB machine — with LRU eviction. Lower for tight memory (e.g. `2147483648` ≈ 2 GiB). |

## Process options
| Flag | Default | Meaning |
|---|---|---|
| `--seed <N>` | 42 | RNG seed (fix for reproducible runs/benchmarks). |
| `--start-iter <N>` | 0 | Resume from this iteration. |
| `--eval-every <N>` | 1000 | Run evaluation every N steps (needs `--eval-split-every`). |
| `--eval-save-to-disk` | off | Save eval renders to `--export-path`. |
| `--export-every <N>` | 5000 | Export a checkpoint `.ply` every N steps. |
| **★ `--export-path <path>`** | `./{dataset}_exports/` | Output dir; `{dataset}` interpolates the dataset name. **Relative paths anchor to the dataset's parent dir; absolute paths are used verbatim.** |
| `--export-name <tmpl>` | `export_{iter}.ply` | Output filename template (`{iter}` interpolates). |
| **★ `--log-resources-every <N>`** | off | Log GPU memory-in-use + splat count every N steps (e.g. `[resources] iter 1000: 412345 splats, GPU 980 MiB in use`). Off by default (the GPU query stalls behind queued work). |

## Rerun options (rerun.io visualization, native only)
| Flag | Default | Meaning |
|---|---|---|
| `--rerun-enabled` | off | Enable rerun.io logging (also enables GPU-memory logging to rerun). |
| `--rerun-log-train-stats-every <N>` | 50 | Cadence for basic training stats. |
| `--rerun-log-splats-every <N>` | none | Cadence for logging the full point cloud (heavy). |
| `--rerun-log-distribution-every <N>` | 1000 | Cadence for scale/opacity/anisotropy distribution stats. |
| `--rerun-max-img-size <px>` | 512 | Max size of dataset images logged to rerun. |

> Every CLI flag can also be supplied via an `args.txt` in the dataset (merged at load); CLI flags override it.

---

## End-of-run summary
At completion the CLI prints, e.g.:
```
Training took 3m50s — 5000 steps (21.7/s), peak 563029 splats, final eval PSNR 25.57 / SSIM 0.8968
```
(steps/s, peak splat count, and final eval quality — sourced from the training message stream.)

## Examples

**Basic train (30k steps) and export:**
```bash
brush-cli ./datasets/tandt/truck
```

**Headless, fixed iters, with eval and a checkpoint at the end:**
```bash
brush-cli ./datasets/tandt/truck \
  --total-train-iters 7000 --eval-split-every 8 --eval-every 1000 \
  --export-every 7000 --export-path /tmp/out --export-name truck_{iter}.ply --seed 42
```

**Low-memory profile (≈ −20% RAM, quality-neutral — the recommended resource preset):**
```bash
brush-cli ./datasets/tandt/truck --sh-degree 2
# more aggressive (diffuse scenes / tight memory): --sh-degree 1 --max-splats 400000
```

**Cap host cache + watch resources on a constrained machine:**
```bash
brush-cli ./big_dataset --max-cache-bytes 2147483648 --log-resources-every 500
```

**Resume from a checkpoint:**
```bash
brush-cli ./datasets/tandt/truck --start-iter 5000 --total-train-iters 30000
```

**Stream a splat/dataset from a URL with the viewer:**
```bash
brush-cli "https://host/scene.ply" --with-viewer
```

**Generate LODs after training:**
```bash
brush-cli ./scene --lod-levels 3 --lod-decimation-keep 50 --lod-image-scale 50
```

## Notes
- **Always `--release`** for performance-sensitive runs.
- Input formats: COLMAP (`sparse/0/{cameras,images,points3D}.{bin,txt}` + `images/`), Nerfstudio
  (`transforms.json`), `.ply`/`.compressed.ply`, `.zip` of the above, or a URL.
- Memory/perf guidance and measured results: [performance.md](./performance.md),
  [optimization-results.md](./optimization-results.md). Profiling: [profiling.md](./profiling.md).
