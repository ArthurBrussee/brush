# Brush Documentation

Contributor-facing documentation for [Brush](../README.md), a cross-platform 3D Gaussian
Splatting engine in Rust (Burn + wgpu). Diagrams are [Mermaid](https://mermaid.js.org/) and
render directly on GitHub.

## Contents
| Doc | What it covers |
|---|---|
| [cli-reference.md](./cli-reference.md) | **Complete `brush-cli` manual** — every flag (grouped, with defaults), the end-of-run summary, and usage examples |
| [man/brush-cli.1](./man/brush-cli.1) | Installable **roff man page** for all flags (`man ./docs/man/brush-cli.1`). **Auto-generated** from the live clap command — regenerate with `cargo run -p brush-cli --example gen-man` (never drifts from `--help`). |
| [cli-internals.md](./cli-internals.md) | **Binary internals** — every function/type in `brush-cli` (`main`, `Cli`, `validate`, `build_process`, `run_headless`, `run_cli_ui`), the helper API it calls, and a control-flow diagram |
| [architecture.md](./architecture.md) | Workspace layout, crates, backend, the training pipeline, concurrency model |
| [data-flow.md](./data-flow.md) | How a dataset becomes trained splats: VFS → Scene → loader → train loop → export |
| [performance.md](./performance.md) | The resource/memory model (esp. Apple Silicon unified memory) and the optimization plan |
| [speed-and-algorithms.md](./speed-and-algorithms.md) | Where training time goes, and research-grounded algorithmic levers to go faster (SH warmup, budgeted densification, tighter culling, Metal atomics) |
| [optimization-results.md](./optimization-results.md) | **Before/after results**: the low-resource profile (−30% memory, −60% splat size at ~baseline quality), what shipped, what was rejected |
| [profiling.md](./profiling.md) | How to measure CPU/GPU/memory, run benchmarks, and use Tracy/rerun |

## Quick start (build & run)
```bash
# Desktop app (workspace default member)
cargo run --release

# Headless CLI trainer
cargo run --release -p brush-cli -- <dataset_path_or_url> --total-train-iters 30000

# Tests / benchmarks
cargo test --all
cargo bench
```
Use `--release` for any performance-sensitive run; debug builds are far slower (the CLI warns).

## Where things live
- **Apps:** `apps/brush-{app,cli,c,js}`
- **Compute:** `crates/brush-{render,render-bwd,loss,sort,prefix-sum,cube}`
- **Training:** `crates/brush-train`
- **Data:** `crates/brush-{dataset,vfs,serde}`, `crates/colmap-reader`
- **Orchestration:** `crates/brush-process`, `crates/brush-async`

See [architecture.md](./architecture.md) for the full map.

> Engineering working notes, the optimization backlog, and session logs live in the
> (gitignored) `memory/` directory at the repo root — they are local working memory, not part
> of the published docs.
