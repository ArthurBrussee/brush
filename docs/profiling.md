# Profiling & Benchmarking

How to measure Brush so optimizations are data-driven, not guesswork. **Always build
`--release`** — debug numbers are meaningless (the CLI even warns).

## What to measure
| Resource | Tool |
|---|---|
| Peak host memory (RSS) | `/usr/bin/time -l` (macOS) / `/usr/bin/time -v` (Linux) |
| GPU / unified memory | Burn `log_memory` via `--rerun-enabled`; macOS: Activity Monitor, `powermetrics`, Instruments (Metal System Trace) |
| CPU / per-span timing | Tracy (`--features tracy`) |
| Throughput (steps/s) | CLI progress `per_sec`; `cargo bench` for isolated kernels |
| Disk | export cadence/sizes (`--export-every`) |

Every recorded number must state: **machine + dataset (image count, resolution) + exact
command + metric source + git commit**.

## Peak memory of a CLI run
```bash
# macOS: "maximum resident set size" is bytes
/usr/bin/time -l ./target/release/brush-cli ./datasets/slice16 \
  --total-train-iters 1000 --with-viewer false 2>&1 | grep -i "maximum resident"
```

## GPU memory curve (via rerun)
```bash
cargo install rerun-cli   # one-time
./target/release/brush-cli ./datasets/slice16 --rerun-enabled --total-train-iters 1000
# open ./brush_blueprint.rbl in the rerun viewer
```
GPU memory stats are only queried when rerun is recording (commit 6627cba), so this is the
intended way to see them.

## Tracy (span-level CPU/GPU timing)
```bash
CARGO_PROFILE_RELEASE_DEBUG=true cargo run --release --features tracy -p brush-app -- <args>
```
Then connect the Tracy profiler. Span coverage on the hot path is currently sparse
(`crates/brush-process/src/train_stream.rs:64` and a few UI spans) — add `tracing` spans with
stable names when profiling new stages.

## Micro-benchmarks (divan)
```bash
cargo bench -p brush-bench-test           # forward / backward / training step
cargo bench -p brush-sort                 # radix argsort (1M–70M elements)
```
- Forward render: 500K–2.5M splats at 1080p + a resolution sweep.
- Backward: 1M–5M splats with autodiff.
- Training step: trainer loop on synthetic batches.
- These use synthetic splats (`gen_splats()`) or a `test_cases/bench_data.safetensors`
  (gitignored). Source: `crates/brush-bench-test/src/benches.rs`.

## The chunked validation protocol
Run on increasing dataset sizes while watching resources, stopping at the first anomaly:

```
for n in 2 4 8 16 32 64:        # images (64 cap this cycle)
  build --release
  run brush-cli on an n-image slice for a fixed iteration count
  capture peak RSS, peak GPU mem, steps/s, PSNR/SSIM, final splat count
  if memory or time grows super-linearly, or quality drops -> stop and investigate
```

This catches blow-ups cheaply before a full run. Record each row to a CSV and to
`memory/process_contract.md`.

## Reproducibility
Fix `--seed 42`, pin the exact image slice, and record the git commit with every measurement.
Define a quality "noise band" from 3 baseline repeats so regressions are distinguishable from
run-to-run variance.

## Getting test data
Small datasets for the chunked protocol: download from
`https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets` (the `hf` CLI is
available) into `./datasets/` (gitignored). Create 2/4/8/16/32/64-image slices.
