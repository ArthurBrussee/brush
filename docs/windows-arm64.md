# Windows ARM64

Use Rust 1.95+ with the native `aarch64-pc-windows-msvc` toolchain and the Visual Studio C++
ARM64 build tools. Check the `host` line in `rustc -vV`, then build from the
workspace root:

```powershell
cargo build --release --locked -p brush-cli -p brush-app
```

The headless executable is `target/release/brush-cli.exe`; the desktop executable
is `target/release/brush.exe`. Keep `Cargo.lock` when building. It includes the
CubeCL fixes for WebAssembly subgroup declarations and browser GPU initialization.

Windows ARM64 selects DirectX 12 for training, splat rendering, and the desktop viewer. On the Adreno
X1-45 with driver 31.0.148.0, automatic backend selection in the upstream build
failed after splat refinement. Native runs log the selected adapter and backend
when `RUST_LOG=info` is set.

## Headless regression check

Use a Nerfstudio or COLMAP dataset ZIP. This short run deliberately refines every
five iterations to exercise allocation changes and splat growth quickly; it is
not a quality-training preset.

```powershell
$env:RUST_LOG = 'info'
.\target\release\brush-cli.exe 'C:\path\to\dataset.zip' `
    --max-frames 48 --max-resolution 960 --total-train-iters 31 `
    --refine-every 5 --eval-split-every 12 `
    --export-path "$PWD\target\arm64-smoke"
if ($LASTEXITCODE -ne 0) { throw 'Brush training failed' }
```

Check that refinement messages appear, training reports completion, and
`target/arm64-smoke/export_31.ply` exists. For the longer check used below:

```powershell
.\target\release\brush-cli.exe 'C:\path\to\dataset.zip' `
    --max-resolution 1920 --total-train-iters 1001 --refine-every 200 `
    --eval-split-every 20 --eval-every 500 --eval-save-to-disk `
    --export-path "$PWD\target\arm64-long"
if ($LASTEXITCODE -ne 0) { throw 'Brush training failed' }
```

For driver diagnosis, `BRUSH_WGPU_BACKEND` can override both native backends with
`auto`, `dx12`, `vulkan`, `metal`, or `gl`. Only select an API supported on the
machine. For example:

```powershell
$env:BRUSH_WGPU_BACKEND = 'vulkan'
# Repeat the headless check, using a different export directory.
Remove-Item Env:BRUSH_WGPU_BACKEND
```

CLI worker panics and streams that end without a training-complete message must
fail the command. Their regression tests do not need a GPU:

```powershell
cargo test --release --locked -p brush-cli --lib
```

## Verified configuration (2026-10-09)

The fix was developed from upstream Brush
`1388f74c6fe0236f68ee4915564bf00e9d2e3747`, on Windows 11 ARM64 with a
Qualcomm Adreno X1-45 and driver `31.0.148.0`. Rust was `1.99.0`, with host
`aarch64-pc-windows-msvc`.

The lockfile selects Burn `da05a198`, CubeCL `52c5086d`, and CubeK `83e2c815`.
CubeCL includes [the WASM execution fix](https://github.com/tracel-ai/cubecl/commit/2a8814843c319ae07b15453ef35dce32fd8b9997):
avoiding synchronous throughput probes in browsers, WGSL execution registration, and subgroup feature
declarations. The WGSL emitted by the dependency is used without modification.

Tests used two private Nerfstudio datasets with initial point clouds: dataset A
has 382 images; dataset B has 714 images at 1080x1920 and had previously failed
around iteration 200. The short runs used the command above. Long runs used all
images, a maximum resolution of 1920 pixels, 1001 iterations, refinement every
200 iterations, and evaluation every twentieth image. Dataset contents are not
included in this repository.

| Test | Result |
| --- | --- |
| Clean upstream native release, automatic backend | Failed after refinement with an invalid GPU buffer; incorrectly exited successfully. |
| Clean upstream browser build | Failed at synchronous GPU readback during initialization and at subgroup WGSL validation. |
| Final desktop executable, headless, dataset A, 1001 iterations | Passed; 4 refinements, 161,590 splats; PSNR 21.026, SSIM 0.812; all 9,533,810 exported floats finite; exit 0, zero warnings/errors. |
| Final CLI executable, dataset B, 1001 iterations | Passed; 4 refinements, 165,916 splats; PSNR 20.684, SSIM 0.809; all 9,789,044 exported floats finite; exit 0, zero warnings/errors. |
| Fixed native CLI, forced Vulkan, 31 iterations | Device lost after refinement; correctly exited with failure (101). |
| Chrome 153.0.8010.55, dataset A, 1001 iterations | Passed; 4 refinements, 163,721 splats; PSNR 21.066, SSIM 0.812; no GPU errors or nonfinite splat values. |
| Edge 154.0.4258.62, dataset A, 31 iterations, supplied GPU device | Passed; 5 refinements, 104,384 splats; no GPU errors or nonfinite splat values. |
| CLI unit tests | 9 passed, including worker panic, stream error, and missing completion regressions. |

Both final native runs used the default backend and cache settings, without
environment overrides. Refinements occurred at iterations 201, 401, 601, and
801. Exported PLY payload lengths matched their headers; all floating-point
values were finite. Saved evaluation images were inspected and showed coherent,
partially trained reconstructions. The iteration-200 failure did not recur.

Browser tests ran the rebuilt `brush-js` module locally, including GPU readback
of final transforms, SH coefficients, and opacities. The Edge run used
`initExisting` with the demo's device features and limits. These tests do not
change or validate the currently deployed public website.

The successful browser tests used isolated headless profiles with `--enable-gpu`
and `--enable-unsafe-webgpu`. A separate Chrome run without the latter flag
returned no adapter before Brush initialization. Chrome's GPU diagnostics
identified vendor `0x4d4f4351`, device `0x37314430`; this device is explicitly
listed in [Chromium's WebGPU blocklist](https://github.com/chromium/chromium/blob/main/gpu/config/webgpu_blocklist_impl.cc).
Thus the browser results verify the rebuilt module when WebGPU is exposed;
they do not establish support under the browser's default policy. The diagnostic
override was confined to test profiles. Native DirectX 12 needs no browser flags.
The reported Chromium `DXGI_ERROR_DEVICE_HUNG` was not reproduced during the
fixed runs; the default-policy block and upstream shader errors are separate
observed failures. The interactive desktop viewer was not exercised by these
headless tests.

The native fixed builds used release optimization with
`CARGO_PROFILE_RELEASE_LTO=false` and `CARGO_BUILD_JOBS=4`. The clean upstream
native baseline and the fixed WebAssembly build used the regular release
profile with thin LTO. LTO was disabled for the native investigation to reduce
relinking cost; this is a build-time setting, not a GPU workaround.
Both final Windows executables were checked for native ARM64 PE machine type
`0xAA64`. The final native builds and the final `brush-js` WebAssembly rebuild
completed successfully. The nine CLI tests also passed after the Windows-only
cache setting was added. Browser runtime tests preceded that Windows-only
addition; the final native runs exercised it automatically.

### Windows tuning cache

The selected CubeCL revision requests Turso's multiprocess WAL mode, which
Turso 0.8.2's default Windows file backend does not support. Brush therefore
disables the persistent autotuning cache and throughput cache on Windows.
Autotuning still retains its selected kernels in memory for the running
process. Initial tuning must run again after restarting Brush, and throughput
probes are not cached. This avoids failed database writes without modifying
the user's existing cache or vendoring the database dependency.

An embedding application's already-installed CubeCL configuration takes
priority. The `CUBECL_AUTOTUNE_CACHE` and `CUBECL_THROUGHPUT_CACHE` environment
overrides also remain available for diagnosis. Re-enabling either cache with
the locked dependency versions can produce `database is readonly` warnings.

## Large-model corruption on Adreno

A later investigation found a second failure on the same X1-45 and driver
`31.0.148.0`: variable shader addresses wrap at **256 MiB**, even though DX12
advertises a storage-buffer limit of nearly 2 GiB. Uploading and copying the
same buffer back succeeds. A three-invocation compute shader reproduces the
wrong writes without Brush training, Burn tensor operations, or a dataset.
Both FXC and DXC reproduce the dynamic-address failure, with shader bounds
checks enabled or disabled.

At SH degree 3, each splat has 48 float32 color coefficients (192 bytes).
The buffer crosses 256 MiB at splat 1,398,102. In the supplied checkpoints,
corruption started at exactly this boundary: the overflowing tail overwrote
the beginning of the color buffer, and later training propagated invalid
values into positions. The healthy checkpoint contained 1,223,015 splats and
no non-finite values; a later 1,443,141-splat checkpoint contained 133,324
NaNs or infinities. Zeroed color coefficients also explain the gray/white
splats. These measurements establish an addressing failure; they do not show
that the machine ran out of system RAM.

Brush now limits Windows Qualcomm storage buffers to 256 MiB. The effective
training budget is derived from the largest per-splat buffer, including the
backward pass's dummy row and the optional LOD Hessian. At SH degree 3 the
maximum is **1,398,100 splats**. Training reports the limit, stops growth there,
and continues optimizing and replacing splats. Lower requested limits remain
in effect. Other GPUs retain their advertised limits. This is a workaround
for the reproduced driver behavior; it does not repair the driver or enable
larger buffers on it.

Oversized models opened for viewing report an error, and oversized custom
render allocations fail before launching their shaders. Initial training
clouds follow the existing subsampling behavior with a warning that includes
the effective budget. Invalid values detected during refinement are also
reported in the normal training log.

To reproduce the underlying driver bug independently of these guards:

```powershell
cargo run --release --locked -p brush-process --example adreno_buffer_boundary
```

This diagnostic allocates slightly more than 256 MiB and deliberately tests
the GPU's advertised capability. A correct result has `[22, 33]` immediately
past the boundary, with zeros at the beginning. The affected driver writes
`[22, 33]` at the beginning instead, and the diagnostic fails its assertion.
`WGPU_DX12_COMPILER=fxc` selects FXC for this diagnostic; `dxc` requires a
matching `dxcompiler.dll` on the process search path.

The Windows cache workaround also disables CubeCL environment session records.
They use the same Turso database independently of the autotuning cache and can
otherwise retry a failed database open for every newly compiled kernel.
`CUBECL_ENVIRONMENT_RECORDS` remains available as an explicit override.

### Large-model validation

A headless native run resumed the supplied healthy 1,223,015-splat checkpoint
with all 714 images at 1200 pixels. It completed 3,001 additional training
steps (start iteration 5,000, end 8,001), including repeated refinements at
1,398,100 splats. The final export contained no NaNs/infinities and no all-zero
SH rows. Ten matching training views improved in mean PSNR from 24.86 to
26.87 dB and mean SSIM from 0.878 to 0.894; inspected renders showed no bright
colored blobs or spreading gray/white corruption. These are training-view
checks, not held-out quality measurements. Resuming a PLY resets optimizer
state, and the test used an 8,001-step horizon; it is not an exact continuation
of the original 30,000-step run or a full-length validation.

The minimal standalone diagnostic also reproduced the wraparound with only
the required storage-buffer limits enabled. Chrome 153 on this machine
already exposes a 256 MiB storage-binding limit, so it refuses this oversized
binding before shader execution. Brush derives the browser splat budget
from that advertised limit. The native buffer-address defect alone does not
explain the public demo's separate failure at the first refinement with a
large number of images. The public deployment has not been changed.

## Desktop viewer and sleep/wake

The desktop viewer creates its own GPU device, independently of the training
backend. Before this follow-up, eframe selected Vulkan on the X1-45 even while
training used DirectX 12. It also initialized an unused OpenGL context. The
viewer now shares the Windows ARM64 DirectX 12 policy and restricts its instance
to that backend. `BRUSH_WGPU_BACKEND` applies to both devices; `WGPU_BACKEND`
remains an explicit viewer-only override and takes precedence for the viewer.
On other platforms, automatic viewer selection remains unchanged by default.

A reported crash occurred after pausing training and letting the laptop sleep.
Egui failed to allocate a 38,496-byte index staging buffer even though its
existing index buffer had 82,248 bytes of capacity, then panicked again while
tearing down the unused OpenGL context. The first failure is consistent with a
lost viewer device, and matches [egui issue 8450](https://github.com/emilk/egui/issues/8450).
It does not establish memory exhaustion or the large-model address-wrap defect.

The viewer logs its selected adapter and the device-loss reason. Logging starts
before compute initialization so both devices are identifiable with
`RUST_LOG=info`. Eframe 0.36 does not automatically rebuild a lost device: this
backend change is a mitigation, not recovery from arbitrary GPU resets. If a
device is lost, restart Brush and load the latest exported checkpoint. Actual
system sleep/wake still needs validation on the affected laptop.