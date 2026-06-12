# Release Notes

## Unreleased

### Highlights

#### Training performance

Most of the loss and backward path was rewritten over the cycle. SSIM moved to a dedicated fused kernel and was then folded together with L1 into a single forward pass ([#394](https://github.com/ArthurBrussee/brush/pull/394), [#401](https://github.com/ArthurBrussee/brush/pull/401)). SSIM's saved-for-backward tensors are now recomputed on the fly ([#409](https://github.com/ArthurBrussee/brush/pull/409)). The per-splat backward early-outs for splats that don't actually contribute to any pixel ([#394](https://github.com/ArthurBrussee/brush/pull/394)). Internal gradients and buffers are sparse where they used to be dense ([#378](https://github.com/ArthurBrussee/brush/pull/378)). Ground-truth images are packed to u32 RGBA on the GPU with background compositing and masking folded into the loss kernel, roughly 4x less pixel-side memory plus a chain of mixed-dtype Burn ops gone ([#410](https://github.com/ArthurBrussee/brush/pull/410)). The radix sort was rewritten to be ~50% faster, which also lifted rendering ~10-15% ([#386](https://github.com/ArthurBrussee/brush/pull/386)). Loading throughput and peak memory both improved meaningfully ([#280](https://github.com/ArthurBrussee/brush/pull/280), [#325](https://github.com/ArthurBrussee/brush/pull/325)).

#### Reconstruction quality & densification

The growth path got a real overhaul: longer-edge split, "uni" noise, and a fix for far-away splats being deleted too early ([#265](https://github.com/ArthurBrussee/brush/pull/265)). Force-split kicks in for splats that grow too big, then later gated by `growth_stop_iter` to avoid pathological end-of-training behaviour ([#390](https://github.com/ArthurBrussee/brush/pull/390), [#398](https://github.com/ArthurBrussee/brush/pull/398)). Random frustum init helps when there are no good initial points ([#372](https://github.com/ArthurBrussee/brush/pull/372)). NaN handling was tightened up with a new fuzz suite ([#389](https://github.com/ArthurBrussee/brush/pull/389)).

#### Appearance compensation (PPISP grid hybrid, bilateral grid & PPISP)

Training can now learn per-view photometric corrections so exposure / white-balance / vignetting variation between input images isn't baked into the splats. The recommended `--ppisp-grid` mode learns per-camera PPISP vignetting plus per-view bilateral grids of PPISP exposure + color parameters (spirulae-splat's hybrid); `--bilateral-grid` (BilaRF affine grids) and `--ppisp` (full per-frame PPISP) are available as comparison modes. All are fused CubeCL kernels with custom autodiff in the new `brush-appearance` crate; corrections apply to the rendered image before the loss and only exist at training time — exported splats keep canonical colors. The grids use a sparse per-view optimizer (per-step cost independent of dataset size) with optional gradient subsampling, and `--train-on-eval` lets evaluation apply the learned per-view corrections.

#### Mip-Splatting

The 2D mip filter from Mip-Splatting is now implemented, with a `render_mode` flag and a UI toggle. Contributed by @fhahlbohm ([#337](https://github.com/ArthurBrussee/brush/pull/337)).

#### LOD baking

After training, Brush can optionally generate N lower-detail levels by iteratively decimating splats and refining against downscaled images. Each level exports as its own `.ply` with a `_lodN` suffix. CLI/GUI knobs: `lod_levels`, `lod_refine_steps`, `lod_decimation_keep`, `lod_image_scale`. Contributed by @mvaligursky ([#365](https://github.com/ArthurBrussee/brush/pull/365)).

#### Embedding & web demos

Two new front doors for using Brush as a library: a C FFI (`brush-c`) for native embedding ([#308](https://github.com/ArthurBrussee/brush/pull/308)), and `brush-js`, a thin `wasm-bindgen` wrapper that lets any JS page drive Brush directly ([#402](https://github.com/ArthurBrussee/brush/pull/402)). Hosts get zero-copy access to the WebGPU buffers backing the transforms / SH coefficients / opacities, so they can bind them straight into their own render pipelines without ever round-tripping splat data through CPU. The web demos moved from Next.js to Vite + React in the process. The web viewer can also accept initial camera position and rotation via URL parameters ([#309](https://github.com/ArthurBrussee/brush/pull/309)).

#### UI overhaul

Several rounds of cleanup: tabbed sidebar, persistent layout, a status bar that hosts export / live-view / play-pause, notches in the progress bar marking export points, a settings window editable while training, full save/load of training configs, framing bars for reference poses, more background controls including random noise, and DPI-aware splat rendering. ([#299](https://github.com/ArthurBrussee/brush/pull/299), [#329](https://github.com/ArthurBrussee/brush/pull/329), [#330](https://github.com/ArthurBrussee/brush/pull/330), [#334](https://github.com/ArthurBrussee/brush/pull/334), [#335](https://github.com/ArthurBrussee/brush/pull/335), [#338](https://github.com/ArthurBrussee/brush/pull/338), [#346](https://github.com/ArthurBrussee/brush/pull/346), [#347](https://github.com/ArthurBrussee/brush/pull/347), [#362](https://github.com/ArthurBrussee/brush/pull/362), [#377](https://github.com/ArthurBrussee/brush/pull/377), [#399](https://github.com/ArthurBrussee/brush/pull/399)).

#### Masking & alpha

More mask folder layouts accepted (COLMAP's `masks/img.jpeg.png`, `img.mask.*`, masks at arbitrary folder depths), mask resizing, and a configurable `--alpha_mode` exposed in CLI and UI ([#298](https://github.com/ArthurBrussee/brush/pull/298), [#300](https://github.com/ArthurBrussee/brush/pull/300), [#301](https://github.com/ArthurBrussee/brush/pull/301)).

#### Web compatibility & robustness

WebGPU's workgroup dispatch limit is handled correctly, so resolutions above 2048 work ([#363](https://github.com/ArthurBrussee/brush/pull/363), from @promontis). A long-standing intersection-buffer corruption is fixed ([#373](https://github.com/ArthurBrussee/brush/pull/373)), as is a separate sort corruption past ~78M keys ([#385](https://github.com/ArthurBrussee/brush/pull/385)). Intersection and splat limits were raised so larger and higher-resolution scenes work ([#340](https://github.com/ArthurBrussee/brush/pull/340)).

#### Under the hood

All WGSL kernels have been ported to CubeCL `#[cube]` kernels, with shared math types (`Vec3A`, `Quat`, `Mat3`, `Sym2`) in a new `brush-cube` crate. The port surfaced and fixed a long-standing fuzz failure where FP drift between `project_forward` and `map_gaussians_to_intersect` left uninitialised slots in the intersection list ([#411](https://github.com/ArthurBrussee/brush/pull/411)).

### PRs of note (chronological)

- [#265](https://github.com/ArthurBrussee/brush/pull/265) - Densification overhaul: longer-edge split, "uni" noise, fix far-away splats being deleted too early
- [#280](https://github.com/ArthurBrussee/brush/pull/280) - Faster loading of big datasets
- [#298](https://github.com/ArthurBrussee/brush/pull/298) - Accept `img.mask.*` masks, add `--alpha_mode`
- [#299](https://github.com/ArthurBrussee/brush/pull/299) - DPI-aware splat rendering
- [#300](https://github.com/ArthurBrussee/brush/pull/300) - Alpha mode in the UI
- [#301](https://github.com/ArthurBrussee/brush/pull/301) - Fix mask discovery
- [#308](https://github.com/ArthurBrussee/brush/pull/308) - Initial Brush C FFI (@dalnoguer)
- [#309](https://github.com/ArthurBrussee/brush/pull/309) - Initial camera position/rotation via URL parameters (@dalnoguer)
- [#319](https://github.com/ArthurBrussee/brush/pull/319) - Use export directory if available
- [#325](https://github.com/ArthurBrussee/brush/pull/325) - Lower memory usage while loading
- [#326](https://github.com/ArthurBrussee/brush/pull/326) - Fix broken wasm-pack links (@lanxinger)
- [#327](https://github.com/ArthurBrussee/brush/pull/327) - 256-thread workgroup for `get_tile_offsets`
- [#329](https://github.com/ArthurBrussee/brush/pull/329) - UI refresh
- [#330](https://github.com/ArthurBrussee/brush/pull/330) - Move export / live view / play-pause into the status bar
- [#334](https://github.com/ArthurBrussee/brush/pull/334) - Export notches on progress bar, more UI updates
- [#335](https://github.com/ArthurBrussee/brush/pull/335) - Persistent UI layout, tab-bar improvements
- [#337](https://github.com/ArthurBrussee/brush/pull/337) - Mip-Splatting 2D mip filter (@fhahlbohm)
- [#338](https://github.com/ArthurBrussee/brush/pull/338) - Another UI revamp
- [#340](https://github.com/ArthurBrussee/brush/pull/340) - Raise intersection and splat limits, higher-resolution benchmarks
- [#346](https://github.com/ArthurBrussee/brush/pull/346) - Shared splat view across viewer + training UI
- [#347](https://github.com/ArthurBrussee/brush/pull/347) - Save/load of training configs
- [#350](https://github.com/ArthurBrussee/brush/pull/350) - Readback-based intersection count
- [#362](https://github.com/ArthurBrussee/brush/pull/362) - Settings window editable during training
- [#363](https://github.com/ArthurBrussee/brush/pull/363) - Support resolutions >2048 by chunking WebGPU dispatch (@promontis)
- [#365](https://github.com/ArthurBrussee/brush/pull/365) - LOD baking (@mvaligursky)
- [#372](https://github.com/ArthurBrussee/brush/pull/372) - Random frustum init
- [#373](https://github.com/ArthurBrussee/brush/pull/373) - Fix intersection-buffer corruption
- [#374](https://github.com/ArthurBrussee/brush/pull/374) - Better error checking for WASM export
- [#377](https://github.com/ArthurBrussee/brush/pull/377) - More background controls
- [#378](https://github.com/ArthurBrussee/brush/pull/378) - Sparse internal gradients / buffers
- [#381](https://github.com/ArthurBrussee/brush/pull/381) - Adam-mini style optimizer for SH
- [#385](https://github.com/ArthurBrussee/brush/pull/385) - Fix sort corruption past ~78M keys, add tests
- [#386](https://github.com/ArthurBrussee/brush/pull/386) - Sort rewrite: ~50% faster sorting, ~10-15% faster rendering
- [#389](https://github.com/ArthurBrussee/brush/pull/389) - Fuzz testing and NaN handling
- [#390](https://github.com/ArthurBrussee/brush/pull/390) - Force-split oversized splats
- [#392](https://github.com/ArthurBrussee/brush/pull/392) - Workaround SH0 crash
- [#394](https://github.com/ArthurBrussee/brush/pull/394) - Fused SSIM kernel, per-splat backward early-out
- [#398](https://github.com/ArthurBrussee/brush/pull/398) - Gate force-split by `growth_stop_iter`
- [#399](https://github.com/ArthurBrussee/brush/pull/399) - Reference-pose framing bars
- [#401](https://github.com/ArthurBrussee/brush/pull/401) - Per-splat backward + fused L1+SSIM loss
- [#402](https://github.com/ArthurBrussee/brush/pull/402) - `brush-js` library + Vite-based web demos
- [#409](https://github.com/ArthurBrussee/brush/pull/409) - Recompute SSIM partials in backward, drop saved tensors
- [#410](https://github.com/ArthurBrussee/brush/pull/410) - Pack GT to u32 RGBA, fold bg-composite + mask into loss kernel
- [#411](https://github.com/ArthurBrussee/brush/pull/411) - Port WGSL kernels to CubeCL

## 0.3

Brush 0.3

Brush 0.3 is a massive update to bring high quality splats to all platforms, while training faster, and bringing a ton of new features!

Brush now trains using the "MCMC" splatting technique, but, with its own variation that still grows splats automatically. This keeps the best of both worlds: splats grow first where they are needed, yet explore the scene like in MCMC, to improve quality. This [table](https://github.com/ArthurBrussee/brush/pull/121) has some preliminary results. You can set a limit of the maximum number of splats like in the original MCMC. Training works especially better on large scenes where not all views are visible from all angles. Training now also supports massive datasets bigger than RAM and starts instantly.

The web version also gains a lot of new features, with fullscreen modes, efficient file loading, directory loading, bundler integration, NPM compatibility, and faster training. Training on the web is not nearly at feature parity with the desktop version.

### Highlights:

**Training**

- "MCMC like" training. Higher quality and more robust. Still grows splats automatically like previous methods, while also allowing a maximum nr. of splats cap. For a more detailed write up, see [this PR](https://github.com/ArthurBrussee/brush/pull/121)

- Train on datasets bigger than RAM. Only up to some amount of gigs are cached, other files are loaded by the dataloader while training. [[1]](https://github.com/ArthurBrussee/brush/commit/8f1a09d2e8a1aef8a2fd0fc78e11e05dee234645)

- Start training faster [[1]](https://github.com/ArthurBrussee/brush/pull/255)

- Training bounds are now based on the splat bounds instead of the camera bounds [[1]](https://github.com/ArthurBrussee/brush/commit/85aa3a770caba800e886ac7a8ca2dd74e9ec9426) [[2]](https://github.com/ArthurBrussee/brush/commit/3efd3043ec6eb2d566a5c088590573025e9034d5)

- Improve backwards speed with thanks to @fhahlbohm [[1]](https://github.com/ArthurBrussee/brush/commit/8d5f7a10ad295a958c3068fa6dfd2a4ad1662d00) [[2]](https://github.com/ArthurBrussee/brush/commit/80b3434b7dce20bccc9fcd2d3b9c563ee219ba8d) [[3]](https://github.com/ArthurBrussee/brush/commit/ae532c30c4f02bd42c761c6de292fe415429ed43) [[4]](https://github.com/ArthurBrussee/brush/commit/589d4ca83e333bb2ed83e87febce187e2d36e40f) [[5]](https://github.com/ArthurBrussee/brush/commit/589d4ca83e333bb2ed83e87febce187e2d36e40f) [[6]](https://github.com/ArthurBrussee/brush/commit/c13d41bca44f33034ac7a683b272f5cb895054f2) [[7]](https://github.com/ArthurBrussee/brush/commit/671911d8dd7194e8da216b8a7b08f356151a3335)

- Always use `init.ply` as the init for the training if it exists [[1]](https://github.com/ArthurBrussee/brush/commit/cc4503ba555bcbe8276ab5cb01fc855e4da45b16)

- Prefer colmap datasets over nerfstudio, fixes import if your dataset has some random json in it [[1]](https://github.com/ArthurBrussee/brush/commit/5ad3dd073da1f1dc38b2f5f261c87be13173cef5)

- Add LPIPS loss [[1]](https://github.com/ArthurBrussee/brush/commit/555be385d5018e4d609adbfb5a83bae97d97c4e8) [[2]](https://github.com/ArthurBrussee/brush/commit/97174d819c7c717fcb2f40fcc608a5c5cc3f05ee)

- Use a separable convolution for SSIM [[1]](https://github.com/ArthurBrussee/brush/commit/fdc9bde948b6fdbad459674e49e74a3a5981da80)

- Lots of other tweaks to the training dynamics, bug fixes, version bumps etc.

**UI**

- The UI has gone through some redesigns to be cleaner and easier to use

- Add a grid widget [[1]](https://github.com/ArthurBrussee/brush/pull/261)

- The arrow keys now rotate the model and move it up/down. Combined with the grid this is helpful to align the ground. [[1]](https://github.com/ArthurBrussee/brush/pull/261)

- Press 'F' to toggle fullscreen mode [[1]](https://github.com/ArthurBrussee/brush/commit/b278f91993ef8ce8f57ca41ce3c7b7b93e4ca57d)

- Add play/pause button when playing a splat sequence [[1]](https://github.com/ArthurBrussee/brush/commit/1292b12d3988d4167e0d111edff6e4aa67b0e0ce)

- Add a FOV slider [[1]](https://github.com/ArthurBrussee/brush/commit/2498afd796b752fecd1159777191e13a3dceeeac)

- Settings UI panel when loading a new dataset [[1]](https://github.com/ArthurBrussee/brush/commit/777e5870c546a52d84253ec1162b9f3d06050237)

- Hide console on windows [[1]](https://github.com/ArthurBrussee/brush/commit/986a17b8ad59c06645087ae23d1c982420664d65)

- Add background color picker [[1]](https://github.com/ArthurBrussee/brush/commit/44c4f61cbe9b093253c11d55e7d268e31f903fdf)

- Add a slider to scale splats [[1]](https://github.com/ArthurBrussee/brush/commit/358e6c808a7cba3fd94f2185525b2c3be1bb9bdd)

- Reduce atomic adds to improve the speed of the backward pass [[1]](https://github.com/ArthurBrussee/brush/commit/122c5ab8823e408423f28b9b4ffc3bb0ed597047)

- Improve accuracy of training steps/s thanks to @fhahlbohm [[1]](https://github.com/ArthurBrussee/brush/commit/173bd43b31339b06b28264db366bbdceffb44917)

**Import/export**

- Support SuperSplat compressed ply format [[1]](https://github.com/ArthurBrussee/brush/commit/1cf21593b5ba3964823720b588bb2e2e19822980)

- Support r/g/b as color names in ply files [[1]](https://github.com/ArthurBrussee/brush/commit/2b8254c8a874575c182f246402bc8867a68dcad1)

- Sort files properly in zip directories for sequence playback [[1]](https://github.com/ArthurBrussee/brush/commit/2df24cfa2a44945d5887bc02d2c5020bf1b0b3a4)

- Fixed file case sensitivity issues [[1]](https://github.com/ArthurBrussee/brush/commit/8f925899c2309826f41d3d8a0a08aa2a3a39a311)

- Allow double floats in plys [[1]](https://github.com/ArthurBrussee/brush/commit/cf4108984aa854f689d92a5eab2fd3b6ed96572b)

- Swap out the PLY importer/exporter for my own. Speeds up import about 5x [[1]](https://github.com/ArthurBrussee/serde_ply)

**Web**

- You can now pick directories on the web, not just individual files [[1]](https://github.com/ArthurBrussee/brush/commit/1358d3467be6c5d417b83f0f8eb8b6094f7f45ed)

- More efficient file reading on the web [[1]](https://github.com/ArthurBrussee/brush/commit/1358d3467be6c5d417b83f0f8eb8b6094f7f45ed)

- Improved interop with JavaScript, see the example for some of the available APIs. [[1]](https://github.com/ArthurBrussee/brush/commit/bf125dbd4a24e471ff0514790049245d1bee898a)

- The web parts of Brush now use WASM modules compatible with bundlers, eg. with the demo now using Next.JS [[1]](https://github.com/ArthurBrussee/brush/commit/6341cc90b5e88ee0829671091ff2deae1e94795c)

- Add a panel showing various warnings that might happen [[1]](https://github.com/ArthurBrussee/brush/commit/a9cb04da9471753c4457a40c8cbbd6c84711b3b4)

- Add touch controls for the viewer UI [[1]](https://github.com/ArthurBrussee/brush/commit/3597006adbae653e527e2ef0688116be0ed70571)

- Add dwarf debug info for the Web [[1]](https://github.com/ArthurBrussee/brush/commit/506c1f09a46996fb3ba762ee3b7d33174e73c346)

**Other**

- Add number of splats to CLI output [[1]](https://github.com/ArthurBrussee/brush/commit/6e9739c78b739ec5c489697234f4e595c239e2a7)
- Improve compile times. Clean builds are ~1.5 minutes on my macbook
- Lots of bug fixes & version bumps
- Add example docker file [[1]](https://github.com/ArthurBrussee/brush/commit/be3112f482cac9864645c377fdebfd2eeda922b6)

## 0.2

Brush 0.2 goes from a proof of concept to a tool ready for real world data! It still only implements the “basics” of Gaussian Splatting, but trains as fast as gsplat to a (slightly) higher quality than gsplat. It also comes with nicer workflows, a CLI, dynamic gaussian rendering, and lots of other new features.

The next release will focus on going beyond the basics of Gaussian Splatting, and implementing extensions that help to make Brush more robust, faster, and higher quality than other splatting alternatives. This might mean that the outputs are no longer 100% compatible with other splat viewers, so more work will also be done to make the Brush web viewer a great experience.

### Features

- Brush now measures higher PSNR/SSIM than gsplat on the mipnerf360 scenes. Of course, gsplat with some more tuned settings might reach these numbers as well, but this shows Brush is grown up now!
  - See the [results table](https://github.com/ArthurBrussee/brush?tab=readme-ov-file#results)

- Faster training overall by optimizing the kernels, fixing various slowdowns, and reducing memory use.

- Brush now has a CLI!
  - Simply run brush –help to get an overview. The basic usage is brush PATH –args.
  - Any command works with `--with-viewer` which opens the UI for easy debugging.

- Add flythrough controls supporting both orbiting, FPS controls, flythrough controls, and panning.
  - See the ‘controls’ popout in the scene view for a full overview.

- Load data from a URL. If possible the data will be streamed in, and the splat will update in real-time.
  -For a web version, just pass in ?url=

- On the web, pass in ?zen=true to enable ‘zen’ mode which makes the viewer fullscreen.

- Add support for viewing dynamic splats
  - Either loaded as a sequence of PLY files (in a folder or zip)
  - Or as a custom data format “ply with delta frames”
  - This was used for [Cat4D](https://cat-4d.github.io/) and for [Cap4D](https://felixtaubner.github.io/cap4d/)
  - Felix kindly shared [their script](https://github.com/felixtaubner/brush_avatar/) to export this data for reference.

- Open directories directly, instead of only zip files.
  - ZIP files are still supported for all operations - as this is important for the web version.

- Support transparent images.
  - Images with alpha channels will force the output splat to _match_ this transparency.
  - Alternatively, you can include a folder of ‘masks’. This will _ignore_ those parts of the image while training.

- More flexible COLMAP & nerfstudio dataset format
  - Support more of the various options, and differing file structures.
  - If your dataset has a single ply file, it will be used for the initial point cloud.

  - While training, the up-axis is rotated such that the ground is flat (thanks to @fhahlbohm)
    - Note: The exported ply will however still match your input data. I’m investigating how to best handle this in the future - either as an option to rotate the splat, or by writing metadata into the exported splat.

### Enhancements

- Add alpha_loss_weight arg to control how heavy to weigh the alpha loss
  - Nb: not applicable to masks mode
- Log memory usage to rerun while training
- Fix SH clamping values to 0 ([#76](https://github.com/ArthurBrussee/brush/pull/76) thanks to @fhahlbohm)
- Better logic to pick ‘nearest’ dataset view
- Better splat pruning logic
- Remove ESC to close
- The web version has SSIM enabled again
- Display more detailed error traces in the UI and CLI when something goes wrong
- Different method of emitting tile intersections ([#63](https://github.com/ArthurBrussee/brush/pull/63) for details)
  - Fixes some potential corruptions depending on your driver/shader compiler.
- Read up-axis from PLY file if it’s included
- Eval PSNR/SSIM now simulate a 8 bit roundtrip for fair comparison
- Add an option `--export-every` to export a ply file every so many steps
  - See `--export-path` and `--export-name` for the location of the ply
- Add an option `--eval-save-to-disk` to save eval images to disk
  - See `–export-path` for
- Add notes in CLI & UI about running in debug mode (advising to compile with `--release`).
- Relax camera constraints, allow further zoom in/out
- Relax constraints on fields in the UI - now can enter values outside of slider range.
- Improvements to the UI, less unnecessary padding.

### Highlighted Fixes

- Dataset and scene view now match exactly 1:1
- Fix UI sometimes not updating when starting a new training run.
- Sort eval images to be consistent with the MipNeRF eval images
- Fix a crash from the KNN initialization

### Demo (Chrome only currently)

[Reference Garden scene (650MB)](https://arthurbrussee.github.io/brush-demo/?url=https://f005.backblazeb2.com/file/brush-splats-bakfiets/garden.ply&focal=1.0&zen=true)

[Mushroom I captured on a walk - only 50 images or so, a bit blurry!](https://arthurbrussee.github.io/brush-demo/?url=https://f005.backblazeb2.com/file/brush-splats-bakfiets/mushroom_centered.ply&zen=true&focal=1.5)

### Thanks

Thanks to everybody in the Brush discord, in particular @fasteinke for reporting many breakages along the way, @fhahlbohm for contributions and helping me fix my results table, @Simon.Bethke and @Gradeeterna for test data, @felixtaubner for the 4D splat export script.

## 0.0.1

- Add ability to train with transparent images
- Add option to select GPU with the CUBECL_DEFAULT_DEVICE environment variable
- Add 2D image trainer example
- Tweak splitting cloning logic, adds +- 0.5 PSNR
- Fixed backwards gradient for quaternions & SH
- Fix exporting ply files
- Fix evaluation not running
- Fix some NaNs on the web version
