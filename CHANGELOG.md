# Release Notes

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
