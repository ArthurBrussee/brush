use crate::{
    INTERSECTS_UPPER_BOUND, MainBackendBase, SplatForward,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::SplatRenderMode,
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    render_aux::RenderAux,
    shaders::{self, MapGaussiansToIntersect, ProjectSplats, ProjectVisible, Rasterize},
};
use brush_kernel::{CubeCount, calc_cube_count_1d, create_dispatch_buffer_1d, create_tensor};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::tensor::{DType, IntDType, ops::FloatTensor};
use burn::tensor::{
    FloatDType,
    ops::{FloatTensorOps, IntTensorOps},
};
use burn_cubecl::cubecl::client::ComputeClient;
use burn_cubecl::cubecl::server::Bindings;

use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::CubeDim;
use burn_wgpu::WgpuRuntime;
use glam::{Vec3, uvec2};

/// Number of tiles per side in a chunk.
pub const TILES_PER_SIDE: u32 = 64;

/// Information about a rendering chunk
#[derive(Debug, Clone, Copy)]
pub struct ChunkInfo {
    pub offset: glam::UVec2,
    pub size: glam::UVec2,
    pub tile_bounds: glam::UVec2,
}

/// Intersection data computed for a chunk, used for rasterization
pub struct ChunkRenderInfo {
    pub tile_offsets: IntTensor<MainBackendBase>,
    pub compact_gid_from_isect: IntTensor<MainBackendBase>,
    pub num_intersections: IntTensor<MainBackendBase>,
}

/// Iterate over chunks needed to render an image of the given size.
/// Each chunk is at most `MAX_CHUNK_SIZE` x `MAX_CHUNK_SIZE` pixels, aligned to tile boundaries.
pub fn iter_chunks(img_size: [u32; 2]) -> impl Iterator<Item = ChunkInfo> {
    let tile_width = shaders::helpers::TILE_WIDTH;
    let chunk_size_pixels = TILES_PER_SIDE * tile_width;

    let num_chunks_x = img_size[0].div_ceil(chunk_size_pixels);
    let num_chunks_y = img_size[1].div_ceil(chunk_size_pixels);

    (0..num_chunks_y).flat_map(move |chunk_y| {
        (0..num_chunks_x).map(move |chunk_x| {
            let offset = uvec2(chunk_x * chunk_size_pixels, chunk_y * chunk_size_pixels);
            let size = uvec2(
                (img_size[0] - offset.x).min(chunk_size_pixels),
                (img_size[1] - offset.y).min(chunk_size_pixels),
            );
            let tile_bounds = calc_tile_bounds(size);
            ChunkInfo {
                offset,
                size,
                tile_bounds,
            }
        })
    })
}

pub(crate) fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

// On wasm, we cannot do a sync readback at all.
// Instead, can just estimate a max number of intersects. All the kernels only handle the actual
// number of intersects, and spin up empty threads for the rest atm. In the future, could use indirect
// dispatch to avoid this.
// Estimating the max number of intersects can be a bad hack though... The worst case scenario is so massive
// that it's easy to run out of memory... How do we actually properly deal with this :/
pub fn max_intersections(img_size: glam::UVec2, num_splats: u32) -> u32 {
    // Divide screen into tiles.
    let tile_bounds = calc_tile_bounds(img_size);
    // Assume on average each splat is maximally covering half x half the screen,
    // and adjust for the variance such that we're fairly certain we have enough intersections.
    let num_tiles = tile_bounds[0] * tile_bounds[1];
    let max_possible = num_tiles.saturating_mul(num_splats);
    // clamp to max nr. of dispatches.
    max_possible.min(INTERSECTS_UPPER_BOUND)
}

use burn::tensor::ops::IntTensor;

pub fn set_chunk_uniforms(uniforms: &mut shaders::helpers::RenderUniforms, chunk: &ChunkInfo) {
    uniforms.chunk_offset = chunk.offset.into();
    uniforms.tile_bounds = chunk.tile_bounds.into();
}

/// Compute intersection buffers for a single chunk.
pub fn compute_chunk_intersections(
    client: &ComputeClient<WgpuRuntime>,
    device: &burn_wgpu::WgpuDevice,
    chunk: &ChunkInfo,
    uniforms: shaders::helpers::RenderUniforms,
    projected_splats: &FloatTensor<MainBackendBase>,
    num_visible: &IntTensor<MainBackendBase>,
    total_splats: usize,
) -> ChunkRenderInfo {
    let max_intersects = max_intersections(chunk.size, total_splats as u32);
    let num_tiles = chunk.tile_bounds.x * chunk.tile_bounds.y;

    let splat_intersect_counts =
        MainBackendBase::int_zeros([total_splats + 1].into(), device, IntDType::U32);

    let num_vis_map_wg = create_dispatch_buffer_1d(
        num_visible.clone(),
        MapGaussiansToIntersect::WORKGROUP_SIZE[0],
    );

    // First do a prepass to compute the tile counts, then fill in intersection counts.
    tracing::trace_span!("MapGaussiansToIntersectPrepass").in_scope(|| {
        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client
                .launch_unchecked(
                    MapGaussiansToIntersect::task(true),
                    CubeCount::Dynamic(num_vis_map_wg.handle.clone().binding()),
                    Bindings::new()
                        .with_buffers(vec![
                            num_visible.handle.clone().binding(),
                            projected_splats.handle.clone().binding(),
                            splat_intersect_counts.handle.clone().binding(),
                        ])
                        .with_metadata(uniforms.to_meta_binding()),
                )
                .expect("Failed to render splats");
        }
    });

    // TODO: Only need to do this up to num_visible gaussians really.
    let cum_tiles_hit =
        tracing::trace_span!("PrefixSumGaussHits").in_scope(|| prefix_sum(splat_intersect_counts));

    let tile_id_from_isect = create_tensor([max_intersects as usize], device, DType::U32);
    let compact_gid_from_isect = create_tensor([max_intersects as usize], device, DType::U32);

    // Zero this out, as the kernel _might_ not run at all if no gaussians are visible.
    let num_intersections = MainBackendBase::int_zeros([1].into(), device, IntDType::U32);

    tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
        // SAFETY: Kernel checked to have no OOB, bounded loops.
        unsafe {
            client
                .launch_unchecked(
                    MapGaussiansToIntersect::task(false),
                    CubeCount::Dynamic(num_vis_map_wg.handle.clone().binding()),
                    Bindings::new()
                        .with_buffers(vec![
                            num_visible.handle.clone().binding(),
                            projected_splats.handle.clone().binding(),
                            cum_tiles_hit.handle.binding(),
                            tile_id_from_isect.handle.clone().binding(),
                            compact_gid_from_isect.handle.clone().binding(),
                            num_intersections.handle.clone().binding(),
                        ])
                        .with_metadata(uniforms.to_meta_binding()),
                )
                .expect("Failed to render splats");
        }
    });

    // We're sorting by tile ID, but we know beforehand what the maximum value
    // can be. We don't need to sort all the leading 0 bits!
    let bits = u32::BITS - num_tiles.leading_zeros();

    let (tile_id_from_isect, compact_gid_from_isect) =
        tracing::trace_span!("Tile sort").in_scope(|| {
            radix_argsort(
                tile_id_from_isect,
                compact_gid_from_isect,
                &num_intersections,
                bits,
            )
        });

    let cube_dim = CubeDim::new_1d(256);
    let num_vis_map_wg =
        create_dispatch_buffer_1d(num_intersections.clone(), 256 * CHECKS_PER_ITER);
    let cube_count = CubeCount::Dynamic(num_vis_map_wg.handle.binding());

    // Tiles without splats will be written as having a range of [0, 0].
    let tile_offsets = MainBackendBase::int_zeros(
        [
            chunk.tile_bounds.y as usize,
            chunk.tile_bounds.x as usize,
            2,
        ]
        .into(),
        device,
        IntDType::U32,
    );

    // SAFETY: Safe kernel.
    unsafe {
        get_tile_offsets::launch_unchecked::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            tile_id_from_isect.as_tensor_arg(1),
            tile_offsets.as_tensor_arg(1),
            num_intersections.as_tensor_arg(1),
        )
        .expect("Failed to render splats");
    }

    ChunkRenderInfo {
        tile_offsets,
        compact_gid_from_isect,
        num_intersections,
    }
}

pub fn create_uniforms(
    camera: &Camera,
    img_size: glam::UVec2,
    sh_degree: u32,
    background: Vec3,
) -> shaders::helpers::RenderUniforms {
    shaders::helpers::RenderUniforms {
        viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        focal: camera.focal(img_size).into(),
        pixel_center: camera.center(img_size).into(),
        img_size: img_size.into(),
        tile_bounds: calc_tile_bounds(img_size).into(),
        sh_degree,
        // Will be updated per-chunk
        paddingA: 0,
        background: [background.x, background.y, background.z, 1.0],
        // Updated per-chunk
        chunk_offset: [0, 0],
    }
}

// Implement forward functions for the inner wgpu backend.
impl SplatForward<Self> for MainBackendBase {
    fn render_splats(
        uniforms: shaders::helpers::RenderUniforms,
        means: FloatTensor<Self>,
        log_scales: FloatTensor<Self>,
        quats: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        bwd_info: bool,
    ) -> (FloatTensor<Self>, RenderAux<Self>) {
        assert!(
            uniforms.img_size[0] > 0 && uniforms.img_size[1] > 0,
            "Can't render images with 0 size."
        );

        // Tensor params might not be contiguous, convert them to contiguous tensors.
        let means = into_contiguous(means);
        let log_scales = into_contiguous(log_scales);
        let quats = into_contiguous(quats);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opacities = into_contiguous(raw_opacities);

        let device = &means.device.clone();
        let _client = means.client.clone();

        let _span = tracing::trace_span!("render_forward").entered();

        // Check whether input dimensions are valid.
        DimCheck::new()
            .check_dims("means", &means, &["D".into(), 3.into()])
            .check_dims("log_scales", &log_scales, &["D".into(), 3.into()])
            .check_dims("quats", &quats, &["D".into(), 4.into()])
            .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
            .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

        // A note on some confusing naming that'll be used throughout this function:
        // Gaussians are stored in various states of buffers, eg. at the start they're all in one big buffer,
        // then we sparsely store some results, then sort gaussian based on depths, etc.
        // Overall this means there's lots of indices flying all over the place, and it's hard to keep track
        // what is indexing what. So, for some sanity, try to match a few "gaussian ids" (gid) variable names.
        // - Global Gaussian ID - global_gid
        // - Compacted Gaussian ID - compact_gid
        // - Per tile intersection depth sorted ID - tiled_gid
        // - Sorted by tile per tile intersection depth sorted ID - sorted_tiled_gid
        // Then, various buffers map between these, which are named x_from_y_gid, eg.
        //  global_from_compact_gid.
        let total_splats = means.shape.dims[0];

        // Separate buffer for num_visible (written atomically by ProjectSplats)
        let num_visible = Self::int_zeros([1].into(), device, IntDType::U32);

        let client = &means.client.clone();

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        // === GLOBAL PHASE: Project all splats, sort by depth ===
        let global_from_compact_gid = {
            let global_from_presort_gid =
                Self::int_zeros([total_splats].into(), device, IntDType::U32);
            let depths = create_tensor([total_splats], device, DType::F32);

            tracing::trace_span!("ProjectSplats").in_scope(||
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
            client.launch_unchecked(
                ProjectSplats::task(mip_splat),
                calc_cube_count_1d(total_splats as u32, ProjectSplats::WORKGROUP_SIZE[0]),
                Bindings::new().with_buffers(
                vec![
                    means.handle.clone().binding(),
                    quats.handle.clone().binding(),
                    log_scales.handle.clone().binding(),
                    raw_opacities.handle.clone().binding(),
                    global_from_presort_gid.handle.clone().binding(),
                    depths.handle.clone().binding(),
                    num_visible.handle.clone().binding(),
                ]).with_metadata(uniforms.to_meta_binding()),
            ).expect("Failed to render splats");
            });

            let (_, global_from_compact_gid) = tracing::trace_span!("DepthSort").in_scope(|| {
                // Interpret the depth as a u32. This is fine for a radix sort, as long as the depth > 0.0,
                // which we know to be the case given how we cull splats.
                radix_argsort(depths, global_from_presort_gid, &num_visible, 32)
            });

            global_from_compact_gid
        };

        // Create a buffer of 'projected' splats, that is,
        // project XY, projected conic, and converted color.
        let proj_size = size_of::<shaders::helpers::ProjectedSplat>() / size_of::<f32>();
        let projected_splats = create_tensor([total_splats, proj_size], device, DType::F32);

        tracing::trace_span!("ProjectVisible").in_scope(|| {
            // Create a buffer to determine how many threads to dispatch for all visible splats.
            let num_vis_wg =
                create_dispatch_buffer_1d(num_visible.clone(), ProjectVisible::WORKGROUP_SIZE[0]);
            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        ProjectVisible::task(mip_splat),
                        CubeCount::Dynamic(num_vis_wg.handle.binding()),
                        Bindings::new()
                            .with_buffers(vec![
                                num_visible.handle.clone().binding(),
                                means.handle.binding(),
                                log_scales.handle.binding(),
                                quats.handle.binding(),
                                sh_coeffs.handle.binding(),
                                raw_opacities.handle.binding(),
                                global_from_compact_gid.handle.clone().binding(),
                                projected_splats.handle.clone().binding(),
                            ])
                            .with_metadata(uniforms.to_meta_binding()),
                    )
                    .expect("Failed to render splats");
            }
        });

        // === ALLOCATE OUTPUT BUFFERS ===
        let out_dim = if bwd_info {
            4
        } else {
            // Channels are packed into 4 bytes, aka one float.
            1
        };

        let out_img = create_tensor(
            [
                uniforms.img_size[1] as usize,
                uniforms.img_size[0] as usize,
                out_dim,
            ],
            device,
            DType::F32,
        );

        let visible = if bwd_info {
            Self::float_zeros([total_splats].into(), device, FloatDType::F32)
        } else {
            create_tensor([1], device, DType::F32)
        };

        // Compile the kernel, including/excluding info for backwards pass.
        // see the BWD_INFO define in the rasterize shader.
        let raster_task = Rasterize::task(bwd_info, cfg!(target_family = "wasm"));

        // === PER-CHUNK PHASE: Intersection mapping and rasterization ===
        let chunks: Vec<_> = iter_chunks(uniforms.img_size).collect();

        // Use mutable uniforms that we update per-chunk
        let mut uniforms = uniforms;

        for chunk in &chunks {
            let _chunk_span = tracing::trace_span!("RenderChunk").entered();

            // Update uniforms for this chunk
            set_chunk_uniforms(&mut uniforms, chunk);

            // Compute intersection buffers for this chunk
            let chunk_render_info = compute_chunk_intersections(
                client,
                device,
                chunk,
                uniforms,
                &projected_splats,
                &num_visible,
                total_splats,
            );

            // Rasterize this chunk
            let _raster_span = tracing::trace_span!("Rasterize").entered();

            let mut bindings = Bindings::new().with_buffers(vec![
                chunk_render_info
                    .compact_gid_from_isect
                    .handle
                    .clone()
                    .binding(),
                chunk_render_info.tile_offsets.handle.clone().binding(),
                projected_splats.handle.clone().binding(),
                out_img.handle.clone().binding(),
            ]);

            if bwd_info {
                bindings = bindings.with_buffers(vec![
                    global_from_compact_gid.handle.clone().binding(),
                    visible.handle.clone().binding(),
                ]);
            }

            bindings = bindings.with_metadata(uniforms.to_meta_binding());

            // SAFETY: Kernel checked to have no OOB, bounded loops.
            unsafe {
                client
                    .launch_unchecked(
                        raster_task.clone(),
                        CubeCount::Static(chunk.tile_bounds.x * chunk.tile_bounds.y, 1, 1),
                        bindings,
                    )
                    .expect("Failed to render splats");
            }
        }

        // Sanity check the buffers.
        assert!(
            global_from_compact_gid.is_contiguous(),
            "Global from compact gid must be contiguous"
        );
        assert!(visible.is_contiguous(), "Visible must be contiguous");
        assert!(
            projected_splats.is_contiguous(),
            "Projected splats must be contiguous"
        );

        (
            out_img,
            RenderAux {
                uniforms,
                projected_splats,
                global_from_compact_gid,
                num_visible,
                visible,
            },
        )
    }
}
