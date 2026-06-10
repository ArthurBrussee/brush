//! Top-level mesh-extraction driver. Glues tetra-point sampling, CPU
//! Delaunay, per-view GPU opacity integration, marching tets, and binary
//! search refinement.

use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_render::SplatOps;
use brush_render::burn_glue::resolve_to_cube_float;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RenderOptions, SplatRenderMode, Splats};
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::tensor::s;
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::WgpuRuntime;
use glam::{UVec2, Vec3};

use crate::Mesh;
use crate::filter::{filter_mesh_with_keep, filter_small_components};
use crate::marching_tet::marching_tets;
use crate::refine::{N_STEPS, RefineState, read_back_f32};
use crate::tetra_points::{TetraPointsConfig, build_tetra_points};

/// User-facing extraction config.
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    pub tetra_points: TetraPointsConfig,
    /// Iso-value for the level set. Carves the surface where transmittance
    /// has dropped to this fraction (GOF uses 0.5; 0.4 works better on
    /// brush-trained splats).
    pub iso_value: f32,
    /// Taubin smoothing iterations (λ|μ pairs) applied to the final mesh.
    /// Non-shrinking low-pass on the marching-tets noise; 0 disables.
    pub smooth_iters: u32,
    /// Drop connected components with fewer faces than this (speckle blobs
    /// from isolated iso-crossings). 0/1 disables.
    pub min_component_faces: usize,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            tetra_points: TetraPointsConfig::default(),
            iso_value: 0.4,
            smooth_iters: 10,
            min_component_faces: 100,
        }
    }
}

/// Extract a triangle mesh from `splats`. `views` carries the per-view
/// `(camera, image_size)` pairs used for both frustum-culling the seed
/// points and integrating the opacity along rays.
pub async fn extract_mesh(splats: Splats, views: &[(Camera, UVec2)], cfg: &ExtractConfig) -> Mesh {
    // Bake the min_scale floor once so the seed sampler and integrate
    // kernels see the same effective splats as the renderer.
    let splats = splats.bake_min_scale();

    let n_splats = splats.num_splats() as usize;
    log::info!(
        "Extracting mesh from {n_splats} splats across {} views",
        views.len()
    );

    // Per-phase profile; each phase ends with a sync so queued GPU work
    // gets billed to the phase that launched it.
    let sync_client = resolve_to_cube_float(splats.transforms.val())
        .client
        .clone();
    let mut phases: Vec<(&'static str, std::time::Duration)> = Vec::new();
    let mut phase_start = std::time::Instant::now();

    // Pull splat tensors back to host for seed-point sampling.
    let means_t = splats.transforms.val().slice(s![.., 0..3]);
    let quats_t = splats.transforms.val().slice(s![.., 3..7]);
    let log_scales_t = splats.transforms.val().slice(s![.., 7..10]);
    let means: Vec<f32> = means_t
        .into_data_async()
        .await
        .expect("read means")
        .into_vec::<f32>()
        .expect("means f32");
    let quats: Vec<f32> = quats_t
        .into_data_async()
        .await
        .expect("read quats")
        .into_vec::<f32>()
        .expect("quats f32");
    let log_scales: Vec<f32> = log_scales_t
        .into_data_async()
        .await
        .expect("read scales")
        .into_vec::<f32>()
        .expect("scales f32");

    sync_client.sync().await.expect("sync");
    phases.push(("load_tensors", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    let cams: Vec<Camera> = views.iter().map(|(c, _)| *c).collect();
    let img_sizes: Vec<UVec2> = views.iter().map(|(_, s)| *s).collect();

    let splat_bbox = bbox(means.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])));
    let pts = build_tetra_points(
        &means,
        &quats,
        &log_scales,
        &cams,
        &img_sizes,
        &cfg.tetra_points,
    );
    log::info!("Seed points (frustum-culled): {}", pts.points.len());
    if pts.points.len() < 4 {
        log::warn!("Too few seed points to triangulate; returning empty mesh");
        return Mesh::default();
    }

    phases.push(("build_seeds", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    // CPU Delaunay and the per-view pre-render are independent and both
    // expensive; run them concurrently. The render cache then feeds the
    // initial alpha eval, every binary-search iter, and the colour eval.
    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };
    let pts_for_delaunay = pts.points.clone();
    let delaunay_handle = tokio::task::spawn_blocking(move || {
        let span = tracing::trace_span!("delaunay_3d").entered();
        let tets = crate::delaunay::delaunay_3d(&pts_for_delaunay);
        drop(span);
        tets
    });
    let splats_ref = &splats;
    let pts_ref = &pts.points;
    let gpu_block = async move {
        let cache = pre_render_views(splats_ref, views, render_mode).await;
        let alpha = evaluate_alpha(splats_ref, pts_ref, &cache).await;
        (cache, alpha)
    };
    let ((view_cache, alpha), tets_result) = tokio::join!(gpu_block, delaunay_handle);
    let tets = tets_result.expect("delaunay task panicked");
    log::info!("Delaunay tets: {}", tets.len());
    sync_client.sync().await.expect("sync");
    phases.push(("delaunay_and_initial_alpha", phase_start.elapsed()));
    phase_start = std::time::Instant::now();
    let sdf: Vec<f32> = alpha.iter().map(|a| a - cfg.iso_value).collect();

    let mt = marching_tets(&tets, &sdf);
    log::info!(
        "Crossings: {}, faces: {}",
        mt.crossings.len(),
        mt.faces.len()
    );
    if mt.crossings.is_empty() {
        log::warn!("No iso-surface crossings; nothing to refine. Returning empty mesh.");
        return Mesh::default();
    }

    phases.push(("marching_tets", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    // Binary-search refinement with GPU-resident bracket state; the only
    // per-step CPU traffic is a 4-byte readback of the active count.
    let device = resolve_to_cube_float(splats.transforms.val()).device;
    let mut state = RefineState::new(&mt.crossings, &pts.points, &sdf, device);
    for step in 0..N_STEPS {
        if state.n_active() == 0 {
            break;
        }
        let mid_pos_t = state.compute_midpoints_t();
        let (min_alpha_t, _, _) =
            integrate_alpha(&splats, &mid_pos_t, state.n_active(), &view_cache, false);
        state.update_bracket_t(min_alpha_t, cfg.iso_value).await;
        sync_client.sync().await.expect("sync");
        log::info!(
            "Binary search step {}/{} done ({} crossings still active)",
            step + 1,
            N_STEPS,
            state.n_active(),
        );
    }
    phases.push(("binary_search", phase_start.elapsed()));
    phase_start = std::time::Instant::now();
    let refined = state.finish().await;

    // GOF's filter_mesh: keep a crossing iff its original Delaunay edge is
    // shorter than the two endpoint Gaussian scales combined (refined
    // positions sit between the endpoints, so test the original edge).
    let keep_crossing: Vec<bool> = mt
        .crossings
        .iter()
        .map(|c| {
            let pa = pts.points[c.a as usize];
            let pb = pts.points[c.b as usize];
            let d = (pa - pb).length();
            let scale_sum = pts.scales[c.a as usize] + pts.scales[c.b as usize];
            d <= scale_sum
        })
        .collect();

    // Vertex colours: visibility-weighted RGB blend evaluated at the
    // refined (on-surface) positions.
    let colors = evaluate_colors(&splats, &refined, &view_cache).await;
    sync_client.sync().await.expect("sync");
    phases.push(("final_color_eval", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    let vertex_colors: Vec<[u8; 3]> = colors
        .iter()
        .map(|c| {
            [
                (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                (c[2].clamp(0.0, 1.0) * 255.0) as u8,
            ]
        })
        .collect();

    let mut mesh = Mesh {
        vertices: refined,
        vertex_colors,
        faces: mt.faces,
    };

    mesh = filter_mesh_with_keep(&mesh, &keep_crossing);

    // Crop outliers from edges spanning the empty far field (huge billboard
    // splats defeat the edge-length filter since their scales are huge too).
    let margin = 0.1 * (splat_bbox.1 - splat_bbox.0).length();
    let crop_lo = splat_bbox.0 - Vec3::splat(margin);
    let crop_hi = splat_bbox.1 + Vec3::splat(margin);
    let inside: Vec<bool> = mesh
        .vertices
        .iter()
        .map(|v| {
            v.x >= crop_lo.x
                && v.x <= crop_hi.x
                && v.y >= crop_lo.y
                && v.y <= crop_hi.y
                && v.z >= crop_lo.z
                && v.z <= crop_hi.z
        })
        .collect();
    let n_dropped = inside.iter().filter(|&&k| !k).count();
    if n_dropped > 0 {
        log::info!(
            "Cropping {n_dropped} verts outside splat-bbox+10% margin ({:.2} of {})",
            n_dropped as f32 / inside.len() as f32 * 100.0,
            inside.len()
        );
        mesh = filter_mesh_with_keep(&mesh, &inside);
    }

    mesh = filter_small_components(&mesh, cfg.min_component_faces);

    if cfg.smooth_iters > 0 {
        let t_smooth = std::time::Instant::now();
        crate::smooth::taubin_smooth(&mut mesh, cfg.smooth_iters);
        log::info!(
            "Taubin smoothing: {} iters in {:.2}s",
            cfg.smooth_iters,
            t_smooth.elapsed().as_secs_f64()
        );
    }

    let mesh_bbox = bbox(mesh.vertices.iter().copied());
    log::info!(
        "Final mesh: {} verts, {} faces, bbox extent ({:.2}x{:.2}x{:.2})",
        mesh.vertices.len(),
        mesh.faces.len(),
        mesh_bbox.1.x - mesh_bbox.0.x,
        mesh_bbox.1.y - mesh_bbox.0.y,
        mesh_bbox.1.z - mesh_bbox.0.z,
    );

    phases.push(("filter_and_crop", phase_start.elapsed()));
    let total: std::time::Duration = phases.iter().map(|(_, d)| *d).sum();
    log::info!("=== EXTRACT PROFILE ===");
    for (name, d) in &phases {
        let pct = if total.is_zero() {
            0.0
        } else {
            d.as_secs_f64() / total.as_secs_f64() * 100.0
        };
        log::info!("  {:>28}: {:>7.2}s ({:>4.1}%)", name, d.as_secs_f64(), pct);
    }
    log::info!("  {:>28}: {:>7.2}s", "TOTAL", total.as_secs_f64());

    mesh
}

fn bbox(points: impl Iterator<Item = Vec3>) -> (Vec3, Vec3) {
    points.fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(mn, mx), p| (mn.min(p), mx.max(p)),
    )
}

/// Cached per-view forward render, shared by the initial alpha eval, the
/// binary-search iters, and the colour eval (the render dominates per-view
/// cost and is identical across them).
pub struct ViewRender {
    pub render: brush_render::RenderOutput<MainBackendBase>,
    pub camera_model: brush_render::kernels::camera_model::CameraModel,
}

/// Render every training view once and stash the output. Caller hands
/// the resulting slice to each `evaluate_alpha` call below.
async fn pre_render_views(
    splats: &Splats,
    views: &[(Camera, UVec2)],
    render_mode: SplatRenderMode,
) -> Vec<ViewRender> {
    type B = MainBackendBase;
    let transforms_p = resolve_to_cube_float(splats.transforms.val());
    let sh_coeffs_p = resolve_to_cube_float(splats.sh_coeffs.val());
    let raw_opacities_p = resolve_to_cube_float(splats.raw_opacities.val());
    let mut cache = Vec::with_capacity(views.len());
    for (view_idx, (cam, sz)) in views.iter().enumerate() {
        let out = <B as SplatOps<B>>::render(
            cam,
            *sz,
            transforms_p.clone(),
            sh_coeffs_p.clone(),
            raw_opacities_p.clone(),
            RenderOptions::color().with_render_mode(render_mode),
        )
        .await;
        cache.push(ViewRender {
            render: out,
            camera_model: cam.camera_model,
        });
        if (view_idx + 1).is_multiple_of(32) || view_idx + 1 == views.len() {
            log::info!("pre-rendered view {}/{}", view_idx + 1, views.len());
        }
    }
    cache
}

/// Per-view tile-cooperative integrate over all views: project vertices to
/// tiles, histogram + prefix-sum + scatter into per-tile slices, then one
/// workgroup per tile streams gaussians through shared memory. Returns the
/// `(min_alpha, color_sum, weight_sum)` running aggregators; the colour
/// tensors are 1-element dummies when `track_color = false`.
fn integrate_alpha(
    splats: &Splats,
    pts_tensor: &FloatTensor<MainBackendBase>,
    n: usize,
    view_renders: &[ViewRender],
    track_color: bool,
) -> (
    FloatTensor<MainBackendBase>,
    FloatTensor<MainBackendBase>,
    FloatTensor<MainBackendBase>,
) {
    type B = MainBackendBase;
    use brush_cube::{create_tensor, create_tensor_from_slice};
    use brush_render::kernels::integrate;
    use burn::tensor::DType;

    let transforms_p = resolve_to_cube_float(splats.transforms.val());
    let raw_opacities_p = resolve_to_cube_float(splats.raw_opacities.val());
    let device = transforms_p.device.clone();
    let client = transforms_p.client.clone();

    // Aggregators: live across all per-view kernels.
    let min_alpha_t: FloatTensor<B> = B::float_from_data(
        burn::tensor::TensorData::new(vec![f32::INFINITY; n], [n]),
        &device,
    );
    let (nc, nw) = if track_color { (n, n) } else { (1, 1) };
    let color_sum_t: FloatTensor<B> =
        B::float_from_data(burn::tensor::TensorData::zeros::<f32, _>([nc, 3]), &device);
    let weight_sum_t: FloatTensor<B> =
        B::float_from_data(burn::tensor::TensorData::zeros::<f32, _>([nw]), &device);

    // Per-view scratch, overwritten each view; only the aggregators carry.
    let tile_ids_t = create_tensor::<1>([n], &device, DType::U32);
    let depths_t = create_tensor::<1>([n], &device, DType::F32);
    let ray_dir_xy_t = create_tensor::<2>([n, 2], &device, DType::F32);
    let pix_x_t = create_tensor::<1>([n], &device, DType::U32);
    let pix_y_t = create_tensor::<1>([n], &device, DType::U32);
    let sorted_indices_t = create_tensor::<1>([n], &device, DType::U32);

    for view_render in view_renders {
        let out = view_render.render.clone();
        let camera_model = view_render.camera_model;
        let tile_bw = out.project_uniforms.tile_bounds[0];
        let tile_bh = out.project_uniforms.tile_bounds[1];
        let n_tiles = (tile_bw * tile_bh) as usize;

        integrate::project_vertices_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            pts_tensor.clone().into_tensor_arg(),
            tile_ids_t.clone().into_tensor_arg(),
            depths_t.clone().into_tensor_arg(),
            ray_dir_xy_t.clone().into_tensor_arg(),
            pix_x_t.clone().into_tensor_arg(),
            pix_y_t.clone().into_tensor_arg(),
            n as u32,
            out.project_uniforms.to_launch_object(),
            camera_model,
        );

        let counts_t = create_tensor_from_slice(&vec![0u32; n_tiles + 1], &device, DType::U32);
        integrate::histogram_tile_ids_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            tile_ids_t.clone().into_tensor_arg(),
            counts_t.clone().into_tensor_arg(),
            n as u32,
            tile_bw * tile_bh,
        );
        let vertex_tile_offsets_t = brush_prefix_sum::prefix_sum(counts_t);

        let write_counters_t = create_tensor_from_slice(&vec![0u32; n_tiles], &device, DType::U32);
        integrate::scatter_vertices_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            tile_ids_t.clone().into_tensor_arg(),
            vertex_tile_offsets_t.clone().into_tensor_arg(),
            write_counters_t.clone().into_tensor_arg(),
            sorted_indices_t.clone().into_tensor_arg(),
            n as u32,
            tile_bw * tile_bh,
        );

        // One workgroup per tile (CUBE_POS = tile id, like rasterize).
        let num_tiles_u32 = tile_bw * tile_bh;
        integrate::integrate_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_tiles_u32 * integrate::TILE_SIZE, integrate::TILE_SIZE),
            CubeDim::new_1d(integrate::TILE_SIZE),
            transforms_p.clone().into_tensor_arg(),
            raw_opacities_p.clone().into_tensor_arg(),
            out.compact_gid_from_isect.into_tensor_arg(),
            out.aux.tile_offsets.into_tensor_arg(),
            out.global_from_compact_gid.into_tensor_arg(),
            out.out_img.into_tensor_arg(),
            sorted_indices_t.clone().into_tensor_arg(),
            vertex_tile_offsets_t.into_tensor_arg(),
            depths_t.clone().into_tensor_arg(),
            ray_dir_xy_t.clone().into_tensor_arg(),
            pix_x_t.clone().into_tensor_arg(),
            pix_y_t.clone().into_tensor_arg(),
            min_alpha_t.clone().into_tensor_arg(),
            color_sum_t.clone().into_tensor_arg(),
            weight_sum_t.clone().into_tensor_arg(),
            out.project_uniforms.to_launch_object(),
            track_color,
        );
    }

    (min_alpha_t, color_sum_t, weight_sum_t)
}

fn points_tensor(points: &[Vec3], splats: &Splats) -> FloatTensor<MainBackendBase> {
    let device = resolve_to_cube_float(splats.transforms.val()).device;
    let pts_flat: Vec<f32> = points.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    MainBackendBase::float_from_data(
        burn::tensor::TensorData::new(pts_flat, [points.len(), 3]),
        &device,
    )
}

/// Per-point alpha over all views (GOF carving rule): `alpha(p) =
/// max_views(T_view(p))`, so a point is open as soon as any view sees
/// through to it and solid only when every view is blocked. Never-seen
/// points (no view in frustum) default to open space (alpha = 1).
async fn evaluate_alpha(splats: &Splats, points: &[Vec3], view_renders: &[ViewRender]) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }
    let pts_tensor = points_tensor(points, splats);
    let (min_alpha_t, _, _) =
        integrate_alpha(splats, &pts_tensor, points.len(), view_renders, false);
    read_back_f32(min_alpha_t)
        .await
        .iter()
        .map(|&a| if a.is_finite() { 1.0 - a } else { 1.0 })
        .collect()
}

/// Per-point visibility-weighted RGB blend over all views, linear `[0, 1]`.
/// Zero-weight vertices fall back to white (GOF's init).
async fn evaluate_colors(
    splats: &Splats,
    points: &[Vec3],
    view_renders: &[ViewRender],
) -> Vec<[f32; 3]> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let pts_tensor = points_tensor(points, splats);
    let (_, color_sum_t, weight_sum_t) =
        integrate_alpha(splats, &pts_tensor, n, view_renders, true);
    let color_sum = read_back_f32(color_sum_t).await;
    let weight_sum = read_back_f32(weight_sum_t).await;
    (0..n)
        .map(|i| {
            let w = weight_sum[i];
            if w > 1.0e-6 {
                [
                    color_sum[3 * i] / w,
                    color_sum[3 * i + 1] / w,
                    color_sum[3 * i + 2] / w,
                ]
            } else {
                [1.0, 1.0, 1.0]
            }
        })
        .collect()
}
