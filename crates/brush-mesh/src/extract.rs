//! Top-level mesh-extraction driver. Glues tetra-point sampling, CPU
//! Delaunay, per-view GPU opacity integration, marching tets, and binary
//! search refinement.

use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_render::SplatOps;
use brush_render::burn_glue::resolve_to_cube_float;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterPass, SplatRenderMode, Splats};
use brush_render::kernels;
use burn::backend::ops::{FloatTensorOps, TransactionOps, TransactionPrimitive};
use burn::backend::tensor::FloatTensor;
use burn::tensor::s;
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::WgpuRuntime;
use glam::{UVec2, Vec3};

use crate::Mesh;
use crate::delaunay::delaunay_3d;
use crate::filter::filter_mesh_with_keep;
use crate::marching_tet::marching_tets;
use crate::refine::{N_STEPS, RefineState};
use crate::tetra_points::{TetraPointsConfig, build_tetra_points};

/// User-facing extraction config.
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    pub tetra_points: TetraPointsConfig,
    /// Iso-value for the level set. GOF default 0.5 — carves the
    /// surface at half-transmittance, the principled "material
    /// boundary" choice.
    pub iso_value: f32,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            tetra_points: TetraPointsConfig::default(),
            iso_value: 0.5,
        }
    }
}

/// Extract a triangle mesh from `splats`. `views` carries the per-view
/// `(camera, image_size)` pairs used for both frustum-culling the seed
/// points and integrating the opacity along rays.
pub async fn extract_mesh(splats: Splats, views: &[(Camera, UVec2)], cfg: &ExtractConfig) -> Mesh {
    // Fold any `min_scale` floor into the raw transforms + opacities up
    // front so the entire downstream pipeline sees a single canonical
    // splat representation. With `min_scale = None`, `bake_min_scale` is
    // a no-op; with it set (training-time splats), it inflates scales
    // and energy-compensates opacity exactly once. Doing this here means
    // the seed-point sampler and the integrate kernel both work off the
    // same effective values the renderer uses — no chance of either one
    // under- or double-correcting.
    let splats = splats.bake_min_scale();

    let n_splats = splats.num_splats() as usize;
    log::info!(
        "Extracting mesh from {n_splats} splats across {} views",
        views.len()
    );

    // Whole-pipeline profile. Each phase ends with a `client.sync()` so
    // any GPU work queued during the phase gets billed to *that* phase
    // and not silently pushed to the next host call. Otherwise burn's
    // async launches make the breakdown unreadable — work shows up
    // wherever the queue happens to flush (typically the next big
    // allocation or readback).
    let sync_client = resolve_to_cube_float(splats.transforms.val())
        .client
        .clone();
    let mut phases: Vec<(&'static str, std::time::Duration)> = Vec::new();
    let mut phase_start = std::time::Instant::now();

    // 1. Pull splat tensors back to host for seed-point sampling.
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

    // Compare bboxes: splat means → seed points → (later) mesh. If seed
    // bbox is much tighter than splat-mean bbox, frustum cull is the
    // bottleneck. If they agree but the mesh is much smaller, the alpha
    // field is the bottleneck (no transitions in background).
    let splat_bbox = bbox_from_flat3(&means);
    log::info!(
        "splat means bbox: min ({:.2},{:.2},{:.2}) max ({:.2},{:.2},{:.2}) extent ({:.2}x{:.2}x{:.2})",
        splat_bbox.0.x,
        splat_bbox.0.y,
        splat_bbox.0.z,
        splat_bbox.1.x,
        splat_bbox.1.y,
        splat_bbox.1.z,
        splat_bbox.1.x - splat_bbox.0.x,
        splat_bbox.1.y - splat_bbox.0.y,
        splat_bbox.1.z - splat_bbox.0.z,
    );

    let pts = build_tetra_points(
        &means,
        &quats,
        &log_scales,
        &cams,
        &img_sizes,
        &cfg.tetra_points,
    );
    log::info!("Seed points (frustum-culled): {}", pts.points.len());
    let seed_bbox = bbox_from_vec3(&pts.points);
    log::info!(
        "seed bbox:        min ({:.2},{:.2},{:.2}) max ({:.2},{:.2},{:.2}) extent ({:.2}x{:.2}x{:.2})",
        seed_bbox.0.x,
        seed_bbox.0.y,
        seed_bbox.0.z,
        seed_bbox.1.x,
        seed_bbox.1.y,
        seed_bbox.1.z,
        seed_bbox.1.x - seed_bbox.0.x,
        seed_bbox.1.y - seed_bbox.0.y,
        seed_bbox.1.z - seed_bbox.0.z,
    );
    if pts.points.len() < 4 {
        log::warn!("Too few seed points to triangulate; returning empty mesh");
        return Mesh::default();
    }

    phases.push(("build_seeds", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    // 2 + 3. CPU Delaunay and the per-view splat pre-render are
    // independent — Delaunay only needs `pts.points` and the renders
    // only need `splats + views`. Both are expensive (Delaunay ~50 s,
    // pre-render ~25 s) so running them concurrently hides the smaller
    // behind the larger. After this point the cache feeds the initial
    // alpha eval, every binary-search iter, and the final colour eval —
    // 6 calls into one render pass.
    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };
    let pts_for_delaunay = pts.points.clone();
    let delaunay_handle = tokio::task::spawn_blocking(move || {
        let span = tracing::trace_span!("delaunay_3d").entered();
        let tets = if std::env::var("BRUSH_DELAUNAY_PAR").is_ok() {
            crate::delaunay_par::delaunay_3d_lockgrid(&pts_for_delaunay)
        } else {
            delaunay_3d(&pts_for_delaunay)
        };
        drop(span);
        tets
    });
    // Pre-render + initial alpha eval together as one async unit so
    // they share the parallel slot with Delaunay. Initial alpha can't
    // start until pre-render is done (it reads the cache), but both
    // together run in parallel with Delaunay.
    let splats_ref = &splats;
    let pts_ref = &pts.points;
    let gpu_block = async move {
        let cache = pre_render_views(splats_ref, views, render_mode).await;
        let alpha = evaluate_alpha(splats_ref, pts_ref, &cache, false)
            .await
            .alpha;
        (cache, alpha)
    };
    let ((view_cache, alpha), tets_result) = tokio::join!(gpu_block, delaunay_handle);
    let tets = tets_result.expect("delaunay task panicked");
    log::info!("Delaunay tets: {}", tets.len());
    sync_client.sync().await.expect("sync");
    phases.push(("delaunay_and_initial_alpha", phase_start.elapsed()));
    phase_start = std::time::Instant::now();
    let sdf: Vec<f32> = alpha.iter().map(|a| a - cfg.iso_value).collect();

    // 4. Marching tets.
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

    // 5. Binary search refinement, with the bracket state living on
    // the GPU. Each iter dispatches: compute_midpoints → 292 integrate
    // launches → bracket_update. The only per-step CPU↔GPU traffic is
    // a 4-byte readback of the new active count.
    let device = resolve_to_cube_float(splats.transforms.val()).device;
    let mut state = RefineState::new(&mt.crossings, &pts.points, &sdf, device);
    for step in 0..N_STEPS {
        if state.n_active() == 0 {
            break;
        }
        let mid_pos_t = state.compute_midpoints_t();
        let min_alpha_t =
            integrate_alpha_tiled_min_t(&splats, &mid_pos_t, state.n_active(), &view_cache);
        state.update_bracket_t(min_alpha_t, cfg.iso_value).await;
        sync_client.sync().await.expect("sync");
        let name: &'static str = match step {
            0 => "  bs_step_1",
            1 => "  bs_step_2",
            2 => "  bs_step_3",
            3 => "  bs_step_4",
            4 => "  bs_step_5",
            5 => "  bs_step_6",
            6 => "  bs_step_7",
            7 => "  bs_step_8",
            _ => "  bs_step_N",
        };
        phases.push((name, phase_start.elapsed()));
        phase_start = std::time::Instant::now();
        log::info!(
            "Binary search step {}/{} done ({} crossings still active)",
            step + 1,
            N_STEPS,
            state.n_active(),
        );
    }
    let refined = state.finish().await;

    // 6. Per-crossing keep mask. Matches GOF's `filter_mesh` step
    // (`extract_mesh.py`): for each crossing, keep iff the *original
    // Delaunay edge length* is ≤ the *sum of the two endpoint Gaussian
    // scales*. Faces are then kept only if all three crossings survive.
    // We use the *Delaunay* edge, not the refined mesh-edge — bisection-
    // refined positions sit between the endpoints, so mesh edges are
    // always shorter. Filtering on the original Delaunay edge length is
    // the principled test for "this crossing bridges two Gaussians that
    // don't really touch."
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

    // Per-vertex colour: evaluate alpha+RGB once more at the *refined*
    // vertex positions and bake a visibility-weighted blend of all
    // contributing views' RGB. Vertex positions are now on the iso-
    // surface so each view samples a pixel that shows the actual
    // surface material rather than whatever was in front.
    let final_eval = evaluate_alpha(&splats, &refined, &view_cache, true).await;
    sync_client.sync().await.expect("sync");
    phases.push(("final_color_eval", phase_start.elapsed()));
    phase_start = std::time::Instant::now();

    let vertex_colors: Vec<[u8; 3]> = final_eval
        .color
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
        vertex_scales: Vec::new(),
        vertex_colors,
        faces: mt.faces,
    };

    mesh = filter_mesh_with_keep(&mesh, &keep_crossing);

    // Drop vertices outside the splat-means bbox + margin. These are
    // outlier verts produced when binary search refines a crossing
    // along a Delaunay edge that spans the empty far field — e.g.
    // billboard splats whose 3σ corners stretch tens of units beyond
    // the actual scene. They don't represent any surface, just noise
    // in the alpha field at the bbox boundary. The edge-length filter
    // above doesn't catch them because their parent Gaussians are
    // huge, so `scale_a + scale_b` is huge too. Cropping in world space
    // does.
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

    let mesh_bbox = bbox_from_vec3(&mesh.vertices);
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

fn bbox_from_flat3(xs: &[f32]) -> (Vec3, Vec3) {
    let n = xs.len() / 3;
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    for i in 0..n {
        let p = Vec3::new(xs[3 * i], xs[3 * i + 1], xs[3 * i + 2]);
        mn = mn.min(p);
        mx = mx.max(p);
    }
    (mn, mx)
}

fn bbox_from_vec3(xs: &[Vec3]) -> (Vec3, Vec3) {
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    for &p in xs {
        mn = mn.min(p);
        mx = mx.max(p);
    }
    (mn, mx)
}

/// Per-point alpha integration across all views.
///
/// Matches GOF's `evaluage_alpha`: `final_alpha = min_views(α_int_view)`,
/// returned `α(p) = 1 − final_alpha = max_views(T_view(p))` where
/// `α_int_view` is the kernel's volume-integrated absorption from the
/// camera to `p` for that view and `T_view = 1 − α_int_view` is the
/// surviving transmittance to `p`.
///
/// The "max over views" reading of T is the carving rule: a point is
/// labeled "open" (α → 1) as soon as **any** view has a clear line of
/// sight to it, and "solid" (α → 0) only when **every** view is blocked
/// by intervening Gaussians. Using min(T) instead would label a point
/// "solid" if even one view is blocked, producing the union of all
/// camera shadow volumes — i.e. fattening every object by the silhouette
/// extruded from every camera.
///
/// The iso-surface `α = 0.5` then sits where "the best view sees half-
/// transmittance," which is the actual material boundary.
/// Output of [`evaluate_alpha`]: max-transmittance alpha per query point,
/// plus (when `track_color = true`) the RGB sample from whichever view
/// gave that max transmittance. Colours are linear [0, 1] floats; the
/// caller is expected to clamp + quantise for PLY output.
pub struct EvaluateOut {
    pub alpha: Vec<f32>,
    /// One RGB triple per point. Empty when `track_color = false`.
    pub color: Vec<[f32; 3]>,
}

/// Cached splat-forward-pass output for one view, reused across the
/// initial alpha eval, all binary-search iters, and the final colour
/// eval. The render is the dominant per-view cost (project + sort +
/// tile + rasterize on ~750k splats) and is identical across these
/// calls — caching cuts ~75% of repeated work.
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
            render_mode,
            Vec3::ZERO,
            RasterPass::Forward,
            false,
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

/// Per-view tile-cooperative ray-gaussian integrate.
///
/// Per view, instead of dispatching one thread per query vertex over the
/// untouched gaussian list, we:
/// 1. project all vertices to `(tile_id, ray_dir_xy, depth, pix_xy)`
/// 2. histogram the `tile_ids` → exclusive prefix sum → `vertex_tile_offsets`
/// 3. atomic-scatter vertex ids into `sorted_indices`, grouped by tile
/// 4. dispatch `integrate_alpha_tiled_kernel` with one workgroup per
///    tile, each workgroup cooperatively streaming the tile's gaussians
///    through shared memory while every thread evaluates its own
///    vertex.
///
/// This matches the structure of brush's rasterize kernel (and GOF's
/// `IntegrateGaussianAlphaToPoints`) — the win is shared-memory reuse
/// of gaussian data across all threads in a workgroup, instead of every
/// thread independently fetching from global memory.
fn integrate_alpha_tiled_min_t(
    splats: &Splats,
    pts_tensor: &FloatTensor<MainBackendBase>,
    n: usize,
    view_renders: &[ViewRender],
) -> FloatTensor<MainBackendBase> {
    type B = MainBackendBase;
    use brush_cube::{create_tensor, create_tensor_from_slice};
    use brush_render::kernels::integrate_tiled;
    use burn::tensor::DType;

    let transforms_p = resolve_to_cube_float(splats.transforms.val());
    let raw_opacities_p = resolve_to_cube_float(splats.raw_opacities.val());
    let device = transforms_p.device.clone();
    let client = transforms_p.client.clone();

    // Aggregator: lives across all 292 view kernels.
    let min_alpha_t: FloatTensor<B> = B::float_from_data(
        burn::tensor::TensorData::new(vec![f32::INFINITY; n], [n]),
        &device,
    );
    // Dummy colour tensors (kernel won't write since track_color = false).
    let color_sum_t: FloatTensor<B> = B::float_from_data(
        burn::tensor::TensorData::new(vec![0.0f32; 3], [1, 3]),
        &device,
    );
    let weight_sum_t: FloatTensor<B> =
        B::float_from_data(burn::tensor::TensorData::new(vec![0.0f32; 1], [1]), &device);

    // Per-view scratch tensors — sized at construction, reused across
    // all 292 views. The kernel inputs are `&mut Tensor<_>` so the
    // contents are overwritten each view; nothing carries between
    // views except the running aggregators.
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

        // 1) Project every vertex to (tile_id, depth, ray_dir, pix).
        integrate_tiled::project_vertices_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate_tiled::TILE_SIZE),
            CubeDim::new_1d(integrate_tiled::TILE_SIZE),
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

        // 2) Shifted histogram → exclusive prefix sum for vertex_tile_offsets.
        // Allocate fresh per view (cheap; ~16 KB).
        let counts_t = create_tensor_from_slice(&vec![0u32; n_tiles + 1], &device, DType::U32);
        integrate_tiled::histogram_tile_ids_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate_tiled::TILE_SIZE),
            CubeDim::new_1d(integrate_tiled::TILE_SIZE),
            tile_ids_t.clone().into_tensor_arg(),
            counts_t.clone().into_tensor_arg(),
            n as u32,
            tile_bw * tile_bh,
        );
        let vertex_tile_offsets_t = brush_prefix_sum::prefix_sum(counts_t);

        // 3) Atomic-scatter into sorted_indices, with per-tile write
        // counters (one u32 per tile — bounded contention).
        let write_counters_t = create_tensor_from_slice(&vec![0u32; n_tiles], &device, DType::U32);
        integrate_tiled::scatter_vertices_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, integrate_tiled::TILE_SIZE),
            CubeDim::new_1d(integrate_tiled::TILE_SIZE),
            tile_ids_t.clone().into_tensor_arg(),
            vertex_tile_offsets_t.clone().into_tensor_arg(),
            write_counters_t.clone().into_tensor_arg(),
            sorted_indices_t.clone().into_tensor_arg(),
            n as u32,
            tile_bw * tile_bh,
        );

        // 4) Tile-cooperative integrate. Dispatch one WG per tile in 1D
        // (rasterize uses the same pattern — `CUBE_POS` is the tile id).
        let num_tiles_u32 = tile_bw * tile_bh;
        integrate_tiled::integrate_alpha_tiled_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(
                num_tiles_u32 * integrate_tiled::TILE_SIZE,
                integrate_tiled::TILE_SIZE,
            ),
            CubeDim::new_1d(integrate_tiled::TILE_SIZE),
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
            false, // track_color
        );
    }

    min_alpha_t
}

async fn evaluate_alpha(
    splats: &Splats,
    points: &[Vec3],
    view_renders: &[ViewRender],
    track_color: bool,
) -> EvaluateOut {
    type B = MainBackendBase;
    let n = points.len();
    if n == 0 {
        return EvaluateOut {
            alpha: Vec::new(),
            color: Vec::new(),
        };
    }

    let transforms_p = resolve_to_cube_float(splats.transforms.val());
    let raw_opacities_p = resolve_to_cube_float(splats.raw_opacities.val());
    let device = transforms_p.device.clone();
    let client = transforms_p.client.clone();

    let pts_flat: Vec<f32> = points.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let pts_tensor: FloatTensor<B> =
        B::float_from_data(burn::tensor::TensorData::new(pts_flat, [n, 3]), &device);

    // Running aggregators live on the GPU so we never round-trip per
    // view: min_alpha holds the per-point running min across all
    // visited views (init +∞), color_sum / weight_sum accumulate the
    // visibility-weighted blend (init 0). Read back exactly once at
    // the end of the per-view loop.
    let min_alpha_t: FloatTensor<B> = B::float_from_data(
        burn::tensor::TensorData::new(vec![f32::INFINITY; n], [n]),
        &device,
    );
    let color_sum_t: FloatTensor<B> =
        B::float_from_data(burn::tensor::TensorData::zeros::<f32, _>([n, 3]), &device);
    let weight_sum_t: FloatTensor<B> =
        B::float_from_data(burn::tensor::TensorData::zeros::<f32, _>([n]), &device);

    for (view_idx, view_render) in view_renders.iter().enumerate() {
        // The render has already been done once during pre-render; we
        // clone the tensor handles (Arc-backed in burn, so cheap) and
        // hand them straight to the integrate kernel, which folds
        // directly into the running per-point aggregators. No per-view
        // scratch tensor, no separate aggregate launch.
        let out = view_render.render.clone();
        let uniforms = out.project_uniforms.to_launch_object();
        let camera_model = view_render.camera_model;

        kernels::integrate_alpha::integrate_alpha_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n as u32, kernels::integrate_alpha::WG_SIZE),
            CubeDim::new_1d(kernels::integrate_alpha::WG_SIZE),
            transforms_p.clone().into_tensor_arg(),
            raw_opacities_p.clone().into_tensor_arg(),
            out.compact_gid_from_isect.into_tensor_arg(),
            out.aux.tile_offsets.into_tensor_arg(),
            out.global_from_compact_gid.into_tensor_arg(),
            out.out_img.into_tensor_arg(),
            pts_tensor.clone().into_tensor_arg(),
            min_alpha_t.clone().into_tensor_arg(),
            color_sum_t.clone().into_tensor_arg(),
            weight_sum_t.clone().into_tensor_arg(),
            n as u32,
            uniforms,
            camera_model,
            track_color,
        );

        if (view_idx + 1).is_multiple_of(32) || view_idx + 1 == view_renders.len() {
            log::info!("integrated view {}/{}", view_idx + 1, view_renders.len());
        }
    }

    // Single readback of the running aggregators after all views are folded.
    let min_alpha_int = read_back_f32(min_alpha_t).await;
    let (color_sum_flat, weight_sum) = if track_color {
        (
            read_back_f32(color_sum_t).await,
            read_back_f32(weight_sum_t).await,
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // Points that were outside every camera's frustum get α_int = +∞ →
    // emit α = 1 (no occlusion) and black RGB. This matches GOF's
    // boundary behaviour where "never seen" defaults to "open space".
    let alpha = min_alpha_int
        .iter()
        .map(|&a| if a.is_finite() { 1.0 - a } else { 1.0 })
        .collect();

    // Resolve accumulated colour. Vertices with zero accumulated
    // weight (no view ever saw them; rare with seed-time frustum cull
    // but possible after binary-search refinement) fall back to white
    // — matches GOF's `torch.ones(...)` init.
    let color: Vec<[f32; 3]> = if track_color {
        (0..n)
            .map(|i| {
                let w = weight_sum[i];
                if w > 1.0e-6 {
                    [
                        color_sum_flat[3 * i] / w,
                        color_sum_flat[3 * i + 1] / w,
                        color_sum_flat[3 * i + 2] / w,
                    ]
                } else {
                    [1.0, 1.0, 1.0]
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    EvaluateOut { alpha, color }
}

/// Drain a 1-D `f32` tensor from the GPU into host memory.
async fn read_back_f32(t: FloatTensor<MainBackendBase>) -> Vec<f32> {
    let tp = TransactionPrimitive::<MainBackendBase>::new(vec![t], vec![], vec![], vec![]);
    let data = <MainBackendBase as TransactionOps<MainBackendBase>>::tr_execute(tp)
        .await
        .expect("read alpha");
    data.read_floats[0]
        .clone()
        .into_vec::<f32>()
        .expect("alpha f32")
}
