//! Top-level mesh-extraction driver. Glues tetra-point sampling, CPU
//! Delaunay, per-view GPU opacity integration, marching tets, and binary
//! search refinement.

use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor};
use brush_render::SplatOps;
use brush_render::burn_glue::resolve_to_cube_float;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterPass, SplatRenderMode, Splats};
use brush_render::kernels;
use burn::backend::ops::{FloatTensorOps, TransactionOps, TransactionPrimitive};
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, s};
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::WgpuRuntime;
use glam::{UVec2, Vec3};

use crate::Mesh;
use crate::binary_search::{N_STEPS, RefineState};
use crate::delaunay::delaunay_3d;
use crate::filter::filter_mesh_with_keep;
use crate::marching_tet::marching_tets;
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

/// Visibility weight exponent for per-vertex colour blending. Each view
/// contributes its rgb weighted by `(1 − α_int)^COLOR_BLEND_POWER`.
/// Higher = more bias to best-vis views (sharper, more like GOF's
/// best-view convention but with smooth fallback for ties); lower =
/// more averaging across moderately-visible views (more robust but
/// blurrier). Empirically k=2 wins the PSNR-vs-splat metric on bonsai,
/// but visually that's a soft blur of view-dependent SH — colours
/// look "smeared". k=8 puts most weight on the top ~2 views per
/// vertex, sharper at the cost of more per-view appearance variance.
const COLOR_BLEND_POWER: f32 = 8.0;

/// Extract a triangle mesh from `splats`. `views` carries the per-view
/// `(camera, image_size)` pairs used for both frustum-culling the seed
/// points and integrating the opacity along rays.
pub async fn extract_mesh(splats: Splats, views: &[(Camera, UVec2)], cfg: &ExtractConfig) -> Mesh {
    let n_splats = splats.num_splats() as usize;
    log::info!(
        "Extracting mesh from {n_splats} splats across {} views",
        views.len()
    );

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

    let cams: Vec<Camera> = views.iter().map(|(c, _)| *c).collect();
    let img_sizes: Vec<UVec2> = views.iter().map(|(_, s)| *s).collect();

    // GOF's 3D filter: per-Gaussian isotropic minimum half-extent based
    // on the closest training camera. Inflates the effective scale and
    // (in the integrate kernel) is paired with an opacity-compensation
    // coefficient to preserve the density's integral.
    let filter_3d = crate::filter_3d::compute_filter_3d(&means, &cams, &img_sizes);
    if !filter_3d.is_empty() {
        let max_f = filter_3d.iter().copied().fold(0.0_f32, f32::max);
        let mean_f = filter_3d.iter().sum::<f32>() / filter_3d.len() as f32;
        log::info!("filter_3d: mean={mean_f:.4}, max={max_f:.4}");
    }

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
        &filter_3d,
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

    // 2. CPU Delaunay 3D.
    let span = tracing::trace_span!("delaunay_3d").entered();
    let tets = delaunay_3d(&pts.points);
    drop(span);
    log::info!("Delaunay tets: {}", tets.len());

    // 3. Evaluate per-vertex alpha across all views (min-T accumulation).
    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };
    let alpha = evaluate_alpha(&splats, &pts.points, views, render_mode, false)
        .await
        .alpha;
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

    // 5. Binary search refinement.
    let mut state = RefineState::new(&mt.crossings, &pts.points, &sdf);
    for step in 0..N_STEPS {
        let mids = state.midpoints();
        let mid_alpha = evaluate_alpha(&splats, &mids, views, render_mode, false)
            .await
            .alpha;
        let mid_sdf: Vec<f32> = mid_alpha.iter().map(|a| a - cfg.iso_value).collect();
        state.step(&mid_sdf);
        log::info!("Binary search step {}/{} done", step + 1, N_STEPS);
    }
    let refined = state.finish();

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
    let final_eval = evaluate_alpha(&splats, &refined, views, render_mode, true).await;
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

async fn evaluate_alpha(
    splats: &Splats,
    points: &[Vec3],
    views: &[(Camera, UVec2)],
    render_mode: SplatRenderMode,
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
    let sh_coeffs_p = resolve_to_cube_float(splats.sh_coeffs.val());
    let raw_opacities_p = resolve_to_cube_float(splats.raw_opacities.val());
    let device = transforms_p.device.clone();
    let client = transforms_p.client.clone();

    let pts_flat: Vec<f32> = points.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let pts_tensor: FloatTensor<B> =
        B::float_from_data(burn::tensor::TensorData::new(pts_flat, [n, 3]), &device);

    // Track the *smallest* α_int seen across views (= the view with the
    // best line of sight). Init to `+∞` so any real value wins on the
    // first view. Used for the SDF aggregation.
    let mut min_alpha_int = vec![f32::INFINITY; n];

    // Colour: GOF's reference takes "best view's RGB" (where(α<final_α,
    // color, final_color)). For the mesh-render-vs-splat-render PSNR
    // metric we instead accumulate a *visibility-weighted* mean — each
    // view contributes its rgb weighted by `(1 − α_int)²`, so the
    // best-visibility view dominates but moderately-visible views
    // smooth out per-view SH variation. The mesh is unlit + per-vertex
    // colour, so its single baked-in colour has to approximate the
    // splat appearance from *all* views simultaneously; a weighted mean
    // does that better than one specific view's snapshot. Off by ~+0.3
    // PSNR(mesh vs splat) on bonsai vs the best-view variant.
    let mut color_sum: Vec<[f32; 3]> = if track_color {
        vec![[0.0; 3]; n]
    } else {
        Vec::new()
    };
    let mut weight_sum: Vec<f32> = if track_color {
        vec![0.0; n]
    } else {
        Vec::new()
    };

    for (view_idx, (cam, sz)) in views.iter().enumerate() {
        // Use Forward pass: produces a `[H, W]` packed-u32 RGBA image
        // (the kernel below bitcasts and unpacks it for colour sampling)
        // and crucially leaves `tile_offsets[1]` un-truncated. Backward
        // would shrink it to the *pixel-centre* `last_useful_isect`,
        // which can be tighter than GOF's 5-sub-pixel-corner contributor
        // range — i.e. a sub-pixel splat that contributes at a corner
        // but not the centre is past brush's center-only cutoff. Using
        // Forward keeps every tile gaussian in scope for the kernel's
        // per-query-point walk and matches GOF's broader contributor
        // set semantically (we just iterate everything; GOF prunes
        // contributed_ids[] for perf, not correctness).
        let out = <B as SplatOps<B>>::render(
            cam,
            *sz,
            transforms_p.clone(),
            sh_coeffs_p.clone(),
            raw_opacities_p.clone(),
            render_mode,
            Vec3::ZERO,
            RasterPass::Forward,
            false, // no geometry render needed
        )
        .await;

        let out_alpha: FloatTensor<B> = create_tensor::<1>([n], &device, DType::F32);
        let out_rgb: FloatTensor<B> = create_tensor::<2>([n, 3], &device, DType::F32);
        let uniforms = out.project_uniforms.to_launch_object();
        let camera_model = cam.camera_model;

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
            out_alpha.clone().into_tensor_arg(),
            out_rgb.clone().into_tensor_arg(),
            n as u32,
            uniforms,
            camera_model,
        );

        let alpha_int = read_back_f32(out_alpha).await;
        let rgb_flat = if track_color {
            read_back_f32(out_rgb).await
        } else {
            Vec::new()
        };
        debug_assert_eq!(alpha_int.len(), n);
        for i in 0..n {
            let a = alpha_int[i];
            if a < min_alpha_int[i] {
                min_alpha_int[i] = a;
            }
            if track_color {
                // Visibility = 1 − α_int (line-of-sight openness). Raise
                // to `COLOR_BLEND_POWER` so the most-visible views
                // dominate the bake without being a hard winner-takes-all.
                let vis = (1.0 - a).max(0.0);
                let w = vis.powf(COLOR_BLEND_POWER);
                if w > 0.0 {
                    color_sum[i][0] += w * rgb_flat[3 * i];
                    color_sum[i][1] += w * rgb_flat[3 * i + 1];
                    color_sum[i][2] += w * rgb_flat[3 * i + 2];
                    weight_sum[i] += w;
                }
            }
        }

        if (view_idx + 1).is_multiple_of(8) || view_idx + 1 == views.len() {
            log::info!("integrated view {}/{}", view_idx + 1, views.len());
        }
    }

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
        color_sum
            .into_iter()
            .zip(weight_sum.into_iter())
            .map(|(c, w)| {
                if w > 1.0e-6 {
                    [c[0] / w, c[1] / w, c[2] / w]
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
