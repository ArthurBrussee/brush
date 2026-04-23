//! Fast iterative performance harness.
//!
//! Run synthetic-splat training steps with explicit GPU sync and report
//! min/median/p95 step time. Designed to be re-run frequently while
//! iterating on optimizations.
//!
//! Usage:
//!   # 5M synthetic splats, 1620p, 12 measured iters:
//!   cargo run --release -p brush-bench-test --example perf -- --bench
//!
//!   # custom shape:
//!   cargo run --release -p brush-bench-test --example perf -- \
//!     --splats 2000000 --width 1920 --height 1080 --iters 8 --bench
//!
//!   # real PLY (skips silently if file is missing):
//!   BRUSH_PLY=path/to/file.ply cargo run --release -p brush-bench-test \
//!     --example perf -- --bench
//!
//! `--bench` skips the debug-validation feature's per-step readbacks (the
//! `brush-render` crate gates on that arg).
//!
//! Designed to be cheap to re-run: one process, no warm-up reload, finishes
//! in ~5s on a 4070 Ti at the defaults.

#![recursion_limit = "256"]

use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{Subscriber, span};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, MainBackend,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
};
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::{
    backend::{Autodiff, wgpu::WgpuDevice},
    prelude::Backend,
    tensor::TensorData,
};
use burn_cubecl::cubecl::future::block_on;
use glam::{Quat, Vec3};

type DiffBackend = Autodiff<MainBackend>;

fn parse_arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == key {
            if let Some(v) = iter.next() {
                if let Ok(parsed) = v.parse() {
                    return parsed;
                }
            }
        }
    }
    default
}

fn gen_splats_random(device: &WgpuDevice, count: usize, seed: u64) -> Splats<DiffBackend> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-5.0..5.0_f32),
                rng.random_range(-3.0..3.0_f32),
                rng.random_range(-10.0..10.0_f32),
            ]
        })
        .collect();
    let log_scales: Vec<f32> = (0..count)
        .flat_map(|_| {
            let base = rng.random_range(0.01..0.1_f32).ln();
            [base, base, base]
        })
        .collect();
    let rotations: Vec<f32> = (0..count)
        .flat_map(|_| {
            let u1 = rng.random::<f32>();
            let u2 = rng.random::<f32>();
            let u3 = rng.random::<f32>();
            let s1 = (1.0 - u1).sqrt();
            let s2 = u1.sqrt();
            let t1 = 2.0 * std::f32::consts::PI * u2;
            let t2 = 2.0 * std::f32::consts::PI * u3;
            [s1 * t1.sin(), s1 * t1.cos(), s2 * t2.sin(), s2 * t2.cos()]
        })
        .collect();
    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(0.1..0.9_f32),
                rng.random_range(0.1..0.9_f32),
                rng.random_range(0.1..0.9_f32),
            ]
        })
        .collect();
    let opacities: Vec<f32> = (0..count).map(|_| rng.random_range(0.05..1.0_f32)).collect();
    Splats::<DiffBackend>::from_raw(
        means,
        rotations,
        log_scales,
        sh_coeffs,
        opacities,
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0)
}

fn make_orbit_batches(
    centroid: Vec3,
    radius: f32,
    n: usize,
    resolution: (u32, u32),
) -> Vec<SceneBatch> {
    let (w, h) = resolution;
    let img_data: Vec<f32> = (0..(w * h * 3) as usize)
        .map(|i| {
            let p = i / 3;
            let x = (p as u32) % w;
            let y = (p as u32) / w;
            let nx = x as f32 / w as f32;
            let ny = y as f32 / h as f32;
            match i % 3 {
                0 => nx * 0.6 + 0.2,
                1 => ny * 0.6 + 0.2,
                2 => (nx + ny) * 0.3 + 0.4,
                _ => 0.0,
            }
        })
        .collect();
    let img_tensor = TensorData::new(img_data, [h as usize, w as usize, 3]);
    (0..n)
        .map(|i| {
            let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
            let eye = centroid
                + Vec3::new(
                    radius * 1.5 * angle.sin(),
                    radius * 0.3,
                    radius * 1.5 * angle.cos(),
                );
            let forward = (centroid - eye).normalize_or_zero();
            let right = Vec3::Y.cross(forward).normalize_or_zero();
            let up = forward.cross(right);
            let mat = glam::Mat3::from_cols(right, up, forward);
            let quat = Quat::from_mat3(&mat);
            let camera = Camera::new(
                eye,
                quat,
                50.0_f64.to_radians(),
                50.0_f64.to_radians(),
                glam::vec2(0.5, 0.5),
            );
            SceneBatch {
                img_tensor: img_tensor.clone(),
                alpha_mode: AlphaMode::Transparent,
                camera,
            }
        })
        .collect()
}

fn synthetic_camera_for(splats: &Splats<DiffBackend>) -> (Vec3, f32) {
    let means_data = block_on(splats.means().clone().into_data_async()).expect("means");
    let means: Vec<f32> = means_data.into_vec().expect("vec");
    let n = (means.len() / 3) as f32;
    let mut sum = Vec3::ZERO;
    for c in means.chunks_exact(3) {
        sum += Vec3::new(c[0], c[1], c[2]);
    }
    let centroid = sum / n;
    let mut max_dist_sq = 0.0_f32;
    for c in means.chunks_exact(3) {
        let p = Vec3::new(c[0], c[1], c[2]);
        let d = (p - centroid).length_squared();
        if d > max_dist_sq {
            max_dist_sq = d;
        }
    }
    (centroid, max_dist_sq.sqrt().max(1.0))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn run_isolated_phases(
    device: &WgpuDevice,
    splats: &Splats<DiffBackend>,
    batch: &SceneBatch,
    iters: usize,
) {
    use brush_render_bwd::render_splats as render_splats_diff;
    use burn::tensor::{Distribution, Tensor, TensorPrimitive, s, activation::sigmoid};

    let img_h = batch.img_tensor.shape.dims::<3>()[0];
    let img_w = batch.img_tensor.shape.dims::<3>()[1];
    let img_size = glam::uvec2(img_w as u32, img_h as u32);
    let camera = batch.camera.clone();
    let bg = Vec3::ZERO;
    let n_splats = splats.num_splats() as usize;

    fn time<F: FnMut()>(name: &str, device: &WgpuDevice, iters: usize, mut f: F) {
        for _ in 0..3 {
            f();
        }
        MainBackend::sync(device).expect("sync");
        let mut ts = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            f();
            MainBackend::sync(device).expect("sync");
            ts.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = ts[0];
        let p50 = ts[ts.len() / 2];
        println!("  {name:36} min {min:7.2} ms  p50 {p50:7.2} ms");
    }

    println!("\n=== isolated phase budget ===");

    // 1. Forward render (with bwd_info=true so it matches train cost).
    {
        let splats = splats.clone();
        let camera = camera.clone();
        time("forward render", device, iters, || {
            let _ = block_on(render_splats_diff(splats.clone(), &camera, img_size, bg));
        });
    }

    // 2. Forward + backward render (degenerate gradient).
    {
        let splats = splats.clone();
        let camera = camera.clone();
        time("forward + degenerate backward", device, iters, || {
            let diff = block_on(render_splats_diff(splats.clone(), &camera, img_size, bg));
            let img: Tensor<DiffBackend, 3> =
                Tensor::from_primitive(TensorPrimitive::Float(diff.img));
            let _ = img.mean().backward();
        });
    }

    // 3. Forward + L1 + SSIM + backward (the actual loss path).
    {
        let splats = splats.clone();
        let camera = camera.clone();
        let gt_data = batch.img_tensor.clone();
        let gt = Tensor::<DiffBackend, 3>::from_data(gt_data, device);
        let gt_rgb = gt.slice(s![.., .., 0..3]);
        time("forward + L1+SSIM + backward", device, iters, || {
            let diff = block_on(render_splats_diff(splats.clone(), &camera, img_size, bg));
            let img: Tensor<DiffBackend, 3> =
                Tensor::from_primitive(TensorPrimitive::Float(diff.img));
            let pred_rgb = img.slice(s![.., .., 0..3]);
            let l1 = (pred_rgb.clone() - gt_rgb.clone()).abs().mean();
            let ssim = brush_fused_ssim::fused_ssim(pred_rgb, gt_rgb.clone()).mean();
            let loss = l1 * 0.8 - ssim * 0.2;
            let _ = loss.backward();
        });
    }

    // 4. Noise add path (the chain that train.rs's noise step submits).
    {
        let raw_op = splats.raw_opacities.val().inner();
        let visible: Tensor<MainBackend, 1> =
            Tensor::random([n_splats], Distribution::Uniform(0.0, 1.0), device);
        let transforms = splats.transforms.val().inner();
        time("noise add chain", device, iters, || {
            let opac = sigmoid(raw_op.clone());
            let inv_opac: Tensor<MainBackend, 1> = 1.0 - opac;
            let nw = inv_opac.powi_scalar(150).clamp(0.0, 1.0) * visible.clone();
            let nw = nw.unsqueeze_dim(1);
            let samples =
                Tensor::<MainBackend, 2>::random([n_splats, 3], Distribution::Normal(0.0, 1.0), device);
            let nwm = nw * 0.001_f32;
            let noise_m = (samples * nwm).clamp(-1.0_f32, 1.0_f32);
            let inner = transforms.clone();
            let noised = inner.clone().slice(s![.., 0..3]) + noise_m;
            let _ = inner.slice_assign(s![.., 0..3], noised);
        });
    }

    // 5. Random tensor only.
    {
        time("Tensor::random Normal[N, 3]", device, iters, || {
            let _ = Tensor::<MainBackend, 2>::random(
                [n_splats, 3],
                Distribution::Normal(0.0, 1.0),
                device,
            );
        });
    }

    // 6. Adam-like step on each param shape.
    {
        let theta = Tensor::<MainBackend, 2>::random(
            [n_splats, 10],
            Distribution::Uniform(-1.0, 1.0),
            device,
        );
        let g = Tensor::<MainBackend, 2>::random(
            [n_splats, 10],
            Distribution::Uniform(-0.1, 0.1),
            device,
        );
        let m1 = Tensor::<MainBackend, 2>::zeros([n_splats, 10], device);
        let m2 = Tensor::<MainBackend, 2>::zeros([n_splats, 10], device);
        time("Adam-like step on transforms [N,10]", device, iters, || {
            let beta1 = 0.9_f32;
            let beta2 = 0.999_f32;
            let eps = 1e-15_f32;
            let lr = 1e-4_f32;
            let m1_n = m1.clone() * beta1 + g.clone() * (1.0 - beta1);
            let m2_n = m2.clone() * beta2 + g.clone().powi_scalar(2) * (1.0 - beta2);
            let _ = theta.clone() - m1_n / (m2_n.sqrt() + eps) * lr;
        });
    }

    // 7. Adam-like step on SH coeffs rest [N, 8, 3] (the big one for sh_degree=2).
    {
        let sh_rest_dim = 8;
        let theta = Tensor::<MainBackend, 3>::random(
            [n_splats, sh_rest_dim, 3],
            Distribution::Uniform(-0.1, 0.1),
            device,
        );
        let g = Tensor::<MainBackend, 3>::random(
            [n_splats, sh_rest_dim, 3],
            Distribution::Uniform(-0.001, 0.001),
            device,
        );
        let m1 = Tensor::<MainBackend, 3>::zeros([n_splats, sh_rest_dim, 3], device);
        let m2 = Tensor::<MainBackend, 3>::zeros([n_splats, sh_rest_dim, 3], device);
        time("Adam-like step on SH rest [N,8,3]", device, iters, || {
            let beta1 = 0.9_f32;
            let beta2 = 0.999_f32;
            let eps = 1e-15_f32;
            let lr = 2e-4_f32;
            let m1_n = m1.clone() * beta1 + g.clone() * (1.0 - beta1);
            let m2_n = m2.clone() * beta2 + g.clone().powi_scalar(2) * (1.0 - beta2);
            let _ = theta.clone() - m1_n / (m2_n.sqrt() + eps) * lr;
        });
    }

    // 8. The from_data hot spot: large image upload.
    {
        let h = 1620usize;
        let w = 2880usize;
        let img_data: Vec<f32> = (0..(h * w * 3)).map(|i| (i as f32 * 0.001).sin()).collect();
        time("from_data 2880x1620x3", device, iters, || {
            let _ = Tensor::<MainBackend, 1>::from_floats(img_data.as_slice(), device).reshape([h, w, 3]);
        });
    }
}

#[derive(Default)]
struct PhaseStats {
    by_name: std::collections::BTreeMap<String, Vec<f64>>,
}
struct PhaseLayer {
    table: Arc<Mutex<PhaseStats>>,
    enabled: Arc<std::sync::atomic::AtomicBool>,
}
struct SpanState {
    started: Instant,
    name: String,
}
impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for PhaseLayer {
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("span");
        span.extensions_mut().insert(SpanState {
            started: Instant::now(),
            name: attrs.metadata().name().to_owned(),
        });
    }
    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let span = ctx.span(&id).expect("span");
        if let Some(state) = span.extensions().get::<SpanState>() {
            let dt_ms = state.started.elapsed().as_secs_f64() * 1000.0;
            self.table
                .lock()
                .expect("lock")
                .by_name
                .entry(state.name.clone())
                .or_default()
                .push(dt_ms);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let splat_count: usize = parse_arg(&args, "--splats", 5_000_000);
    let width: u32 = parse_arg(&args, "--width", 2880);
    let height: u32 = parse_arg(&args, "--height", 1620);
    let iters: usize = parse_arg(&args, "--iters", 12);
    let warmup: usize = parse_arg(&args, "--warmup", 3);
    let show_phases = args.iter().any(|a| a == "--phases");
    let kernel_sync = args.iter().any(|a| a == "--kernel-sync");
    if kernel_sync {
        // Each render kernel will be followed by a blocking GPU sync, so the
        // surrounding tracing spans measure true GPU time. Throws step
        // throughput out the window — only useful with --phases.
        brush_render::render::PROFILE_SYNC
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    let table = Arc::new(Mutex::new(PhaseStats::default()));
    let enabled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    if show_phases {
        let layer = PhaseLayer {
            table: table.clone(),
            enabled: enabled.clone(),
        };
        tracing_subscriber::registry().with(layer).init();
    }

    let device = WgpuDevice::default();

    let (splats, splat_count_actual) = if let Ok(ply_path) = std::env::var("BRUSH_PLY") {
        eprintln!("Loading PLY {ply_path}...");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            use brush_serde::stream_splat_from_ply;
            use futures_lite::StreamExt;
            let file = tokio::fs::File::open(&ply_path).await.expect("open ply");
            let reader = tokio::io::BufReader::new(file);
            let mut stream = Box::pin(stream_splat_from_ply(reader, None, false));
            let msg = stream
                .next()
                .await
                .expect("ply produced no messages")
                .expect("ply parse");
            let n = msg.data.num_splats();
            let splats = msg
                .data
                .into_splats::<DiffBackend>(&device, SplatRenderMode::Default);
            (splats, n)
        })
    } else {
        let s = gen_splats_random(&device, splat_count, 42);
        (s, splat_count)
    };
    eprintln!(
        "perf: {splat_count_actual} splats, sh_degree={}, {width}x{height}, \
         {iters} measured iters ({warmup} warmup)",
        splats.sh_degree()
    );

    let (centroid, radius) = synthetic_camera_for(&splats);
    let batches = make_orbit_batches(centroid, radius, 8, (width, height));

    let mut config = TrainConfig::default();
    // Skip SH warmup so the bench measures the post-warmup step time,
    // which is the dominant regime for a real training run (5000 of 30k
    // iters are warmup; the remaining 25k iters are this regime).
    config.sh_warmup_iters = 0;
    let config = config;
    let bounds = BoundingBox::from_min_max(
        centroid - Vec3::splat(radius),
        centroid + Vec3::splat(radius),
    );

    let mut splats = splats;
    let mut trainer = SplatTrainer::new(&config, &device, bounds);

    for step in 0..warmup {
        let batch = batches[step % batches.len()].clone();
        let (new_splats, _) = block_on(trainer.step(batch, splats));
        splats = new_splats;
    }
    MainBackend::sync(&device).expect("sync");

    enabled.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut times_ms = Vec::with_capacity(iters);
    for step in 0..iters {
        let batch = batches[step % batches.len()].clone();
        let start = Instant::now();
        let (new_splats, _) = block_on(trainer.step(batch, splats));
        splats = new_splats;
        MainBackend::sync(&device).expect("sync");
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    enabled.store(false, std::sync::atomic::Ordering::Relaxed);

    if args.iter().any(|a| a == "--isolated") {
        run_isolated_phases(&device, &splats, &batches[0], iters);
    }

    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let p50 = percentile(&sorted, 0.5);
    let p95 = percentile(&sorted, 0.95);
    let mean: f64 = sorted.iter().sum::<f64>() / sorted.len() as f64;
    println!(
        "min {min:6.2} ms  p50 {p50:6.2} ms  p95 {p95:6.2} ms  mean {mean:6.2} ms  \
         (n={})",
        sorted.len()
    );

    if show_phases {
        let table = table.lock().unwrap();
        let mut entries: Vec<(String, f64, usize)> = table
            .by_name
            .iter()
            .map(|(name, samples)| {
                let n = samples.len();
                let mean: f64 = samples.iter().sum::<f64>() / n as f64;
                (name.clone(), mean, n)
            })
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("\n{:36} {:>10} {:>6}", "span", "mean ms", "count");
        for (name, mean, n) in entries.iter().take(20) {
            if *mean < 0.05 {
                continue;
            }
            println!("{name:36} {mean:10.3} {n:6}");
        }
    }
}
