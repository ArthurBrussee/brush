//! Correctness tests for the appearance kernels.
//!
//! Each fused kernel is checked against a plain-Rust CPU reference (ported
//! from the same CUDA/PyTorch sources the kernels were ported from), and
//! gradients are verified with central finite differences through the
//! forward-only paths.

use brush_appearance::GradSubsample;
use brush_appearance::bilagrid::{BilagridModel, bilagrid_apply, bilagrid_tv_loss};
use brush_appearance::ppisp::{PpispModel, PpispStages, ppisp_apply};
use burn::tensor::{Device, Tensor, TensorData};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

async fn ad_device() -> Device {
    burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

/// Deterministic pseudo-random floats in [lo, hi).
fn pattern(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    (0..n)
        .map(|_| {
            // PCG-ish hash; deterministic across platforms.
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            let h = (word >> 22) ^ word;
            lo + (hi - lo) * (h as f32 / u32::MAX as f32)
        })
        .collect()
}

async fn read_vec(t: Tensor<3>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .to_vec()
        .expect("vec")
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

const C2G: [f32; 3] = [0.299, 0.587, 0.114];

/// CPU port of the fused bilateral-grid slice (`LichtFeld` HWC forward).
#[allow(clippy::too_many_arguments)]
fn bilagrid_slice_cpu(
    grid: &[f32], // [12, L, H, W] (single view)
    rgb: &[f32],  // [h, w, 3]
    gl: usize,
    gh: usize,
    gw: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * 3];
    let cell = gl * gh * gw;
    for hi in 0..h {
        for wi in 0..w {
            let base = (hi * w + wi) * 3;
            let (sr, sg, sb) = (rgb[base], rgb[base + 1], rgb[base + 2]);
            let x = wi as f32 / (w - 1).max(1) as f32 * (gw - 1) as f32;
            let y = hi as f32 / (h - 1).max(1) as f32 * (gh - 1) as f32;
            let z = (C2G[0] * sr + C2G[1] * sg + C2G[2] * sb) * (gl - 1) as f32;

            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let y1 = (y0 + 1).min(gh - 1);
            let z0i = z.floor() as i64;
            let z0 = z0i.clamp(0, gl as i64 - 1) as usize;
            let z1 = (z0i + 1).clamp(0, gl as i64 - 1) as usize;
            let fx = x - x.floor();
            let fy = y - y.floor();
            let fz = z - z0 as f32;

            let mut acc = [0.0f32; 3];
            for ci in 0..12 {
                let g =
                    |zz: usize, yy: usize, xx: usize| grid[ci * cell + (zz * gh + yy) * gw + xx];
                let c00 = g(z0, y0, x0) * (1.0 - fx) + g(z0, y0, x1) * fx;
                let c01 = g(z0, y1, x0) * (1.0 - fx) + g(z0, y1, x1) * fx;
                let c10 = g(z1, y0, x0) * (1.0 - fx) + g(z1, y0, x1) * fx;
                let c11 = g(z1, y1, x0) * (1.0 - fx) + g(z1, y1, x1) * fx;
                let c0 = c00 * (1.0 - fy) + c01 * fy;
                let c1 = c10 * (1.0 - fy) + c11 * fy;
                let val = c0 * (1.0 - fz) + c1 * fz;
                let si = ci % 4;
                let di = ci / 4;
                let coeff = match si {
                    0 => sr,
                    1 => sg,
                    2 => sb,
                    _ => 1.0,
                };
                acc[di] += val * coeff;
            }
            out[base] = acc[0];
            out[base + 1] = acc[1];
            out[base + 2] = acc[2];
        }
    }
    out
}

/// CPU total-variation loss matching gsplat's `total_variation_loss`.
fn bilagrid_tv_cpu(grids: &[f32], n: usize, gl: usize, gh: usize, gw: usize) -> f32 {
    let cell = gl * gh * gw;
    let mut tv = 0.0f64;
    let at = |ni: usize, ci: usize, li: usize, hi: usize, wi: usize| {
        grids[((ni * 12 + ci) * gl + li) * gh * gw + hi * gw + wi] as f64
    };
    for ni in 0..n {
        for ci in 0..12 {
            for li in 0..gl {
                for hi in 0..gh {
                    for wi in 0..gw {
                        let v = at(ni, ci, li, hi, wi);
                        if wi > 0 {
                            let d = v - at(ni, ci, li, hi, wi - 1);
                            tv += d * d / (gl * gh * (gw - 1)) as f64;
                        }
                        if hi > 0 {
                            let d = v - at(ni, ci, li, hi - 1, wi);
                            tv += d * d / (gl * (gh - 1) * gw) as f64;
                        }
                        if li > 0 {
                            let d = v - at(ni, ci, li - 1, hi, wi);
                            tv += d * d / ((gl - 1) * gh * gw) as f64;
                        }
                    }
                }
            }
        }
    }
    let _ = cell;
    (tv / (12 * n) as f64) as f32
}

/// CPU port of the PPISP reference pipeline (`torch_reference.py`).
#[allow(clippy::too_many_arguments)]
fn ppisp_cpu(
    exposure: f32,
    vignetting: &[f32], // [3, 5] for the active camera
    color: &[f32],      // [8] for the active frame
    crf: &[f32],        // [3, 4] for the active camera
    rgb: &[f32],        // [h, w, 3]
    h: usize,
    w: usize,
    apply_crf: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * 3];

    // Homography from latent color params.
    let zca: [[f32; 4]; 4] = [
        [0.0480542, -0.0043631, -0.0043631, 0.0481283],
        [0.0580570, -0.0179872, -0.0179872, 0.0431061],
        [0.0433336, -0.0180537, -0.0180537, 0.0580500],
        [0.0128369, -0.0034654, -0.0034654, 0.0128158],
    ];
    let off = |i: usize| {
        let (l0, l1) = (color[i * 2], color[i * 2 + 1]);
        [
            zca[i][0] * l0 + zca[i][1] * l1,
            zca[i][2] * l0 + zca[i][3] * l1,
        ]
    };
    let (bd, rd, gd, nd) = (off(0), off(1), off(2), off(3));
    let t_b = [bd[0], bd[1], 1.0];
    let t_r = [1.0 + rd[0], rd[1], 1.0];
    let t_g = [gd[0], 1.0 + gd[1], 1.0];
    let t_n = [1.0 / 3.0 + nd[0], 1.0 / 3.0 + nd[1], 1.0];

    // Row-major matrices.
    let t = [
        [t_b[0], t_r[0], t_g[0]],
        [t_b[1], t_r[1], t_g[1]],
        [t_b[2], t_r[2], t_g[2]],
    ];
    let skew = [
        [0.0, -t_n[2], t_n[1]],
        [t_n[2], 0.0, -t_n[0]],
        [-t_n[1], t_n[0], 0.0],
    ];
    let matmul = |a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]| {
        let mut c = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for (k, bk) in b.iter().enumerate() {
                    c[i][j] += a[i][k] * bk[j];
                }
            }
        }
        c
    };
    let m = matmul(&skew, &t);
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut lam = cross(m[0], m[1]);
    if dot3(lam, lam) < 1.0e-20 {
        lam = cross(m[0], m[2]);
        if dot3(lam, lam) < 1.0e-20 {
            lam = cross(m[1], m[2]);
        }
    }
    let d = [[lam[0], 0.0, 0.0], [0.0, lam[1], 0.0], [0.0, 0.0, lam[2]]];
    let s_inv = [[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut hmat = matmul(&matmul(&t, &d), &s_inv);
    let s = hmat[2][2];
    if s.abs() > 1.0e-20 {
        for row in &mut hmat {
            for v in row.iter_mut() {
                *v /= s;
            }
        }
    }

    let softplus = |x: f32| x.exp().ln_1p();
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

    for hi in 0..h {
        for wi in 0..w {
            let base = (hi * w + wi) * 3;
            let mut c = [rgb[base], rgb[base + 1], rgb[base + 2]];

            // Exposure.
            let factor = 2.0f32.powf(exposure);
            for v in &mut c {
                *v *= factor;
            }

            // Vignetting.
            let max_res = (w as f32).max(h as f32);
            let uvx = (wi as f32 + 0.5 - w as f32 * 0.5) / max_res;
            let uvy = (hi as f32 + 0.5 - h as f32 * 0.5) / max_res;
            for ch in 0..3 {
                let p = &vignetting[ch * 5..ch * 5 + 5];
                let dx = uvx - p[0];
                let dy = uvy - p[1];
                let r2 = dx * dx + dy * dy;
                let falloff =
                    (1.0 + p[2] * r2 + p[3] * r2 * r2 + p[4] * r2 * r2 * r2).clamp(0.0, 1.0);
                c[ch] *= falloff;
            }

            // Color homography in RGI space.
            let intensity = c[0] + c[1] + c[2];
            let rgi_in = [c[0], c[1], intensity];
            let mut rgi = [0.0f32; 3];
            for i in 0..3 {
                rgi[i] = dot3(hmat[i], rgi_in);
            }
            let norm = intensity / (rgi[2] + 1.0e-5);
            for v in &mut rgi {
                *v *= norm;
            }
            c = [rgi[0], rgi[1], rgi[2] - rgi[0] - rgi[1]];

            // CRF.
            for ch in 0..3 {
                if !apply_crf {
                    continue;
                }
                let p = &crf[ch * 4..ch * 4 + 4];
                let toe = 0.3 + softplus(p[0]);
                let shoulder = 0.3 + softplus(p[1]);
                let gamma = 0.1 + softplus(p[2]);
                let center = sigmoid(p[3]);
                let x = c[ch].clamp(0.0, 1.0);
                let lerp_val = toe + center * (shoulder - toe);
                let a = shoulder * center / lerp_val;
                let b = 1.0 - a;
                let y = if x <= center {
                    a * (x / center).powf(toe)
                } else {
                    1.0 - b * ((1.0 - x) / (1.0 - center)).powf(shoulder)
                };
                c[ch] = y.max(0.0).powf(gamma);
            }

            out[base] = c[0];
            out[base + 1] = c[1];
            out[base + 2] = c[2];
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let mut max_err = 0.0f32;
    let mut max_at = 0;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let err = (a - e).abs();
        if err > max_err {
            max_err = err;
            max_at = i;
        }
    }
    assert!(
        max_err <= tol,
        "{what}: max err {max_err} at {max_at} (got {}, want {}), tol {tol}",
        actual[max_at],
        expected[max_at]
    );
}

// ---------------------------------------------------------------------------
// Bilateral grid tests
// ---------------------------------------------------------------------------

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilagrid_identity_grid_is_noop() {
    let device = ad_device().await;
    let (h, w) = (17, 23);
    let model = BilagridModel::new(3, 8, 8, 4, &device);
    let rgb_data = pattern(h * w * 3, 7, 0.0, 1.0);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([h, w, 3]);

    let out = model.apply(rgb.clone(), 1);
    let out_v = read_vec(out).await;
    assert_close(&out_v, &rgb_data, 1e-5, "identity bilagrid");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilagrid_slice_matches_cpu_reference() {
    let device = ad_device().await;
    let (h, w) = (15, 21);
    let (gx, gy, gl) = (6, 5, 4);
    let n = 2;
    let view = 1;

    let grids_data = pattern(n * 12 * gl * gy * gx, 3, -0.6, 1.2);
    let rgb_data = pattern(h * w * 3, 11, 0.0, 1.0);

    let grids =
        Tensor::<1>::from_floats(grids_data.as_slice(), &device).reshape([n, 12, gl, gy, gx]);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([h, w, 3]);

    let out = bilagrid_apply(grids, rgb, view, GradSubsample::default());
    let out_v = read_vec(out).await;

    let view_grid = &grids_data[view * 12 * gl * gy * gx..(view + 1) * 12 * gl * gy * gx];
    let expected = bilagrid_slice_cpu(view_grid, &rgb_data, gl, gy, gx, h, w);
    assert_close(&out_v, &expected, 1e-4, "bilagrid slice vs cpu");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilagrid_alpha_passthrough() {
    let device = ad_device().await;
    let (h, w) = (9, 13);
    let model = BilagridModel::new(1, 4, 4, 2, &device);
    let rgba_data = pattern(h * w * 4, 5, 0.0, 1.0);
    let rgba = Tensor::<1>::from_floats(rgba_data.as_slice(), &device).reshape([h, w, 4]);

    let out = model.apply(rgba, 0);
    let out_v = read_vec(out).await;
    for i in 0..h * w {
        let a_in = rgba_data[i * 4 + 3];
        let a_out = out_v[i * 4 + 3];
        assert!(
            (a_in - a_out).abs() < 1e-6,
            "alpha not passed through at {i}: {a_in} vs {a_out}"
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilagrid_tv_matches_cpu_reference() {
    let device = ad_device().await;
    let (gx, gy, gl) = (5, 4, 3);
    let n = 3;
    let grids_data = pattern(n * 12 * gl * gy * gx, 13, -1.0, 1.0);
    let grids =
        Tensor::<1>::from_floats(grids_data.as_slice(), &device).reshape([n, 12, gl, gy, gx]);

    let tv = bilagrid_tv_loss(grids)
        .into_data_async()
        .await
        .expect("readback")
        .to_vec::<f32>()
        .expect("vec")[0];
    let expected = bilagrid_tv_cpu(&grids_data, n, gl, gy, gx);
    assert!(
        (tv - expected).abs() < 1e-4 * expected.abs().max(1.0),
        "tv {tv} vs cpu {expected}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilagrid_gradients_match_finite_differences() {
    let device = ad_device().await;
    let (h, w) = (9, 11);
    let (gx, gy, gl) = (4, 4, 3);
    let n = 2;
    let view = 0;

    let grids_data = pattern(n * 12 * gl * gy * gx, 17, -0.4, 1.1);
    let rgb_data = pattern(h * w * 3, 19, 0.05, 0.95);
    // Loss weights make dL/dout vary per pixel.
    let wts_data = pattern(h * w * 3, 23, -1.0, 1.0);

    let make_grids = |d: &[f32]| Tensor::<1>::from_floats(d, &device).reshape([n, 12, gl, gy, gx]);
    let make_rgb = |d: &[f32]| Tensor::<1>::from_floats(d, &device).reshape([h, w, 3]);
    let wts = Tensor::<1>::from_floats(wts_data.as_slice(), &device).reshape([h, w, 3]);

    // Analytic gradients.
    let grids = make_grids(&grids_data).require_grad();
    let rgb = make_rgb(&rgb_data).require_grad();
    let out = bilagrid_apply(grids.clone(), rgb.clone(), view, GradSubsample::default());
    let loss = (out * wts.clone()).sum();
    let grads = loss.backward();
    let g_grids: Vec<f32> = grids
        .grad(&grads)
        .expect("grids grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("vec");
    let g_rgb: Vec<f32> = rgb
        .grad(&grads)
        .expect("rgb grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("vec");

    // Finite differences on a scattering of indices.
    let loss_for = |gd: &[f32], rd: &[f32]| {
        let out = bilagrid_apply(make_grids(gd), make_rgb(rd), view, GradSubsample::default());
        (out * wts.clone()).sum()
    };
    let eps = 2e-3;
    for idx in [0usize, 7, 101, 333, 12 * gl * gy * gx - 5] {
        let mut plus = grids_data.clone();
        plus[idx] += eps;
        let mut minus = grids_data.clone();
        minus[idx] -= eps;
        let lp = loss_for(&plus, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&minus, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - g_grids[idx]).abs() < 2e-2 * fd.abs().max(1.0),
            "grid grad mismatch at {idx}: analytic {} vs fd {fd}",
            g_grids[idx]
        );
    }
    for idx in [4usize, 50, 151] {
        let mut plus = rgb_data.clone();
        plus[idx] += eps;
        let mut minus = rgb_data.clone();
        minus[idx] -= eps;
        let lp = loss_for(&grids_data, &plus)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&grids_data, &minus)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - g_rgb[idx]).abs() < 2e-2 * fd.abs().max(1.0),
            "rgb grad mismatch at {idx}: analytic {} vs fd {fd}",
            g_rgb[idx]
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn bilagrid_tv_gradients_match_finite_differences() {
    let device = ad_device().await;
    let (gx, gy, gl) = (4, 3, 3);
    let n = 2;
    let grids_data = pattern(n * 12 * gl * gy * gx, 29, -0.8, 0.8);
    let make_grids = |d: &[f32]| Tensor::<1>::from_floats(d, &device).reshape([n, 12, gl, gy, gx]);

    let grids = make_grids(&grids_data).require_grad();
    let loss = bilagrid_tv_loss(grids.clone());
    let grads = loss.backward();
    let g: Vec<f32> = grids
        .grad(&grads)
        .expect("grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("vec");

    let eps = 1e-2;
    for idx in [0usize, 13, 200, n * 12 * gl * gy * gx - 1] {
        let mut plus = grids_data.clone();
        plus[idx] += eps;
        let mut minus = grids_data.clone();
        minus[idx] -= eps;
        let lp = bilagrid_tv_loss(make_grids(&plus))
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = bilagrid_tv_loss(make_grids(&minus))
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - g[idx]).abs() < 2e-2 * fd.abs().max(0.1),
            "tv grad mismatch at {idx}: analytic {} vs fd {fd}",
            g[idx]
        );
    }
}

// ---------------------------------------------------------------------------
// PPISP tests
// ---------------------------------------------------------------------------

struct PpispTestData {
    exposure: Vec<f32>,
    vignetting: Vec<f32>,
    color: Vec<f32>,
    crf: Vec<f32>,
    rgb: Vec<f32>,
    num_frames: usize,
    num_cameras: usize,
    h: usize,
    w: usize,
}

fn ppisp_test_data(h: usize, w: usize) -> PpispTestData {
    let num_frames = 3;
    let num_cameras = 2;
    let mut crf = pattern(num_cameras * 12, 41, -0.3, 0.3);
    // Bias CRF toward the identity-ish init region.
    for (i, v) in crf.iter_mut().enumerate() {
        match i % 4 {
            0 | 1 => *v += 0.0137,
            2 => *v += 0.3781,
            _ => {}
        }
    }
    // Vignetting: small centers, clearly-negative alphas. Keeps the falloff
    // strictly inside the (0, 1) clamp so finite differences don't straddle
    // the clamp's gradient gate (the subgradient there matches the CUDA
    // reference but is not FD-measurable).
    let centers = pattern(num_cameras * 6, 36, -0.1, 0.1);
    let alphas = pattern(num_cameras * 9, 37, -0.5, -0.05);
    let mut vignetting = Vec::with_capacity(num_cameras * 15);
    for i in 0..num_cameras * 3 {
        vignetting.extend_from_slice(&centers[i * 2..i * 2 + 2]);
        vignetting.extend_from_slice(&alphas[i * 3..i * 3 + 3]);
    }
    PpispTestData {
        exposure: pattern(num_frames, 31, -0.3, 0.3),
        vignetting,
        color: pattern(num_frames * 8, 39, -0.5, 0.5),
        crf,
        rgb: pattern(h * w * 3, 43, 0.05, 0.95),
        num_frames,
        num_cameras,
        h,
        w,
    }
}

#[allow(clippy::type_complexity)]
fn ppisp_tensors(
    d: &PpispTestData,
    device: &Device,
) -> (Tensor<1>, Tensor<3>, Tensor<2>, Tensor<3>, Tensor<3>) {
    (
        Tensor::<1>::from_floats(d.exposure.as_slice(), device),
        Tensor::<1>::from_floats(d.vignetting.as_slice(), device).reshape([d.num_cameras, 3, 5]),
        Tensor::<1>::from_floats(d.color.as_slice(), device).reshape([d.num_frames, 8]),
        Tensor::<1>::from_floats(d.crf.as_slice(), device).reshape([d.num_cameras, 3, 4]),
        Tensor::<1>::from_floats(d.rgb.as_slice(), device).reshape([d.h, d.w, 3]),
    )
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_identity_init_is_near_noop() {
    let device = ad_device().await;
    let (h, w) = (12, 16);
    let model = PpispModel::new(1, 2, vec![0, 0], &device);
    let rgb_data = pattern(h * w * 3, 47, 0.05, 0.95);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([h, w, 3]);

    let out = model.apply(rgb, 1);
    let out_v = read_vec(out).await;
    // Zero-init exposure/vignetting/color and identity CRF — output should
    // match the input closely (CRF identity init is exact: toe=shoulder=
    // gamma=1, center=0.5 gives y = x).
    assert_close(&out_v, &rgb_data, 1e-4, "ppisp identity");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_matches_cpu_reference() {
    let device = ad_device().await;
    let (h, w) = (13, 19);
    let d = ppisp_test_data(h, w);
    let (exposure, vignetting, color, crf, rgb) = ppisp_tensors(&d, &device);
    let (camera_idx, frame_idx) = (1usize, 2usize);

    let out = ppisp_apply(
        exposure,
        vignetting,
        color,
        crf,
        rgb,
        camera_idx,
        frame_idx,
        PpispStages::ALL,
    );
    let out_v = read_vec(out).await;

    let expected = ppisp_cpu(
        d.exposure[frame_idx],
        &d.vignetting[camera_idx * 15..camera_idx * 15 + 15],
        &d.color[frame_idx * 8..frame_idx * 8 + 8],
        &d.crf[camera_idx * 12..camera_idx * 12 + 12],
        &d.rgb,
        h,
        w,
        true,
    );
    assert_close(&out_v, &expected, 1e-4, "ppisp vs cpu");
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_gradients_match_finite_differences() {
    let device = ad_device().await;
    let (h, w) = (11, 14);
    let d = ppisp_test_data(h, w);
    let (camera_idx, frame_idx) = (0usize, 1usize);
    let wts_data = pattern(h * w * 3, 53, -1.0, 1.0);
    let wts = Tensor::<1>::from_floats(wts_data.as_slice(), &device).reshape([h, w, 3]);

    // Analytic gradients.
    let (exposure, vignetting, color, crf, rgb) = ppisp_tensors(&d, &device);
    let exposure = exposure.require_grad();
    let vignetting = vignetting.require_grad();
    let color = color.require_grad();
    let crf = crf.require_grad();
    let rgb = rgb.require_grad();
    let out = ppisp_apply(
        exposure.clone(),
        vignetting.clone(),
        color.clone(),
        crf.clone(),
        rgb.clone(),
        camera_idx,
        frame_idx,
        PpispStages::ALL,
    );
    let loss = (out * wts.clone()).sum();
    let grads = loss.backward();

    let read1 = |t: Option<Tensor<1>>| t.expect("grad");
    let g_exp: Vec<f32> = read1(exposure.grad(&grads))
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");
    let g_vig: Vec<f32> = vignetting
        .grad(&grads)
        .expect("grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");
    let g_color: Vec<f32> = color
        .grad(&grads)
        .expect("grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");
    let g_crf: Vec<f32> = crf
        .grad(&grads)
        .expect("grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");
    let g_rgb: Vec<f32> = rgb
        .grad(&grads)
        .expect("grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");

    // Gradients for non-active frames/cameras must be zero.
    for f in 0..d.num_frames {
        if f != frame_idx {
            assert_eq!(g_exp[f], 0.0, "inactive frame {f} exposure grad");
            for k in 0..8 {
                assert_eq!(g_color[f * 8 + k], 0.0, "inactive frame color grad");
            }
        }
    }
    for c in 0..d.num_cameras {
        if c != camera_idx {
            for k in 0..15 {
                assert_eq!(g_vig[c * 15 + k], 0.0, "inactive camera vig grad");
            }
            for k in 0..12 {
                assert_eq!(g_crf[c * 12 + k], 0.0, "inactive camera crf grad");
            }
        }
    }

    // Finite differences through the forward (f64-accumulated CPU loss for
    // stability of the comparison would be ideal; GPU forward is enough at
    // these sizes).
    let loss_for = |e: &[f32], v: &[f32], co: &[f32], cr: &[f32], r: &[f32]| {
        let exposure = Tensor::<1>::from_floats(e, &device);
        let vignetting = Tensor::<1>::from_floats(v, &device).reshape([d.num_cameras, 3, 5]);
        let color = Tensor::<1>::from_floats(co, &device).reshape([d.num_frames, 8]);
        let crf = Tensor::<1>::from_floats(cr, &device).reshape([d.num_cameras, 3, 4]);
        let rgb = Tensor::<1>::from_floats(r, &device).reshape([h, w, 3]);
        let out = ppisp_apply(
            exposure,
            vignetting,
            color,
            crf,
            rgb,
            camera_idx,
            frame_idx,
            PpispStages::ALL,
        );
        (out * wts.clone()).sum()
    };

    let eps = 1e-3;
    let check = |name: &str, analytic: f32, fd: f32| {
        assert!(
            (fd - analytic).abs() < 3e-2 * fd.abs().max(0.5),
            "{name}: analytic {analytic} vs fd {fd}"
        );
    };

    // Exposure (active frame).
    {
        let mut p = d.exposure.clone();
        p[frame_idx] += eps;
        let mut m = d.exposure.clone();
        m[frame_idx] -= eps;
        let lp = loss_for(&p, &d.vignetting, &d.color, &d.crf, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&m, &d.vignetting, &d.color, &d.crf, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        check("exposure", g_exp[frame_idx], (lp - lm) / (2.0 * eps));
    }
    // Vignetting (every param of the active camera).
    for k in 0..15 {
        let idx = camera_idx * 15 + k;
        let mut p = d.vignetting.clone();
        p[idx] += eps;
        let mut m = d.vignetting.clone();
        m[idx] -= eps;
        let lp = loss_for(&d.exposure, &p, &d.color, &d.crf, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&d.exposure, &m, &d.color, &d.crf, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        check(&format!("vig[{k}]"), g_vig[idx], (lp - lm) / (2.0 * eps));
    }
    // Color (every latent of the active frame).
    for k in 0..8 {
        let idx = frame_idx * 8 + k;
        let mut p = d.color.clone();
        p[idx] += eps;
        let mut m = d.color.clone();
        m[idx] -= eps;
        let lp = loss_for(&d.exposure, &d.vignetting, &p, &d.crf, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&d.exposure, &d.vignetting, &m, &d.crf, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        check(
            &format!("color[{k}]"),
            g_color[idx],
            (lp - lm) / (2.0 * eps),
        );
    }
    // CRF (every raw param of the active camera).
    for k in 0..12 {
        let idx = camera_idx * 12 + k;
        let mut p = d.crf.clone();
        p[idx] += eps;
        let mut m = d.crf.clone();
        m[idx] -= eps;
        let lp = loss_for(&d.exposure, &d.vignetting, &d.color, &p, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&d.exposure, &d.vignetting, &d.color, &m, &d.rgb)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        check(&format!("crf[{k}]"), g_crf[idx], (lp - lm) / (2.0 * eps));
    }
    // RGB (a few pixels).
    for idx in [0usize, 77, 311] {
        let mut p = d.rgb.clone();
        p[idx] += eps;
        let mut m = d.rgb.clone();
        m[idx] -= eps;
        let lp = loss_for(&d.exposure, &d.vignetting, &d.color, &d.crf, &p)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&d.exposure, &d.vignetting, &d.color, &d.crf, &m)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        check(&format!("rgb[{idx}]"), g_rgb[idx], (lp - lm) / (2.0 * eps));
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_reg_loss_zero_at_init_and_positive_off_init() {
    let device = ad_device().await;
    let model = PpispModel::new(2, 4, vec![0, 0, 1, 1], &device);
    let loss = model
        .reg_loss()
        .into_scalar_async::<f32>()
        .await
        .expect("loss");
    // At init: exposure/color/vignetting are zero and CRF is identical
    // across channels → every term vanishes.
    assert!(
        loss.abs() < 1e-6,
        "reg loss at init should be 0, got {loss}"
    );

    // Perturbed params produce a positive loss.
    let mut model = model;
    model.exposure =
        burn::module::Param::from_tensor(Tensor::<1>::from_floats([0.5, 0.4, 0.3, 0.2], &device));
    let loss = model
        .reg_loss()
        .into_scalar_async::<f32>()
        .await
        .expect("loss");
    assert!(
        loss > 1e-4,
        "reg loss should penalise exposure mean, got {loss}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_module_keeps_camera_mapping() {
    let device = ad_device().await;
    let model = PpispModel::new(2, 3, vec![0, 1, 1], &device);
    let _ = TensorData::new(vec![0.0f32], [1]);
    assert_eq!(model.camera_indices, vec![0, 1, 1]);
}

/// Rough kernel timing split (run with --nocapture --ignored).
#[tokio::test]
#[ignore = "manual benchmark"]
async fn bench_bilagrid_kernels() {
    let device = ad_device().await;
    let (h, w) = (1080, 1080);
    let (gx, gy, gl) = (16, 16, 8);
    let n = 1;
    let grids_data = pattern(n * 12 * gl * gy * gx, 3, -0.5, 1.0);
    let rgb_data = pattern(h * w * 4, 11, 0.0, 1.0);
    let grids =
        Tensor::<1>::from_floats(grids_data.as_slice(), &device).reshape([n, 12, gl, gy, gx]);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([h, w, 4]);

    let sync = |t: Tensor<3>| async move {
        let _ = t
            .slice(burn::tensor::s![0..1, 0..1])
            .into_data_async()
            .await;
    };

    // Warmup.
    for _ in 0..10 {
        let out = bilagrid_apply(grids.clone(), rgb.clone(), 0, GradSubsample::default());
        sync(out).await;
    }
    let iters = 100;

    // Forward only.
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let out = bilagrid_apply(grids.clone(), rgb.clone(), 0, GradSubsample::default());
        std::hint::black_box(&out);
    }
    let out = bilagrid_apply(grids.clone(), rgb.clone(), 0, GradSubsample::default());
    sync(out).await;
    println!(
        "fwd: {:.3} ms/iter",
        start.elapsed().as_secs_f64() * 1000.0 / (iters + 1) as f64
    );

    // Forward + backward.
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let g = grids.clone().require_grad();
        let r = rgb.clone().require_grad();
        let out = bilagrid_apply(g, r, 0, GradSubsample::default());
        let loss = out.sum();
        let _grads = loss.backward();
    }
    let g = grids.clone().require_grad();
    let out = bilagrid_apply(g.clone(), rgb.clone(), 0, GradSubsample::default());
    let loss = out.sum();
    let grads = loss.backward();
    let _ = g.grad(&grads).unwrap().into_data_async().await;
    println!(
        "fwd+bwd: {:.3} ms/iter",
        start.elapsed().as_secs_f64() * 1000.0 / (iters + 1) as f64
    );

    // Raw backward kernel only (no autodiff plumbing).
    let rgb_smooth_data: Vec<f32> = (0..h * w * 4)
        .map(|i| {
            let px = (i / 4) % w;
            let py = (i / 4) / w;
            (px + py) as f32 / (w + h) as f32
        })
        .collect();
    let rgb_smooth =
        Tensor::<1>::from_floats(rgb_smooth_data.as_slice(), &device).reshape([h, w, 4]);
    for (name, img) in [("noise", rgb.clone()), ("smooth", rgb_smooth)] {
        use brush_appearance::bilagrid::BilagridOps;
        use brush_cube::MainBackend;
        use brush_render::burn_glue::{unwrap_ad_wgpu_float, wrap_wgpu_float};
        let grids_p = unwrap_ad_wgpu_float(grids.clone()).primitive;
        let rgb_p = unwrap_ad_wgpu_float(img.clone()).primitive;
        let v_out_p = unwrap_ad_wgpu_float(img.clone()).primitive;
        // Warmup
        for _ in 0..5 {
            let (gg, _gr) = <MainBackend as BilagridOps<MainBackend>>::bilagrid_slice_bwd(
                grids_p.clone(),
                rgb_p.clone(),
                v_out_p.clone(),
                0,
                GradSubsample::default(),
            );
            let t: Tensor<5> = wrap_wgpu_float(gg);
            let _ = t
                .slice(burn::tensor::s![0..1, 0..1, 0..1, 0..1, 0..1])
                .into_data_async()
                .await;
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let (gg, _gr) = <MainBackend as BilagridOps<MainBackend>>::bilagrid_slice_bwd(
                grids_p.clone(),
                rgb_p.clone(),
                v_out_p.clone(),
                0,
                GradSubsample::default(),
            );
            std::hint::black_box(&gg);
        }
        let (gg, _gr) = <MainBackend as BilagridOps<MainBackend>>::bilagrid_slice_bwd(
            grids_p.clone(),
            rgb_p.clone(),
            v_out_p.clone(),
            0,
            GradSubsample::default(),
        );
        let t: Tensor<5> = wrap_wgpu_float(gg);
        let _ = t
            .slice(burn::tensor::s![0..1, 0..1, 0..1, 0..1, 0..1])
            .into_data_async()
            .await;
        println!(
            "raw bwd kernel ({name}): {:.3} ms/iter",
            start.elapsed().as_secs_f64() * 1000.0 / (iters + 1) as f64
        );
    }

    // TV fwd+bwd on a 200-view grid.
    let big = Tensor::<5>::zeros([200, 12, gl, gy, gx], &device);
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let g = big.clone().require_grad();
        let loss = bilagrid_tv_loss(g);
        let _grads = loss.backward();
    }
    let g = big.clone().require_grad();
    let loss = bilagrid_tv_loss(g.clone());
    let grads = loss.backward();
    let _ = g.grad(&grads).unwrap().into_data_async().await;
    println!(
        "tv (200 views) fwd+bwd: {:.3} ms/iter",
        start.elapsed().as_secs_f64() * 1000.0 / (iters + 1) as f64
    );
}

// ---------------------------------------------------------------------------
// Hybrid PPISP-grid tests
// ---------------------------------------------------------------------------

use brush_appearance::ppisp_grid::{GridPayload, ppisp_grid_apply};

const CRF_IDENT: [f32; 4] = [0.013_658_988, 0.013_658_988, 0.378_164_53, 0.0];

/// CPU reference: per-camera vignetting, then slice a C-channel payload
/// grid at the vignetted color's guidance, then exposure → color → CRF
/// (per the payload flags).
// tuple_array_conversions: clippy 1.93 false-positives on `[sr, sg, sb]`.
#[allow(clippy::too_many_arguments, clippy::tuple_array_conversions)]
fn ppisp_grid_cpu(
    grid: &[f32],       // [C, L, H, W] single view
    vignetting: &[f32], // [3, 5] for the active camera
    rgb: &[f32],        // [h, w, 3]
    payload: GridPayload,
    gl: usize,
    gh: usize,
    gw: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let pc = payload.channels();
    let cell = gl * gh * gw;
    let mut out = vec![0.0f32; h * w * 3];
    for hi in 0..h {
        for wi in 0..w {
            let base = (hi * w + wi) * 3;
            let (mut sr, mut sg, mut sb) = (rgb[base], rgb[base + 1], rgb[base + 2]);
            if payload.vignetting {
                let max_res = (w as f32).max(h as f32);
                let uvx = (wi as f32 + 0.5 - w as f32 * 0.5) / max_res;
                let uvy = (hi as f32 + 0.5 - h as f32 * 0.5) / max_res;
                let mut c = [sr, sg, sb];
                for (ch, v) in c.iter_mut().enumerate() {
                    let p = &vignetting[ch * 5..ch * 5 + 5];
                    let dx = uvx - p[0];
                    let dy = uvy - p[1];
                    let r2 = dx * dx + dy * dy;
                    let falloff =
                        (1.0 + p[2] * r2 + p[3] * r2 * r2 + p[4] * r2 * r2 * r2).clamp(0.0, 1.0);
                    *v *= falloff;
                }
                sr = c[0];
                sg = c[1];
                sb = c[2];
            }
            let x = wi as f32 / (w - 1).max(1) as f32 * (gw - 1) as f32;
            let y = hi as f32 / (h - 1).max(1) as f32 * (gh - 1) as f32;
            let z = (C2G[0] * sr + C2G[1] * sg + C2G[2] * sb) * (gl - 1) as f32;
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let y1 = (y0 + 1).min(gh - 1);
            let z0i = z.floor() as i64;
            let z0 = z0i.clamp(0, gl as i64 - 1) as usize;
            let z1 = (z0i + 1).clamp(0, gl as i64 - 1) as usize;
            let fx = x - x.floor();
            let fy = y - y.floor();
            let fz = z - z0 as f32;

            let mut p = vec![0.0f32; pc];
            for (ci, pv) in p.iter_mut().enumerate() {
                let g =
                    |zz: usize, yy: usize, xx: usize| grid[ci * cell + (zz * gh + yy) * gw + xx];
                let c00 = g(z0, y0, x0) * (1.0 - fx) + g(z0, y0, x1) * fx;
                let c01 = g(z0, y1, x0) * (1.0 - fx) + g(z0, y1, x1) * fx;
                let c10 = g(z1, y0, x0) * (1.0 - fx) + g(z1, y0, x1) * fx;
                let c11 = g(z1, y1, x0) * (1.0 - fx) + g(z1, y1, x1) * fx;
                let c0 = c00 * (1.0 - fy) + c01 * fy;
                let c1 = c10 * (1.0 - fy) + c11 * fy;
                *pv = c0 * (1.0 - fz) + c1 * fz;
            }

            // Build a full PPISP param set: exposure + (color) + (CRF with
            // identity offsets), reuse the full-pipeline CPU reference with
            // vignetting zeroed via cx=cy=alphas=0 (falloff = 1).
            let exposure = p[0];
            let mut color = [0.0f32; 8];
            if payload.color {
                color.copy_from_slice(&p[1..9]);
            }
            let kb = if payload.color { 9 } else { 1 };
            let mut crf = [0.0f32; 12];
            for ch in 0..3 {
                for k in 0..4 {
                    crf[ch * 4 + k] =
                        CRF_IDENT[k] + if payload.crf { p[kb + ch * 4 + k] } else { 0.0 };
                }
            }
            // Vignetting was applied above (before slicing); the pipeline
            // here starts at the exposure stage. Without the CRF stage the
            // kernel does not clamp the output.
            let zero_vig = [0.0f32; 15];
            let sliced_rgb = [sr, sg, sb];
            let px = ppisp_cpu(
                exposure,
                &zero_vig,
                &color,
                &crf,
                &sliced_rgb,
                1,
                1,
                payload.crf,
            );
            out[base] = px[0];
            out[base + 1] = px[1];
            out[base + 2] = px[2];
        }
    }
    out
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_grid_zero_is_identity() {
    let device = ad_device().await;
    let (h, w) = (14, 18);
    let (gx, gy, gl) = (6, 5, 4);
    let rgb_data = pattern(h * w * 3, 61, 0.05, 0.95);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([h, w, 3]);

    for payload in [
        GridPayload {
            color: false,
            crf: false,
            vignetting: false,
        },
        GridPayload {
            color: true,
            crf: false,
            vignetting: false,
        },
        GridPayload {
            color: true,
            crf: true,
            vignetting: true,
        },
    ] {
        let grids = Tensor::zeros([2, payload.channels(), gl, gy, gx], &device);
        // Zero vignetting params are an identity falloff.
        let vig = Tensor::zeros([1, 3, 5], &device);
        let out = ppisp_grid_apply(
            grids,
            vig,
            rgb.clone(),
            1,
            0,
            payload,
            GradSubsample::default(),
        );
        let out_v = read_vec(out).await;
        assert_close(&out_v, &rgb_data, 2e-4, &format!("identity {payload:?}"));
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_grid_matches_cpu_reference() {
    let device = ad_device().await;
    let (h, w) = (13, 17);
    let (gx, gy, gl) = (5, 4, 3);
    let n = 2;
    let view = 1;
    let rgb_data = pattern(h * w * 3, 67, 0.05, 0.95);
    let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([h, w, 3]);

    // Vignetting data shared by the vignetting-enabled payloads: small
    // centers, clearly-negative alphas (away from the falloff clamp).
    let centers = pattern(6, 36, -0.1, 0.1);
    let alphas = pattern(9, 37, -0.5, -0.05);
    let mut vig_data = vec![0.0f32; 15];
    for ch in 0..3 {
        vig_data[ch * 5] = centers[ch * 2];
        vig_data[ch * 5 + 1] = centers[ch * 2 + 1];
        vig_data[ch * 5 + 2..ch * 5 + 5].copy_from_slice(&alphas[ch * 3..ch * 3 + 3]);
    }

    for payload in [
        GridPayload {
            color: false,
            crf: false,
            vignetting: false,
        },
        GridPayload {
            color: true,
            crf: false,
            vignetting: false,
        },
        GridPayload {
            color: true,
            crf: false,
            vignetting: true,
        },
        GridPayload {
            color: true,
            crf: true,
            vignetting: true,
        },
    ] {
        let pc = payload.channels();
        // Small payload values keep the homography/CRF well-conditioned.
        let grids_data = pattern(n * pc * gl * gy * gx, 71, -0.25, 0.25);
        let grids =
            Tensor::<1>::from_floats(grids_data.as_slice(), &device).reshape([n, pc, gl, gy, gx]);
        let vig = Tensor::<1>::from_floats(vig_data.as_slice(), &device).reshape([1, 3, 5]);
        let out = ppisp_grid_apply(
            grids,
            vig,
            rgb.clone(),
            view,
            0,
            payload,
            GradSubsample::default(),
        );
        let out_v = read_vec(out).await;

        let view_grid = &grids_data[view * pc * gl * gy * gx..(view + 1) * pc * gl * gy * gx];
        let expected = ppisp_grid_cpu(view_grid, &vig_data, &rgb_data, payload, gl, gy, gx, h, w);
        assert_close(&out_v, &expected, 3e-4, &format!("grid vs cpu {payload:?}"));
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_grid_gradients_match_finite_differences() {
    let device = ad_device().await;
    let (h, w) = (9, 11);
    let (gx, gy, gl) = (4, 3, 3);
    let n = 1;
    let view = 0;
    let payload = GridPayload {
        color: true,
        crf: true,
        vignetting: true,
    };
    let pc = payload.channels();

    let grids_data = pattern(n * pc * gl * gy * gx, 73, -0.2, 0.2);
    let rgb_data = pattern(h * w * 3, 79, 0.1, 0.9);
    let wts_data = pattern(h * w * 3, 83, -1.0, 1.0);
    let wts = Tensor::<1>::from_floats(wts_data.as_slice(), &device).reshape([h, w, 3]);
    // Two cameras; camera 1 is active. Clearly-negative alphas keep the
    // falloff away from its clamp (where FD can't measure the subgradient).
    let cam = 1usize;
    let centers = pattern(12, 101, -0.1, 0.1);
    let alphas = pattern(18, 103, -0.5, -0.05);
    let mut vig_data = vec![0.0f32; 30];
    for k in 0..6 {
        vig_data[k * 5] = centers[k * 2];
        vig_data[k * 5 + 1] = centers[k * 2 + 1];
        vig_data[k * 5 + 2..k * 5 + 5].copy_from_slice(&alphas[k * 3..k * 3 + 3]);
    }

    let make_grids = |d: &[f32]| Tensor::<1>::from_floats(d, &device).reshape([n, pc, gl, gy, gx]);
    let make_rgb = |d: &[f32]| Tensor::<1>::from_floats(d, &device).reshape([h, w, 3]);
    let make_vig = |d: &[f32]| Tensor::<1>::from_floats(d, &device).reshape([2, 3, 5]);

    let grids = make_grids(&grids_data).require_grad();
    let rgb = make_rgb(&rgb_data).require_grad();
    let vig = make_vig(&vig_data).require_grad();
    let out = ppisp_grid_apply(
        grids.clone(),
        vig.clone(),
        rgb.clone(),
        view,
        cam,
        payload,
        GradSubsample::default(),
    );
    let loss = (out * wts.clone()).sum();
    let grads = loss.backward();
    let g_grids: Vec<f32> = grids
        .grad(&grads)
        .expect("grids grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");
    let g_rgb: Vec<f32> = rgb
        .grad(&grads)
        .expect("rgb grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");
    let g_vig: Vec<f32> = vig
        .grad(&grads)
        .expect("vig grad")
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");

    // The inactive camera's vignetting rows must have exactly zero grads.
    for k in 0..15 {
        assert_eq!(g_vig[k], 0.0, "inactive camera vig grad at {k}");
    }

    let loss_for = |gd: &[f32], vd: &[f32], rd: &[f32]| {
        let out = ppisp_grid_apply(
            make_grids(gd),
            make_vig(vd),
            make_rgb(rd),
            view,
            cam,
            payload,
            GradSubsample::default(),
        );
        (out * wts.clone()).sum()
    };
    let eps = 1e-3;
    // Sample indices across all payload sections: exposure (ch 0), color
    // (ch 1..9), CRF (ch 9..21).
    let cell = gl * gy * gx;
    for idx in [
        0usize,
        3 * cell + 5,
        10 * cell + 7,
        15 * cell + 2,
        20 * cell + 9,
    ] {
        let mut plus = grids_data.clone();
        plus[idx] += eps;
        let mut minus = grids_data.clone();
        minus[idx] -= eps;
        let lp = loss_for(&plus, &vig_data, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&minus, &vig_data, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - g_grids[idx]).abs() < 3e-2 * fd.abs().max(0.5),
            "grid grad mismatch at {idx}: analytic {} vs fd {fd}",
            g_grids[idx]
        );
    }
    // Every vignetting param of the active camera.
    for k in 0..15 {
        let idx = cam * 15 + k;
        let mut plus = vig_data.clone();
        plus[idx] += eps;
        let mut minus = vig_data.clone();
        minus[idx] -= eps;
        let lp = loss_for(&grids_data, &plus, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&grids_data, &minus, &rgb_data)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - g_vig[idx]).abs() < 3e-2 * fd.abs().max(0.5),
            "vig grad mismatch at {k}: analytic {} vs fd {fd}",
            g_vig[idx]
        );
    }
    for idx in [4usize, 50, 151] {
        let mut plus = rgb_data.clone();
        plus[idx] += eps;
        let mut minus = rgb_data.clone();
        minus[idx] -= eps;
        let lp = loss_for(&grids_data, &vig_data, &plus)
            .into_scalar_async::<f32>()
            .await
            .expect("lp");
        let lm = loss_for(&grids_data, &vig_data, &minus)
            .into_scalar_async::<f32>()
            .await
            .expect("lm");
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - g_rgb[idx]).abs() < 3e-2 * fd.abs().max(0.5),
            "rgb grad mismatch at {idx}: analytic {} vs fd {fd}",
            g_rgb[idx]
        );
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ppisp_grid_subsampled_gradients_are_unbiased() {
    // Planes elect themselves independently per seed (Bernoulli, p = 1/N,
    // scaled by N), so the mean over many seeds must converge to the
    // unsubsampled gradient. Deterministic: fixed seed list.
    let device = ad_device().await;
    let (h, w) = (32, 48);
    let (gx, gy, gl) = (4, 4, 3);
    let payload = GridPayload {
        color: true,
        crf: false,
        vignetting: false,
    };
    let pc = payload.channels();
    let grids_data = pattern(pc * gl * gy * gx, 89, -0.3, 0.3);
    let rgb_data = pattern(h * w * 3, 97, 0.05, 0.95);

    let grad_for = |sub: GradSubsample| {
        let grids = Tensor::<1>::from_floats(grids_data.as_slice(), &device)
            .reshape([1, pc, gl, gy, gx])
            .require_grad();
        let vig = Tensor::zeros([1, 3, 5], &device);
        let rgb = Tensor::<1>::from_floats(rgb_data.as_slice(), &device).reshape([h, w, 3]);
        let out = ppisp_grid_apply(grids.clone(), vig, rgb, 0, 0, payload, sub);
        let grads = out.sum().backward();
        grids.grad(&grads).expect("grad")
    };

    let full: Vec<f32> = grad_for(GradSubsample::default())
        .into_data_async()
        .await
        .expect("rb")
        .to_vec()
        .expect("v");
    let every = 4u32;
    let mut acc = vec![0.0f32; full.len()];
    let seeds = 64u32;
    for seed in 0..seeds {
        let g: Vec<f32> = grad_for(GradSubsample { every, seed })
            .into_data_async()
            .await
            .expect("rb")
            .to_vec()
            .expect("v");
        for (a, b) in acc.iter_mut().zip(&g) {
            *a += b / seeds as f32;
        }
    }
    // Relative L2 error of the seed-averaged gradient against dense.
    let num: f32 = acc.iter().zip(&full).map(|(a, b)| (a - b) * (a - b)).sum();
    let den: f32 = full.iter().map(|b| b * b).sum();
    let rel = (num / den.max(1e-12)).sqrt();
    assert!(
        rel < 0.1,
        "subsampled gradient biased: rel L2 err {rel:.4} over {seeds} seeds"
    );
}
