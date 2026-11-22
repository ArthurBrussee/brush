//! `map_gaussians_to_intersect` must emit exactly the tiles each splat's
//! cutoff ellipse touches. Checked against a brute-force test per tile.

use crate::kernels::helpers::{PROJECTED_LANES_USIZE, TILE_WIDTH};
use crate::kernels::map_gaussians::{WG_SIZE, map_gaussians_to_intersect_kernel};
use brush_cube::{CubeDevice, MainBackendBase, create_tensor, create_tensor_from_slice};
use burn::backend::ops::IntTensorOps;
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::DType;
use wasm_bindgen_test::wasm_bindgen_test;

// Deliberately not square, and not a multiple of anything.
const TILE_BW: u32 = 23;
const TILE_BH: u32 = 9;
const NUM_TILES: u32 = TILE_BW * TILE_BH;

struct Splat {
    xy: [f32; 2],
    /// `[c00, c01, c11]`.
    conic: [f32; 3],
    opac: f32,
}

/// Smallest `sigma` (see `calc_sigma`) the splat reaches anywhere in a tile.
/// Zero if the mean is inside, else it's on one of the four edges, where
/// sigma is a 1D quadratic.
fn min_sigma_in_tile(s: &Splat, tx: u32, ty: u32) -> f64 {
    let [a, b, c] = s.conic.map(f64::from);
    let tw = f64::from(TILE_WIDTH);
    let x0 = f64::from(tx) * tw - f64::from(s.xy[0]);
    let y0 = f64::from(ty) * tw - f64::from(s.xy[1]);
    let (x1, y1) = (x0 + tw, y0 + tw);
    if x0 <= 0.0 && x1 >= 0.0 && y0 <= 0.0 && y1 >= 0.0 {
        return 0.0;
    }
    let sigma = |x: f64, y: f64| 0.5 * (a * x * x + c * y * y) + b * x * y;
    let on_x_edge = |x: f64| sigma(x, (-b * x / c).clamp(y0, y1));
    let on_y_edge = |y: f64| sigma((-b * y / a).clamp(x0, x1), y);
    on_x_edge(x0)
        .min(on_x_edge(x1))
        .min(on_y_edge(y0))
        .min(on_y_edge(y1))
}

/// Run the real kernel and return each splat's emitted tile ids, sorted.
async fn emitted_tiles(splats: &[Splat]) -> Vec<Vec<u32>> {
    let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
    let client = device.client();

    let mut projected = vec![0.0f32; splats.len() * PROJECTED_LANES_USIZE];
    for (s, lanes) in splats
        .iter()
        .zip(projected.as_chunks_mut::<PROJECTED_LANES_USIZE>().0)
    {
        lanes[..2].copy_from_slice(&s.xy);
        lanes[2..5].copy_from_slice(&s.conic);
        lanes[5] = s.opac;
    }
    // Reserve room for the whole grid per splat; unused slots come back as
    // the sentinel tile id.
    let cum_counts: Vec<u32> = (1..=splats.len() as u32).map(|i| i * NUM_TILES).collect();
    let num_slots = splats.len() * NUM_TILES as usize;
    let tile_ids = create_tensor([num_slots], &device, DType::U32);
    let gids = create_tensor([num_slots], &device, DType::U32);

    let cube_dim = CubeDim::new_1d(WG_SIZE);
    map_gaussians_to_intersect_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, splats.len(), cube_dim),
        cube_dim,
        create_tensor_from_slice(&projected, &device, DType::F32).into_tensor_arg(),
        create_tensor_from_slice(&cum_counts, &device, DType::U32).into_tensor_arg(),
        tile_ids.clone().into_tensor_arg(),
        gids.into_tensor_arg(),
        TILE_BW,
        TILE_BH,
        splats.len() as u32,
    );

    let tile_ids = MainBackendBase::int_into_data(tile_ids)
        .await
        .expect("readback");
    tile_ids
        .as_slice::<u32>()
        .expect("u32 tile ids")
        .as_chunks::<{ NUM_TILES as usize }>()
        .0
        .iter()
        .map(|slots| {
            let mut ids: Vec<u32> = slots.iter().copied().filter(|&id| id < NUM_TILES).collect();
            ids.sort_unstable();
            ids
        })
        .collect()
}

fn random_splats(count: usize) -> Vec<Splat> {
    use std::num::Wrapping;
    // SplitMix64: tiny, no deps, deterministic.
    let mut state = Wrapping(0x5EED_u64);
    let mut next = || {
        state += Wrapping(0x9E3779B97F4A7C15u64);
        let mut z = state.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let (w, h) = (
        f64::from(TILE_BW * TILE_WIDTH),
        f64::from(TILE_BH * TILE_WIDTH),
    );
    (0..count)
        .map(|_| {
            // Means in and a bit around the image, std-devs from sub-pixel
            // to most of the image, any rotation, up to ~100:1 stretch.
            let xy = [next() * 1.4 * w - 0.2 * w, next() * 1.4 * h - 0.2 * h];
            let major = (next() * 6.0 - 0.7).exp();
            let minor = major * (-next() * 4.6).exp();
            let (sin, cos) = (next() * std::f64::consts::TAU).sin_cos();
            let (ia, ib) = (1.0 / (major * major), 1.0 / (minor * minor));
            let conic = [
                cos * cos * ia + sin * sin * ib,
                sin * cos * (ia - ib),
                sin * sin * ia + cos * cos * ib,
            ];
            Splat {
                xy: xy.map(|v| v as f32),
                conic: conic.map(|v| v as f32),
                opac: (next() * 0.99 + 0.01) as f32,
            }
        })
        .collect()
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn emits_exactly_the_touched_tiles() {
    let splats = random_splats(4000);
    let emitted = emitted_tiles(&splats).await;

    let mut total = 0;
    for (i, (s, emitted)) in splats.iter().zip(&emitted).enumerate() {
        let threshold = f64::from(s.opac * 255.0).ln();
        for ty in 0..TILE_BH {
            for tx in 0..TILE_BW {
                let id = tx + ty * TILE_BW;
                let got = emitted.binary_search(&id).is_ok();
                // Leave f32 rounding room right at the ellipse's edge.
                let sigma = min_sigma_in_tile(s, tx, ty);
                let slack = 1e-3 * threshold + 1e-4;
                if sigma < threshold - slack {
                    assert!(got, "splat {i} misses tile ({tx}, {ty})");
                } else if sigma > threshold + slack {
                    assert!(!got, "splat {i} emits untouched tile ({tx}, {ty})");
                }
            }
        }
        // No tile twice.
        assert!(emitted.windows(2).all(|w| w[0] < w[1]), "splat {i}");
        total += emitted.len();
    }
    // The scene should exercise real coverage, not mostly-empty splats.
    assert!(total > 10 * splats.len(), "only {total} tiles emitted");
}
