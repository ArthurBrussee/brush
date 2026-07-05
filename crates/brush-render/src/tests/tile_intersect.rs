use super::tile_intersect_ref::{
    accutile_tile_count, compute_snugbox, snugbox_tile_count, stop_the_pop_tile_count,
};
use brush_cube::Sym2;

fn hash_f32(x: u32) -> f32 {
    let mut h = x.wrapping_mul(747796405);
    h ^= h >> 16;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    (h as f32) / u32::MAX as f32
}

fn random_cases(n: usize, tile_bw: u32, tile_bh: u32) -> Vec<(Sym2, f32, f32, f32, u32, u32)> {
    (0..n)
        .map(|i| {
            let conic = Sym2 {
                c00: 0.01 + hash_f32(i as u32 * 3) * 4.0,
                c01: hash_f32(i as u32 * 3 + 1) - 0.5,
                c11: 0.01 + hash_f32(i as u32 * 3 + 2) * 4.0,
            };
            let opac = 0.05 + hash_f32(i as u32 * 5) * 0.95;
            let mx = hash_f32(i as u32 * 7) * 1920.0;
            let my = hash_f32(i as u32 * 11) * 1080.0;
            (conic, opac, mx, my, tile_bw, tile_bh)
        })
        .collect()
}

#[test]
fn accutile_count_within_snugbox_bbox() {
    for (conic, opac, mx, my, tile_bw, tile_bh) in random_cases(512, 120, 68) {
        let power = (opac * 255.0).ln();
        let sb = compute_snugbox(conic, power, mx, my, tile_bw, tile_bh);
        let snug = snugbox_tile_count(sb.tile_rect);
        let accu = accutile_tile_count(sb, conic);
        assert!(
            accu <= snug,
            "accutile {accu} > snugbox bbox {snug} at ({mx},{my})"
        );
    }
}

#[test]
fn accutile_count_at_most_stop_the_pop() {
    for (conic, opac, mx, my, tile_bw, tile_bh) in random_cases(512, 120, 68) {
        let power = (opac * 255.0).ln();
        let sb = compute_snugbox(conic, power, mx, my, tile_bw, tile_bh);
        let accu = accutile_tile_count(sb, conic);
        let stop = stop_the_pop_tile_count(sb, conic, power);
        assert!(
            accu <= stop,
            "accutile {accu} > stop-the-pop {stop} at ({mx},{my})"
        );
    }
}
