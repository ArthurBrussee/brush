//! Tile intersection via SnugBox bounds + AccuTile enumeration.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::{TILE_WIDTH, get_tile_bbox};
use brush_cube::{Sym2, TileBbox};

/// SnugBox ellipse metadata (opacity-aware AABB + critical points).
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct SnugBox {
    pub mean_x: f32,
    pub mean_y: f32,
    pub bbox_min_x: f32,
    pub bbox_min_y: f32,
    pub bbox_max_x: f32,
    pub bbox_max_y: f32,
    pub argmin_x: f32,
    pub argmin_y: f32,
    pub argmax_x: f32,
    pub argmax_y: f32,
    pub tile_rect: TileBbox,
    pub t: f32,
    pub disc: f32,
}

#[cube]
pub fn compute_snugbox(
    conic: Sym2,
    power_threshold: f32,
    mean_x: f32,
    mean_y: f32,
    tile_bw: u32,
    tile_bh: u32,
) -> SnugBox {
    let a = conic.c00;
    let b = conic.c01;
    let c = conic.c11;
    let t = 2.0f32 * power_threshold;
    let disc = b * b - a * c;
    let neg_t_over_disc = -t / disc;
    let x_extent = f32::sqrt(neg_t_over_disc * c);
    let y_extent = f32::sqrt(neg_t_over_disc * a);

    let bx_over_c = b * x_extent / c;
    let by_over_a = b * y_extent / a;

    SnugBox {
        mean_x,
        mean_y,
        bbox_min_x: mean_x - x_extent,
        bbox_min_y: mean_y - y_extent,
        bbox_max_x: mean_x + x_extent,
        bbox_max_y: mean_y + y_extent,
        argmin_y: mean_y + bx_over_c,
        argmax_y: mean_y - bx_over_c,
        argmin_x: mean_x + by_over_a,
        argmax_x: mean_x - by_over_a,
        tile_rect: get_tile_bbox(mean_x, mean_y, x_extent, y_extent, tile_bw, tile_bh),
        t,
        disc,
    }
}

#[cube]
pub fn snugbox_is_valid(sb: SnugBox) -> bool {
    sb.tile_rect.max_x > sb.tile_rect.min_x
        && sb.tile_rect.max_y > sb.tile_rect.min_y
        && sb.bbox_max_x > sb.bbox_min_x
        && sb.bbox_max_y > sb.bbox_min_y
}

#[cube]
fn accutile_ellipse_intersection(
    a: f32,
    b: f32,
    c: f32,
    disc: f32,
    t: f32,
    mean_x: f32,
    mean_y: f32,
    is_y: bool,
    coord: f32,
) -> (f32, f32) {
    let p_u = select(is_y, mean_y, mean_x);
    let p_v = select(is_y, mean_x, mean_y);
    let coeff = select(is_y, a, c);
    let h = coord - p_u;
    let sqrt_term = f32::sqrt(disc * h * h + t * coeff);
    let v0 = (-b * h - sqrt_term) / coeff + p_v;
    let v1 = (-b * h + sqrt_term) / coeff + p_v;
    (v0, v1)
}

#[cube]
fn accutile_swap(sb: SnugBox, is_y: bool) -> SnugBox {
    SnugBox {
        mean_x: select(is_y, sb.mean_y, sb.mean_x),
        mean_y: select(is_y, sb.mean_x, sb.mean_y),
        bbox_min_x: select(is_y, sb.bbox_min_y, sb.bbox_min_x),
        bbox_min_y: select(is_y, sb.bbox_min_x, sb.bbox_min_y),
        bbox_max_x: select(is_y, sb.bbox_max_y, sb.bbox_max_x),
        bbox_max_y: select(is_y, sb.bbox_max_x, sb.bbox_max_y),
        argmin_x: select(is_y, sb.argmin_y, sb.argmin_x),
        argmin_y: select(is_y, sb.argmin_x, sb.argmin_y),
        argmax_x: select(is_y, sb.argmax_y, sb.argmax_x),
        argmax_y: select(is_y, sb.argmax_x, sb.argmax_y),
        tile_rect: TileBbox {
            min_x: select(is_y, sb.tile_rect.min_y, sb.tile_rect.min_x),
            min_y: select(is_y, sb.tile_rect.min_x, sb.tile_rect.min_y),
            max_x: select(is_y, sb.tile_rect.max_y, sb.tile_rect.max_x),
            max_y: select(is_y, sb.tile_rect.max_x, sb.tile_rect.max_y),
        },
        t: sb.t,
        disc: sb.disc,
    }
}

#[cube]
fn accutile_row(
    work: SnugBox,
    a: f32,
    b: f32,
    c: f32,
    is_y: bool,
    rect_min_y: u32,
    rect_max_y: u32,
    min_line: f32,
    max_line: f32,
    intersect_min_v0: f32,
    intersect_min_v1: f32,
    block: f32,
) -> (u32, u32, f32, f32) {
    let mut intersect_max_v0 = intersect_min_v0;
    let mut intersect_max_v1 = intersect_min_v1;
    if max_line <= work.bbox_max_x {
        let (v0, v1) = accutile_ellipse_intersection(
            a,
            b,
            c,
            work.disc,
            work.t,
            work.mean_x,
            work.mean_y,
            is_y,
            max_line,
        );
        intersect_max_v0 = v0;
        intersect_max_v1 = v1;
    }

    let ellipse_min = if min_line <= work.argmin_y && work.argmin_y < max_line {
        work.bbox_min_y
    } else {
        f32::min(intersect_min_v0, intersect_max_v0)
    };

    let ellipse_max = if min_line <= work.argmax_y && work.argmax_y < max_line {
        work.bbox_max_y
    } else {
        f32::max(intersect_min_v1, intersect_max_v1)
    };

    let min_tile_v = max(rect_min_y, min(rect_max_y, (ellipse_min / block) as u32));
    let max_tile_v = min(
        rect_max_y,
        max(rect_min_y, (ellipse_max / block + 1.0f32) as u32),
    );

    (min_tile_v, max_tile_v, intersect_max_v0, intersect_max_v1)
}

#[cube]
fn accutile_begin(
    work: SnugBox,
    is_y: bool,
    a: f32,
    b: f32,
    c: f32,
    block: f32,
) -> (f32, f32, f32) {
    let rect_min_x = work.tile_rect.min_x;
    let mut intersect_min_v0 = work.bbox_max_y;
    let mut intersect_min_v1 = work.bbox_min_y;
    let min_line = rect_min_x as f32 * block;
    if work.bbox_min_x <= min_line {
        let (v0, v1) = accutile_ellipse_intersection(
            a,
            b,
            c,
            work.disc,
            work.t,
            work.mean_x,
            work.mean_y,
            is_y,
            min_line,
        );
        intersect_min_v0 = v0;
        intersect_min_v1 = v1;
    }
    (min_line, intersect_min_v0, intersect_min_v1)
}

#[cube]
pub fn accutile_tile_count(sb: SnugBox, conic: Sym2) -> u32 {
    let y_span = sb.tile_rect.max_y - sb.tile_rect.min_y;
    let x_span = sb.tile_rect.max_x - sb.tile_rect.min_x;
    let is_y = y_span < x_span;
    let work = accutile_swap(sb, is_y);

    let a = conic.c00;
    let b = conic.c01;
    let c = conic.c11;
    let block = TILE_WIDTH as f32;

    let rect_min_x = work.tile_rect.min_x;
    let rect_min_y = work.tile_rect.min_y;
    let rect_max_x = work.tile_rect.max_x;
    let rect_max_y = work.tile_rect.max_y;
    let non_empty = (rect_max_y - rect_min_y) * (rect_max_x - rect_min_x) != 0u32;

    let (mut min_line, mut intersect_min_v0, mut intersect_min_v1) =
        accutile_begin(work, is_y, a, b, c, block);

    let mut tiles_count = 0u32;
    let num_u = select(non_empty, rect_max_x - rect_min_x, 0u32);
    for _step in 0u32..num_u {
        let max_line = min_line + block;
        let (min_tile_v, max_tile_v, new_v0, new_v1) = accutile_row(
            work,
            a,
            b,
            c,
            is_y,
            rect_min_y,
            rect_max_y,
            min_line,
            max_line,
            intersect_min_v0,
            intersect_min_v1,
            block,
        );
        tiles_count += max_tile_v.saturating_sub(min_tile_v);
        intersect_min_v0 = new_v0;
        intersect_min_v1 = new_v1;
        min_line = max_line;
    }

    tiles_count
}

#[cube]
pub fn accutile_write_tiles(
    sb: SnugBox,
    conic: Sym2,
    tile_bw: u32,
    base_isect_id: u32,
    compact_gid: u32,
    writable: u32,
    tile_id_from_isect: &mut Tensor<u32>,
    compact_gid_from_isect: &mut Tensor<u32>,
) -> u32 {
    let y_span = sb.tile_rect.max_y - sb.tile_rect.min_y;
    let x_span = sb.tile_rect.max_x - sb.tile_rect.min_x;
    let is_y = y_span < x_span;
    let work = accutile_swap(sb, is_y);

    let a = conic.c00;
    let b = conic.c01;
    let c = conic.c11;
    let block = TILE_WIDTH as f32;

    let rect_min_x = work.tile_rect.min_x;
    let rect_min_y = work.tile_rect.min_y;
    let rect_max_x = work.tile_rect.max_x;
    let rect_max_y = work.tile_rect.max_y;
    let non_empty = (rect_max_y - rect_min_y) * (rect_max_x - rect_min_x) != 0u32;

    let (mut min_line, mut intersect_min_v0, mut intersect_min_v1) =
        accutile_begin(work, is_y, a, b, c, block);

    let mut tiles_count = 0u32;
    let num_u = select(non_empty, rect_max_x - rect_min_x, 0u32);
    for step in 0u32..num_u {
        let u = rect_min_x + step;
        let max_line = min_line + block;
        let (min_tile_v, max_tile_v, new_v0, new_v1) = accutile_row(
            work,
            a,
            b,
            c,
            is_y,
            rect_min_y,
            rect_max_y,
            min_line,
            max_line,
            intersect_min_v0,
            intersect_min_v1,
            block,
        );

        let row_tiles = max_tile_v.saturating_sub(min_tile_v);
        for offset in 0u32..row_tiles {
            let v = min_tile_v + offset;
            if tiles_count < writable {
                let tile_id = if is_y {
                    u + v * tile_bw
                } else {
                    v + u * tile_bw
                };
                let isect_id = base_isect_id + tiles_count;
                tile_id_from_isect[isect_id as usize] = tile_id;
                compact_gid_from_isect[isect_id as usize] = compact_gid;
            }
            tiles_count += 1u32;
        }

        intersect_min_v0 = new_v0;
        intersect_min_v1 = new_v1;
        min_line = max_line;
    }

    tiles_count
}
