//! CPU mirror of [`crate::kernels::tile_intersect`] for property tests.

use brush_cube::{Sym2, TileBbox};

const TILE_WIDTH: u32 = 16;

#[derive(Clone, Copy)]
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
    let t = 2.0 * power_threshold;
    let disc = b * b - a * c;
    let neg_t_over_disc = -t / disc;
    let x_extent = (neg_t_over_disc * c).sqrt();
    let y_extent = (neg_t_over_disc * a).sqrt();
    let bx_over_c = b * x_extent / c;
    let by_over_a = b * y_extent / a;
    let tw = TILE_WIDTH as f32;
    let bwf = tile_bw as f32;
    let bhf = tile_bh as f32;

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
        tile_rect: TileBbox {
            min_x: ((mean_x - x_extent) / tw).clamp(0.0, bwf) as u32,
            min_y: ((mean_y - y_extent) / tw).clamp(0.0, bhf) as u32,
            max_x: ((mean_x + x_extent + 1.0) / tw).clamp(0.0, bwf) as u32,
            max_y: ((mean_y + y_extent + 1.0) / tw).clamp(0.0, bhf) as u32,
        },
        t,
        disc,
    }
}

fn accutile_swap(sb: SnugBox, is_y: bool) -> SnugBox {
    if !is_y {
        return sb;
    }
    SnugBox {
        mean_x: sb.mean_y,
        mean_y: sb.mean_x,
        bbox_min_x: sb.bbox_min_y,
        bbox_min_y: sb.bbox_min_x,
        bbox_max_x: sb.bbox_max_y,
        bbox_max_y: sb.bbox_max_x,
        argmin_x: sb.argmin_y,
        argmin_y: sb.argmin_x,
        argmax_x: sb.argmax_y,
        argmax_y: sb.argmax_x,
        tile_rect: TileBbox {
            min_x: sb.tile_rect.min_y,
            min_y: sb.tile_rect.min_x,
            max_x: sb.tile_rect.max_y,
            max_y: sb.tile_rect.max_x,
        },
        t: sb.t,
        disc: sb.disc,
    }
}

fn ellipse_intersection(
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
    let (p_u, p_v) = if is_y {
        (mean_y, mean_x)
    } else {
        (mean_x, mean_y)
    };
    let coeff = if is_y { a } else { c };
    let h = coord - p_u;
    let sqrt_term = (disc * h * h + t * coeff).sqrt();
    let v0 = (-b * h - sqrt_term) / coeff + p_v;
    let v1 = (-b * h + sqrt_term) / coeff + p_v;
    (v0, v1)
}

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
        let (v0, v1) = ellipse_intersection(
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
        intersect_min_v0.min(intersect_max_v0)
    };

    let ellipse_max = if min_line <= work.argmax_y && work.argmax_y < max_line {
        work.bbox_max_y
    } else {
        intersect_min_v1.max(intersect_max_v1)
    };

    let min_tile_v = rect_min_y.max(rect_max_y.min((ellipse_min / block) as u32));
    let max_tile_v = rect_max_y.min(rect_min_y.max((ellipse_max / block + 1.0) as u32));

    (min_tile_v, max_tile_v, intersect_max_v0, intersect_max_v1)
}

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
    if (rect_max_y - rect_min_y) * (rect_max_x - rect_min_x) == 0 {
        return 0;
    }

    let mut intersect_min_v0 = work.bbox_max_y;
    let mut intersect_min_v1 = work.bbox_min_y;
    let mut min_line = rect_min_x as f32 * block;
    if work.bbox_min_x <= min_line {
        let (v0, v1) = ellipse_intersection(
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

    let mut tiles_count = 0u32;
    for _ in rect_min_x..rect_max_x {
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

pub fn snugbox_tile_count(bb: TileBbox) -> u32 {
    (bb.max_x - bb.min_x) * (bb.max_y - bb.min_y)
}

pub fn stop_the_pop_tile_count(sb: SnugBox, conic: Sym2, power_threshold: f32) -> u32 {
    let bb = sb.tile_rect;
    let bb_w = bb.max_x - bb.min_x;
    let mut count = 0u32;
    for tile_idx in 0..(bb.max_y - bb.min_y) * bb_w {
        let tx = (tile_idx % bb_w) + bb.min_x;
        let ty = (tile_idx / bb_w) + bb.min_y;
        let min_x = (tx * TILE_WIDTH) as f32;
        let min_y = (ty * TILE_WIDTH) as f32;
        let max_x = min_x + TILE_WIDTH as f32;
        let max_y = min_y + TILE_WIDTH as f32;
        if tile_contributes(
            min_x,
            min_y,
            max_x,
            max_y,
            sb.mean_x,
            sb.mean_y,
            conic,
            power_threshold,
        ) {
            count += 1;
        }
    }
    count
}

fn tile_contributes(
    rect_min_x: f32,
    rect_min_y: f32,
    rect_max_x: f32,
    rect_max_y: f32,
    mx: f32,
    my: f32,
    conic: Sym2,
    power_threshold: f32,
) -> bool {
    let x_left = mx < rect_min_x;
    let x_right = mx > rect_max_x;
    let in_x_range = !(x_left || x_right);
    let y_above = my < rect_min_y;
    let y_below = my > rect_max_y;
    let in_y_range = !(y_above || y_below);

    if in_x_range && in_y_range {
        return true;
    }

    let corner_x = if x_left { rect_min_x } else { rect_max_x };
    let corner_y = if y_above { rect_min_y } else { rect_max_y };
    let width = rect_max_x - rect_min_x;
    let height = rect_max_y - rect_min_y;
    let dxf = if x_left { width } else { -width };
    let dyf = if y_above { height } else { -height };
    let diff_x = mx - corner_x;
    let diff_y = my - corner_y;

    let tx_raw = (dxf * conic.c00 * diff_x + dxf * conic.c01 * diff_y) / (dxf * conic.c00 * dxf);
    let ty_raw = (dyf * conic.c01 * diff_x + dyf * conic.c11 * diff_y) / (dyf * conic.c11 * dyf);
    let tx = if in_y_range {
        0.0
    } else {
        tx_raw.clamp(0.0, 1.0)
    };
    let ty = if in_x_range {
        0.0
    } else {
        ty_raw.clamp(0.0, 1.0)
    };

    let px = corner_x + tx * dxf;
    let py = corner_y + ty * dyf;
    let dx = px - mx;
    let dy = py - my;
    let sigma = 0.5 * (conic.c00 * dx * dx + conic.c11 * dy * dy) + conic.c01 * dx * dy;
    sigma <= power_threshold
}
