// SH coefficient struct — field names avoid trailing digits (naga_oil limitation).
// Band 0: dc. Bands 1-4 use band letter (b/c/d/e) + coefficient letter (a-i).
struct ShCoeffs {
    dc: vec3f,
    ba: vec3f, bb: vec3f, bc: vec3f,
    ca: vec3f, cb: vec3f, cc: vec3f, cd: vec3f, ce: vec3f,
    da: vec3f, db: vec3f, dd: vec3f, de: vec3f, df: vec3f, dg: vec3f, dh: vec3f,
    ea: vec3f, eb: vec3f, ec: vec3f, ed: vec3f, ee: vec3f, ef: vec3f, eg: vec3f, eh: vec3f, ei: vec3f,
}

const SH_C0: f32 = 0.2820947917738781f;

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1) * (degree + 1);
}

// Evaluate spherical harmonics bases at unit direction for high orders using approach described by
// Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
// See https://jcgt.org/published/0002/02/06/ for reference implementation
fn sh_coeffs_to_color(
    degree: u32,
    viewdir: vec3f,
    sh: ShCoeffs,
) -> vec3f {
    var colors = SH_C0 * sh.dc;

    if (degree == 0) {
        return colors;
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292f;
    colors += fTmp0A * (-y * sh.ba + z * sh.bb - x * sh.bc);

    if (degree == 1) {
        return colors;
    }
    let z2 = z * z;

    let fTmp0B = -1.092548430592079f * z;
    let fTmp1A = 0.5462742152960395f;
    let fC1 = x * x - y * y;
    let fS1 = 2.f * x * y;
    let pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;

    colors +=
        pSH4 * sh.ca + pSH5 * sh.cb + pSH6 * sh.cc +
        pSH7 * sh.cd + pSH8 * sh.ce;

    if (degree == 2) {
        return colors;
    }

    let fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    let fTmp1B = 1.445305721320277f * z;
    let fTmp2A = -0.5900435899266435f;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    let pSH13 = fTmp0C * x;
    let pSH11 = fTmp0C * y;
    let pSH14 = fTmp1B * fC1;
    let pSH10 = fTmp1B * fS1;
    let pSH15 = fTmp2A * fC2;
    let pSH9  = fTmp2A * fS2;
    colors += pSH9 * sh.da + pSH10 * sh.db + pSH11 * sh.dd +
              pSH12 * sh.de + pSH13 * sh.df + pSH14 * sh.dg +
              pSH15 * sh.dh;

    if (degree == 3) {
        return colors;
    }

    let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    let fTmp2B = -1.770130769779931f * z;
    let fTmp3A = 0.6258357354491763f;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
    let pSH21 = fTmp0D * x;
    let pSH19 = fTmp0D * y;
    let pSH22 = fTmp1C * fC1;
    let pSH18 = fTmp1C * fS1;
    let pSH23 = fTmp2B * fC2;
    let pSH17 = fTmp2B * fS2;
    let pSH24 = fTmp3A * fC3;
    let pSH16 = fTmp3A * fS3;
    colors += pSH16 * sh.ea + pSH17 * sh.eb + pSH18 * sh.ec +
              pSH19 * sh.ed + pSH20 * sh.ee + pSH21 * sh.ef +
              pSH22 * sh.eg + pSH23 * sh.eh + pSH24 * sh.ei;
    return colors;
}

// VJP of sh_coeffs_to_color: given v_colors (gradient of output color),
// compute gradient w.r.t. each SH coefficient.
fn sh_coeffs_to_color_vjp(
    degree: u32,
    viewdir: vec3f,
    v_colors: vec3f,
) -> ShCoeffs {
    var v = ShCoeffs();

    v.dc = SH_C0 * v_colors;
    if (degree == 0) { return v; }

    let x = viewdir.x; let y = viewdir.y; let z = viewdir.z;
    let fTmp0A = 0.48860251190292f;
    v.ba = -fTmp0A * y * v_colors;
    v.bb = fTmp0A * z * v_colors;
    v.bc = -fTmp0A * x * v_colors;
    if (degree == 1) { return v; }

    let z2 = z * z;
    let fTmp0B = -1.092548430592079f * z;
    let fTmp1A = 0.5462742152960395f;
    let fC1 = x * x - y * y;
    let fS1 = 2.f * x * y;
    let pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    let pSH4 = fTmp1A * fS1; let pSH5 = fTmp0B * y;
    let pSH7 = fTmp0B * x; let pSH8 = fTmp1A * fC1;
    v.ca = pSH4 * v_colors; v.cb = pSH5 * v_colors;
    v.cc = pSH6 * v_colors; v.cd = pSH7 * v_colors;
    v.ce = pSH8 * v_colors;
    if (degree == 2) { return v; }

    let fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    let fTmp1B = 1.445305721320277f * z;
    let fTmp2A = -0.5900435899266435f;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    let pSH9 = fTmp2A * fS2; let pSH10 = fTmp1B * fS1; let pSH11 = fTmp0C * y;
    let pSH13 = fTmp0C * x; let pSH14 = fTmp1B * fC1; let pSH15 = fTmp2A * fC2;
    v.da = pSH9 * v_colors; v.db = pSH10 * v_colors;
    v.dd = pSH11 * v_colors; v.de = pSH12 * v_colors;
    v.df = pSH13 * v_colors; v.dg = pSH14 * v_colors;
    v.dh = pSH15 * v_colors;
    if (degree == 3) { return v; }

    let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    let fTmp2B = -1.770130769779931f * z;
    let fTmp3A = 0.6258357354491763f;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    let pSH16 = fTmp3A * fS3; let pSH17 = fTmp2B * fS2; let pSH18 = fTmp1C * fS1;
    let pSH19 = fTmp0D * y; let pSH21 = fTmp0D * x; let pSH22 = fTmp1C * fC1;
    let pSH23 = fTmp2B * fC2; let pSH24 = fTmp3A * fC3;
    v.ea = pSH16 * v_colors; v.eb = pSH17 * v_colors;
    v.ec = pSH18 * v_colors; v.ed = pSH19 * v_colors;
    v.ee = pSH20 * v_colors; v.ef = pSH21 * v_colors;
    v.eg = pSH22 * v_colors; v.eh = pSH23 * v_colors;
    v.ei = pSH24 * v_colors;
    return v;
}
