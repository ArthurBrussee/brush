use crate::kernels::camera_model::CameraModel::{
    KannalaBrandt4, Pinhole, RadialTangential8, ThinPrismFisheye,
};
use crate::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use crate::kernels::camera_model::pinhole::PinholeParams;
use crate::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use crate::kernels::camera_model::{CameraModel, JacobianClampLimits};
use glam::Affine3A;
use std::f64::consts::PI;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Camera {
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: glam::Vec2,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub camera_model: CameraModel,
}

impl Camera {
    pub fn new(
        position: glam::Vec3,
        rotation: glam::Quat,
        fov_x: f64,
        fov_y: f64,
        center_uv: glam::Vec2,
        camera_model: CameraModel,
    ) -> Self {
        Self {
            fov_x,
            fov_y,
            center_uv,
            position,
            rotation,
            camera_model,
        }
    }

    /// Check if the camera has valid (non-nan/inf) settings.
    pub fn is_valid(&self) -> bool {
        self.fov_x.is_finite()
            && self.fov_y.is_finite()
            && self.center_uv.is_finite()
            && self.position.is_finite()
            && self.rotation.is_finite()
    }

    pub fn focal(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2(
            fov_to_focal(self.fov_x, img_size.x, &self.camera_model) as f32,
            fov_to_focal(self.fov_y, img_size.y, &self.camera_model) as f32,
        )
    }

    pub fn center(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2(
            self.center_uv.x * img_size.x as f32,
            self.center_uv.y * img_size.y as f32,
        )
    }

    pub fn build_pinhole_params(&self, img_size: glam::UVec2) -> PinholeParams {
        let focal = self.focal(img_size);
        let pixel_center = self.center(img_size);

        PinholeParams {
            fx: focal.x,
            fy: focal.y,
            cx: pixel_center.x,
            cy: pixel_center.y,
        }
    }

    /// The same pose and field of view with the lens model swapped for an
    /// ideal pinhole, e.g. to render a preview without distortion.
    pub fn with_pinhole(&self) -> Self {
        Self {
            camera_model: Pinhole,
            ..*self
        }
    }

    pub fn local_to_world(&self) -> Affine3A {
        Affine3A::from_rotation_translation(self.rotation, self.position)
    }

    pub fn world_to_local(&self) -> Affine3A {
        self.local_to_world().inverse()
    }
}

// Converts field of view to focal length
pub fn fov_to_focal(fov: f64, pixels: u32, model: &CameraModel) -> f64 {
    let half_fov = fov / 2.0;
    let r_pix = (pixels as f64) / 2.0;

    // We want focal f such that r_pix = f · projection(half_fov).
    let projected = match model {
        Pinhole => half_fov.tan(),
        KannalaBrandt4(p) => kb4_d(half_fov, p),
        RadialTangential8(p) => {
            let r = half_fov.tan();
            r * rt8_radial(r, p)
        }
        ThinPrismFisheye(p) => kb4_d(half_fov, &p.kb4),
    };

    r_pix / projected
}

// Converts focal length to field of view
pub fn focal_to_fov(focal: f64, pixels: u32, model: &CameraModel) -> f64 {
    let r_pix = (pixels as f64) / 2.0;
    let r_norm = r_pix / focal; // distorted normalized radius (= d(θ) for KB4)

    let half_fov = match model {
        Pinhole => r_norm.atan(),
        KannalaBrandt4(p) => kb4_invert_d(r_norm, p),
        RadialTangential8(p) => {
            let r_undist = rt8_undistort_radius(r_norm, p);
            r_undist.atan()
        }
        ThinPrismFisheye(p) => kb4_invert_d(r_norm, &p.kb4),
    };

    2.0 * half_fov
}

// KB4 distortion polynomial: d(θ) = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹
#[inline]
fn kb4_d(theta: f64, p: &KannalaBrandt4Params) -> f64 {
    let t2 = theta * theta;
    let t3 = t2 * theta;
    let t5 = t3 * t2;
    let t7 = t5 * t2;
    let t9 = t7 * t2;
    theta + p.k1 as f64 * t3 + p.k2 as f64 * t5 + p.k3 as f64 * t7 + p.k4 as f64 * t9
}

// d'(θ) = 1 + 3k1·θ² + 5k2·θ⁴ + 7k3·θ⁶ + 9k4·θ⁸
#[inline]
fn kb4_dd_dtheta(theta: f64, p: &KannalaBrandt4Params) -> f64 {
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t6 * t2;
    1.0 + 3.0 * p.k1 as f64 * t2
        + 5.0 * p.k2 as f64 * t4
        + 7.0 * p.k3 as f64 * t6
        + 9.0 * p.k4 as f64 * t8
}

// Largest polar angle the model projects monotonically. A fitted KB4
// polynomial is only increasing up to its first stationary point, and real
// ultrawide fits fold well inside the hemisphere (a phone ultrawide peaks
// around 65° and goes negative past 80°). Past the fold, points project
// mirrored back into the image with garbage radii, so the renderer must not
// let them through the angular cull. Returns π for models without a fold.
pub fn max_render_theta(model: &CameraModel) -> f64 {
    match model {
        Pinhole | RadialTangential8(_) => PI,
        KannalaBrandt4(p) => kb4_fold_theta(p),
        ThinPrismFisheye(p) => kb4_fold_theta(&p.kb4),
    }
}

// First θ ∈ (0, π] where d'(θ) reaches zero, or π if d stays increasing.
// Coarse scan for the first sign change of d', then bisection on that step.
fn kb4_fold_theta(p: &KannalaBrandt4Params) -> f64 {
    const STEPS: usize = 128;
    let step = PI / STEPS as f64;

    let mut lo = 0.0;
    let mut hi = PI;
    let mut found = false;
    for i in 1..=STEPS {
        let theta = i as f64 * step;
        if kb4_dd_dtheta(theta, p) <= 0.0 {
            hi = theta;
            found = true;
            break;
        }
        lo = theta;
    }
    if !found {
        return PI;
    }

    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if kb4_dd_dtheta(mid, p) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

// Solve d(θ) = target on the rising branch θ ∈ [0, fold] via bisection.
// d is strictly increasing there, so this is the unique physical solution;
// targets past the fold (larger than the lens can ever produce) clamp to it.
// Newton seeded at the target could settle on the far, folded branch instead.
fn kb4_invert_d(target: f64, p: &KannalaBrandt4Params) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let fold = kb4_fold_theta(p);
    if target >= kb4_d(fold, p) {
        return fold;
    }

    let mut lo = 0.0;
    let mut hi = fold;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if kb4_d(mid, p) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-14 {
            break;
        }
    }
    0.5 * (lo + hi)
}

// Radial distortion factor for RadTan8: (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶)
#[inline]
fn rt8_radial(r: f64, p: &RadialTangential8Params) -> f64 {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let num = 1.0 + p.k1 as f64 * r2 + p.k2 as f64 * r4 + p.k3 as f64 * r6;
    let den = 1.0 + p.k4 as f64 * r2 + p.k5 as f64 * r4 + p.k6 as f64 * r6;
    num / den
}

// Given the *distorted* normalized radius r_d, recover the undistorted r
// such that r · radial(r) = r_d. Fixed-point iteration (standard OpenCV approach).
fn rt8_undistort_radius(r_d: f64, p: &RadialTangential8Params) -> f64 {
    let mut r = r_d;
    for _ in 0..30 {
        let factor = rt8_radial(r, p);
        if factor.abs() < 1e-12 {
            break;
        }
        let r_new = r_d / factor;
        if (r_new - r).abs() < 1e-12 {
            r = r_new;
            break;
        }
        r = r_new;
    }
    r
}

// Undistorted radius r with r · radial(r) = r_d, found by bracketing the first
// crossing on a coarse scan and bisecting it. Unlike the fixed-point inverse
// this doesn't diverge where the polynomial is steep. None if the distortion
// never reaches r_d below r = 16 (the polynomial folds first).
fn rt8_undistort_corner_radius(r_d: f64, p: &RadialTangential8Params) -> Option<f64> {
    const R_MAX: f64 = 16.0;
    const STEPS: usize = 4096;
    let distort = |r: f64| r * rt8_radial(r, p);
    let step = R_MAX / STEPS as f64;

    let mut lo = 0.0;
    let mut hi = None;
    for i in 1..=STEPS {
        let r = i as f64 * step;
        if distort(r) >= r_d {
            hi = Some(r);
            break;
        }
        lo = r;
    }
    let mut hi = hi?;

    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if distort(mid) < r_d {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    Some(0.5 * (lo + hi))
}

pub fn calculate_jacobian_clamp_limits(
    img_size: glam::UVec2,
    pinhole_params: PinholeParams,
    camera_model: CameraModel,
) -> JacobianClampLimits {
    let PinholeParams { fx, fy, cx, cy } = pinhole_params;

    let mut lim_pos_x = 0.;
    let mut lim_neg_x = 0.;
    let mut lim_pos_y = 0.;
    let mut lim_neg_y = 0.;
    let mut lim_r = f32::MAX;

    let img_w = img_size.x as f32;
    let img_h = img_size.y as f32;

    // The clamp bounds the normalized coord x/z that feeds the projection, so
    // the EWA covariance Jacobian isn't evaluated where the perspective
    // projection blows up near the field-of-view edge. The pinhole margin
    // `1.15 * img - c` equals the canonical 3DGS limit `1.3 * tan(fov/2)`
    // (graphdeco-inria/diff-gaussian-rasterization, `computeCov2D`).
    match camera_model {
        Pinhole => {
            lim_pos_x = (1.15 * img_w - cx) / fx;
            lim_pos_y = (1.15 * img_h - cy) / fy;
            lim_neg_x = (-0.15 * img_w - cx) / fx;
            lim_neg_y = (-0.15 * img_h - cy) / fy;
        }
        RadialTangential8(p) => {
            // The clamp bounds x/z, the *undistorted* coord that `project_rt8`
            // feeds the distortion. A pixel maps to the distorted coord
            // `(px - c) / f`; invert the radial model to get the undistorted
            // bound. With the same image margin as pinhole this collapses to the
            // pinhole limit for a near-pinhole lens (so a tiny distortion no
            // longer loosens the clamp and lets wide-fov splats blow up) while
            // widening it for real barrel distortion.
            let undistort =
                |edge: f32| (rt8_undistort_radius((edge as f64).abs(), &p) as f32) * edge.signum();
            let edge_pos_x = (1.15 * img_w - cx) / fx;
            let edge_pos_y = (1.15 * img_h - cy) / fy;
            lim_pos_x = undistort(edge_pos_x);
            lim_pos_y = undistort(edge_pos_y);
            let edge_neg_x = (-0.15 * img_w - cx) / fx;
            let edge_neg_y = (-0.15 * img_h - cy) / fy;
            lim_neg_x = undistort(edge_neg_x);
            lim_neg_y = undistort(edge_neg_y);

            // The per-axis box still admits its corner at hypot(lim_x, lim_y),
            // where the polynomial has left its calibrated range and the
            // fixed-point inverse diverges; splats there got Jacobians orders
            // of magnitude too large. Cap the radius at the undistorted radius
            // of the same-margin image corner instead.
            let corner_d = (edge_pos_x.abs().max(edge_neg_x.abs()) as f64)
                .hypot(edge_pos_y.abs().max(edge_neg_y.abs()) as f64);
            lim_r = rt8_undistort_corner_radius(corner_d, &p).map_or(f32::MAX, |r| r as f32);
        }
        // Fisheye models project the full hemisphere without the perspective
        // singularity, so their Jacobians aren't clamped (their kernels ignore
        // these limits); leave them at zero.
        KannalaBrandt4(_) | ThinPrismFisheye(_) => {}
    }

    JacobianClampLimits {
        lim_pos_x,
        lim_pos_y,
        lim_neg_x,
        lim_neg_y,
        lim_r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::camera_model::thin_prism_fisheye::ThinPrismFisheyeParams;

    // Near-rectilinear phone ultrawide fit: d(θ) tracks tan(θ) inside the
    // image, peaks at 65° and goes negative past 80°. The Newton solver seeded
    // at the target (1.25 rad, already past the fold) used to report a 146°
    // fov for this lens.
    const ULTRAWIDE: KannalaBrandt4Params = KannalaBrandt4Params {
        k1: 0.278781,
        k2: 0.464781,
        k3: -0.148871,
        k4: -0.150010,
    };
    const ULTRAWIDE_IMG: glam::UVec2 = glam::uvec2(4032, 3024);
    const ULTRAWIDE_FOCAL: f64 = 1606.3915;
    const ULTRAWIDE_FOV_X: f64 = 103.5;

    fn deg(rad: f64) -> f64 {
        rad.to_degrees()
    }

    #[test]
    fn ultrawide_polynomial_folds_inside_hemisphere() {
        let fold = kb4_fold_theta(&ULTRAWIDE);
        assert!((deg(fold) - 65.0).abs() < 0.1, "fold at {}°", deg(fold));
        assert!(kb4_dd_dtheta(fold, &ULTRAWIDE).abs() < 1e-6);
        // Strictly increasing up to the fold, folded back and negative after.
        let mut prev = 0.0;
        for i in 1..=1000 {
            let d = kb4_d(fold * i as f64 / 1000.0, &ULTRAWIDE);
            assert!(d > prev);
            prev = d;
        }
        assert!(kb4_d(70f64.to_radians(), &ULTRAWIDE) < kb4_d(fold, &ULTRAWIDE));
        assert!(kb4_d(85f64.to_radians(), &ULTRAWIDE) < 0.0);
    }

    #[test]
    fn ultrawide_fov_from_focal_stays_on_rising_branch() {
        let model = KannalaBrandt4(ULTRAWIDE);
        let fov_x = deg(focal_to_fov(ULTRAWIDE_FOCAL, ULTRAWIDE_IMG.x, &model));
        assert!(
            (fov_x - ULTRAWIDE_FOV_X).abs() < 0.01,
            "fov_x = {fov_x}°, expected {ULTRAWIDE_FOV_X}°"
        );
        let fov_y = deg(focal_to_fov(ULTRAWIDE_FOCAL, ULTRAWIDE_IMG.y, &model));
        assert!((fov_y - 85.6).abs() < 0.1, "fov_y = {fov_y}°");

        // Thin prism shares the radial polynomial, so it must agree.
        let tpf = ThinPrismFisheye(ThinPrismFisheyeParams {
            kb4: ULTRAWIDE,
            ..Default::default()
        });
        let tpf_fov_x = deg(focal_to_fov(ULTRAWIDE_FOCAL, ULTRAWIDE_IMG.x, &tpf));
        assert!((tpf_fov_x - fov_x).abs() < 1e-9);
    }

    #[test]
    fn ultrawide_focal_round_trips() {
        let model = KannalaBrandt4(ULTRAWIDE);
        for (pixels, focal) in [
            (ULTRAWIDE_IMG.x, ULTRAWIDE_FOCAL),
            (ULTRAWIDE_IMG.y, ULTRAWIDE_FOCAL),
            (1920, 900.0),
            (640, 260.0),
        ] {
            let fov = focal_to_fov(focal, pixels, &model);
            let back = fov_to_focal(fov, pixels, &model);
            assert!(
                (back - focal).abs() < 1e-6 * focal,
                "focal {focal} -> fov {}° -> focal {back}",
                deg(fov)
            );
        }
        for fov_deg in [10.0, 60.0, 103.5, 125.0] {
            let fov = f64::to_radians(fov_deg);
            let focal = fov_to_focal(fov, ULTRAWIDE_IMG.x, &model);
            let back = focal_to_fov(focal, ULTRAWIDE_IMG.x, &model);
            assert!(
                (back - fov).abs() < 1e-9,
                "fov {fov_deg}° -> {}°",
                deg(back)
            );
        }

        // A focal so short the image edge lies past the polynomial's peak
        // can't round trip: the fov saturates at the fold instead.
        let fov = focal_to_fov(200.0, 640, &model);
        assert_eq!(fov, 2.0 * kb4_fold_theta(&ULTRAWIDE));
    }

    #[test]
    fn targets_past_the_fold_clamp_to_it() {
        let fold = kb4_fold_theta(&ULTRAWIDE);
        let d_max = kb4_d(fold, &ULTRAWIDE);
        assert_eq!(kb4_invert_d(d_max, &ULTRAWIDE), fold);
        assert_eq!(kb4_invert_d(d_max * 10.0, &ULTRAWIDE), fold);
        assert_eq!(kb4_invert_d(0.0, &ULTRAWIDE), 0.0);
        assert_eq!(kb4_invert_d(-1.0, &ULTRAWIDE), 0.0);
    }

    #[test]
    fn render_theta_is_capped_at_the_fold() {
        let model = KannalaBrandt4(ULTRAWIDE);
        let fov_x = focal_to_fov(ULTRAWIDE_FOCAL, ULTRAWIDE_IMG.x, &model);
        let fov_y = focal_to_fov(ULTRAWIDE_FOCAL, ULTRAWIDE_IMG.y, &model);
        // The diagonal-based cull margin reaches past the fold for this lens,
        // so without the cap splats outside the lens get projected mirrored.
        let uncapped = fov_x.hypot(fov_y) * 1.05 * 0.5;
        let cap = max_render_theta(&model);
        assert!(cap < uncapped, "cap {}° vs {}°", deg(cap), deg(uncapped));
        assert!((deg(cap) - 65.0).abs() < 0.1);

        let tpf = ThinPrismFisheye(ThinPrismFisheyeParams {
            kb4: ULTRAWIDE,
            ..Default::default()
        });
        assert_eq!(max_render_theta(&tpf), cap);
        assert_eq!(max_render_theta(&Pinhole), PI);
    }

    #[test]
    fn rt8_radial_cap_matches_image_corner() {
        // Pincushion-ish rational fit: the box corner reaches further out
        // than the image corner's true undistorted radius.
        let p = RadialTangential8Params {
            k1: 0.15,
            k2: 0.08,
            k3: 0.02,
            k4: 0.03,
            k5: 0.0,
            k6: 0.0,
            p1: 0.001,
            p2: -0.002,
        };
        let img = glam::uvec2(1920, 1080);
        let pinhole = PinholeParams {
            fx: 800.0,
            fy: 800.0,
            cx: 960.0,
            cy: 540.0,
        };
        let limits = calculate_jacobian_clamp_limits(img, pinhole, RadialTangential8(p));

        // lim_r is the undistorted radius of the same-margin image corner.
        let corner_d = ((1.15 * 1920.0 - 960.0) / 800.0f64).hypot((1.15 * 1080.0 - 540.0) / 800.0);
        let lim_r = limits.lim_r as f64;
        assert!((lim_r * rt8_radial(lim_r, &p) - corner_d).abs() < 1e-5);
        // ... and tighter than the box corner it is meant to cut off.
        let box_corner = limits.lim_pos_x.hypot(limits.lim_pos_y);
        assert!(
            limits.lim_r < box_corner,
            "{} vs {box_corner}",
            limits.lim_r
        );
        // Per-axis limits stay inside the cap.
        assert!(limits.lim_pos_x < limits.lim_r && limits.lim_pos_y < limits.lim_r);
    }

    #[test]
    fn rt8_radial_cap_is_unbounded_when_the_polynomial_folds() {
        // Strong barrel: r·radial(r) peaks below the corner radius, so there
        // is no crossing and the cap must not engage.
        let p = RadialTangential8Params {
            k1: -0.5,
            ..Default::default()
        };
        let img = glam::uvec2(1920, 1080);
        let pinhole = PinholeParams {
            fx: 500.0,
            fy: 500.0,
            cx: 960.0,
            cy: 540.0,
        };
        let limits = calculate_jacobian_clamp_limits(img, pinhole, RadialTangential8(p));
        assert_eq!(limits.lim_r, f32::MAX);
        assert_eq!(rt8_undistort_corner_radius(10.0, &p), None);

        // Other models leave the field unused.
        for model in [Pinhole, KannalaBrandt4(ULTRAWIDE)] {
            assert_eq!(
                calculate_jacobian_clamp_limits(img, pinhole, model).lim_r,
                f32::MAX
            );
        }
    }

    #[test]
    fn with_pinhole_only_swaps_the_model() {
        let cam = Camera::new(
            glam::vec3(1.0, 2.0, 3.0),
            glam::Quat::from_rotation_y(0.3),
            1.2,
            0.9,
            glam::vec2(0.4, 0.6),
            KannalaBrandt4(ULTRAWIDE),
        );
        let pin = cam.with_pinhole();
        assert!(matches!(pin.camera_model, Pinhole));
        assert_eq!(
            pin,
            Camera {
                camera_model: Pinhole,
                ..cam
            }
        );
    }

    #[test]
    fn equidistant_lens_has_no_fold() {
        let ident = KannalaBrandt4Params::default();
        assert_eq!(kb4_fold_theta(&ident), PI);
        assert_eq!(max_render_theta(&KannalaBrandt4(ident)), PI);
        for theta in [0.1, 1.0, 2.0, 3.0] {
            assert!((kb4_invert_d(theta, &ident) - theta).abs() < 1e-12);
        }
    }
}
