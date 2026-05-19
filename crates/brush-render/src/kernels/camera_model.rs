use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

use crate::camera::{KANNALA_BRANDT_4, PINHOLE};
use brush_cube::{Mat2x3, Vec3A};

#[derive(CubeLaunch, CubeType, Debug, Clone, Copy)]
pub struct CameraParams {
    pub focal_x: f32,
    pub focal_y: f32,
    pub pixel_center_x: f32,
    pub pixel_center_y: f32,
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub k4: f32,
}

#[cube]
impl CameraParams {
    pub fn project(&self, point: Vec3A, #[comptime] camera_model_id: i32) -> (f32, f32) {
        let inv_z = 1.0f32 / point.z();
        let pinhole_u = self.focal_x * point.x() * inv_z + self.pixel_center_x;
        let pinhole_v = self.focal_y * point.y() * inv_z + self.pixel_center_y;

        if comptime![camera_model_id == PINHOLE] {
            (pinhole_u, pinhole_v)
        } else if comptime![camera_model_id == KANNALA_BRANDT_4] {
            let r = f32::sqrt(point.x() * point.x() + point.y() * point.y());

            let near_axis = r < 1e-6f32;

            let theta = r.atan2(point.z());
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta2 * theta4;
            let theta8 = theta4 * theta4;
            let d_theta = theta
                * (1.0f32
                    + self.k1 * theta2
                    + self.k2 * theta4
                    + self.k3 * theta6
                    + self.k4 * theta8);
            let inv_r = 1.0f32 / r;
            let fisheye_u = self.focal_x * (d_theta * point.x() * inv_r) + self.pixel_center_x;
            let fisheye_v = self.focal_y * (d_theta * point.y() * inv_r) + self.pixel_center_y;

            (
                select(near_axis, pinhole_u, fisheye_u),
                select(near_axis, pinhole_v, fisheye_v),
            )
        } else {
            panic!("not implemented")
        }
    }

    /// Computes the Jacobian of the projection w.r.t. the projected 3d point
    /// `J = (3x2)`, returned column-major as `(j00, j01, j10, j11, j20, j21)`.
    /// Caller can drop the first half if they only need the trailing column
    /// for the perspective vjp.
    pub fn calc_jacobian(
        &self,
        point: Vec3A,
        img_w: u32,
        img_h: u32,
        #[comptime] camera_model_id: i32,
    ) -> Mat2x3 {
        let x = point.x();
        let y = point.y();
        let z = point.z();

        let inv_z = 1.0f32 / point.z();
        let dx = self.focal_x * inv_z;
        let dy = self.focal_y * inv_z;

        let img_w_f = img_w as f32;
        let img_h_f = img_h as f32;
        let lim_pos_x = (1.15f32 * img_w_f - self.pixel_center_x) / self.focal_x;
        let lim_pos_y = (1.15f32 * img_h_f - self.pixel_center_y) / self.focal_y;
        let lim_neg_x = (-0.15f32 * img_w_f - self.pixel_center_x) / self.focal_x;
        let lim_neg_y = (-0.15f32 * img_h_f - self.pixel_center_y) / self.focal_y;

        let pinhole_u = clamp(x * inv_z, lim_neg_x, lim_pos_x);
        let pinhole_v = clamp(y * inv_z, lim_neg_y, lim_pos_y);

        if comptime![camera_model_id == PINHOLE] {
            Mat2x3 {
                c0_x: dx,
                c0_y: 0.0,
                c1_x: 0.0,
                c1_y: dy,
                c2_x: -dx * pinhole_u,
                c2_y: -dy * pinhole_v,
            }
        } else if comptime![camera_model_id == KANNALA_BRANDT_4] {
            let x2 = x * x;
            let y2 = y * y;
            let r2 = x2 + y2;
            let r = r2.sqrt();

            let near_axis = r < 1e-6f32;

            let theta = r.atan2(z);
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;
            let d = theta
                * (1.0f32
                    + self.k1 * theta2
                    + self.k2 * theta4
                    + self.k3 * theta6
                    + self.k4 * theta8);

            let inv_r = 1.0f32 / r;
            let t7 = inv_r * d;
            let t8 = d / (r2 * r);
            let t9 = self.focal_x * t8;
            let t10 = inv_r * x;
            let t11 = 1.0f32 / (r2 + z * z);
            let t12 = t11 * z;
            let t13 = t10 * t12;
            let t14 = 3.0f32 * self.k1 * theta2;
            let t15 = 5.0f32 * self.k2 * theta4;
            let t16 = 7.0f32 * self.k3 * theta6;
            let t17 = 9.0f32 * self.k4 * theta8;
            let t18 = t13 * t14 + t13 * t15 + t13 * t16 + t13 * t17 + t13;
            let t19 = self.focal_x * t10;
            let t20 = x * y;
            let t21 = inv_r * y;
            let t22 = t12 * t21;
            let t23 = t14 * t22 + t15 * t22 + t16 * t22 + t17 * t22 + t22;
            let t24 = t11 * r;
            let t25 = -t14 * t24 - t15 * t24 - t16 * t24 - t17 * t24 - t24;
            let t26 = self.focal_y * t8;
            let t27 = self.focal_y * t21;
            let du_dx = self.focal_x * t7 - x2 * t9 + t18 * t19;
            let du_dy = self.focal_x * t23 * inv_r * x - t20 * t9;
            let du_dz = t19 * t25;
            let dv_dx = self.focal_y * t18 * inv_r * y - t20 * t26;
            let dv_dy = self.focal_y * t7 - y2 * t26 + t23 * t27;
            let dv_dz = t25 * t27;

            Mat2x3 {
                c0_x: select(near_axis, dx, du_dx),
                c0_y: select(near_axis, 0.0, dv_dx),
                c1_x: select(near_axis, 0.0, du_dy),
                c1_y: select(near_axis, dy, dv_dy),
                c2_x: select(near_axis, -dx * pinhole_u, du_dz),
                c2_y: select(near_axis, -dy * pinhole_v, dv_dz),
            }
        } else {
            panic!("not implemented")
        }
    }
}

impl CameraParams {
    pub fn to_launch_object<R: Runtime>(&self) -> CameraParamsLaunch<R> {
        CameraParamsLaunch::new(
            self.focal_x,
            self.focal_y,
            self.pixel_center_x,
            self.pixel_center_y,
            self.k1,
            self.k2,
            self.k3,
            self.k4,
        )
    }
}
