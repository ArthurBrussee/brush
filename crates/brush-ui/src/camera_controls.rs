use core::f32;

use egui::{Event, Response};
use glam::{Quat, Vec2, Vec3};

use crate::app::CameraSettings;

#[derive(Clone, Default, PartialEq)]
pub struct CameraClamping {
    pub min_focus_distance: Option<f32>,
    pub max_focus_distance: Option<f32>,
    pub min_pitch: Option<f32>,
    pub max_pitch: Option<f32>,
    pub min_yaw: Option<f32>,
    pub max_yaw: Option<f32>,
}

pub struct CameraController {
    pub position: Vec3,
    pub rotation: Quat,
    pub focus_distance: f32,
    pub settings: CameraSettings,

    roll: Quat,
    fly_velocity: Vec3,
    orbit_velocity: Vec2,
}

pub fn smooth_orbit(
    position: Vec3,
    rotation: Quat,
    base_roll: Quat,
    delta_yaw: f32,
    delta_pitch: f32,
    clamping: &CameraClamping,
    dt: f32,
    distance: f32,
) -> (Vec3, Quat) {
    // Calculate focal point (where we're looking at)
    let focal_point = position + rotation * Vec3::Z * distance;

    // Extract current pitch angle from rotation
    let forward = rotation * Vec3::Z;
    let current_pitch = -forward.y.asin();

    // Clamp the new pitch angle
    let new_pitch = smooth_clamp(
        current_pitch - delta_pitch,
        clamping.min_pitch.map(|x| x.to_radians()),
        clamping.max_pitch.map(|x| x.to_radians()),
        dt,
        50.0,
    );

    let delta_pitch = current_pitch - new_pitch;
    let pitch = Quat::from_axis_angle(rotation * Vec3::X, -delta_pitch);

    let forward_proj = Vec3::new(forward.x, 0.0, forward.z).normalize();
    let current_yaw = (-forward_proj.x).atan2(forward_proj.z);

    // Clamp the new yaw angle
    let new_yaw = smooth_clamp(
        current_yaw - delta_yaw,
        clamping.min_yaw.map(|x| x.to_radians()),
        clamping.max_yaw.map(|x| x.to_radians()),
        dt,
        50.0,
    );

    // New delta clamped to not go over min/max yaw
    let delta_yaw = current_yaw - new_yaw;

    // Create yaw rotation quaternion
    let yaw_axis = base_roll * Vec3::NEG_Y;
    let yaw = Quat::from_axis_angle(yaw_axis, -delta_yaw);

    let new_rotation = (yaw * pitch * rotation).normalize();
    let new_position = focal_point - new_rotation * Vec3::Z * distance;

    (new_position, new_rotation)
}

fn exp_lerp(a: f32, b: f32, dt: f32, lambda: f32) -> f32 {
    let lerp_exp = (-lambda * dt).exp();
    a * lerp_exp + b * (1.0 - lerp_exp)
}

fn exp_lerp2(a: Vec2, b: Vec2, dt: f32, lambda: f32) -> Vec2 {
    glam::vec2(
        exp_lerp(a.x, b.x, dt, lambda),
        exp_lerp(a.y, b.y, dt, lambda),
    )
}

fn exp_lerp3(a: Vec3, b: Vec3, dt: f32, lambda: f32) -> Vec3 {
    glam::vec3(
        exp_lerp(a.x, b.x, dt, lambda),
        exp_lerp(a.y, b.y, dt, lambda),
        exp_lerp(a.z, b.z, dt, lambda),
    )
}

fn smooth_clamp(val: f32, min: Option<f32>, max: Option<f32>, dt: f32, lambda: f32) -> f32 {
    let mut target = val;
    if let Some(min) = min {
        target = target.max(min);
    }
    if let Some(max) = max {
        target = target.min(max);
    }
    exp_lerp(val, target, dt, lambda)
}

impl CameraController {
    pub fn new(position: Vec3, rotation: Quat, settings: CameraSettings) -> Self {
        Self {
            position,
            rotation,
            focus_distance: 4.0,
            settings,
            roll: Quat::IDENTITY,
            fly_velocity: Vec3::ZERO,
            orbit_velocity: Vec2::ZERO,
        }
    }

    pub fn tick(&mut self, response: &Response, ui: &egui::Ui) {
        let delta_time = ui.input(|r| r.predicted_dt);

        // Check for two-finger touch panning
        let multi_touch = ui.input(|r| r.multi_touch());
        let has_multi_touch = multi_touch.is_some();

        let mut mouse_delta = ui
            .input(|r| r.pointer.motion())
            .unwrap_or(ui.input(|r| r.pointer.delta()));

        // Ignore any delta when new touch iis detected as atm that is glitchy in egui, see
        // https://github.com/emilk/egui/issues/5550
        if ui.input(|r| {
            r.events
                .iter()
                .any(|ev| matches!(ev, Event::PointerButton { .. }))
        }) {
            mouse_delta = egui::Vec2::ZERO;
        }

        let lmb = response.dragged_by(egui::PointerButton::Primary);
        let rmb = response.dragged_by(egui::PointerButton::Secondary);
        let mmb = response.dragged_by(egui::PointerButton::Middle);

        let look_pan = mmb || (lmb && ui.input(|r| r.modifiers.ctrl)) || has_multi_touch;
        let look_fps =
            (rmb || (lmb && ui.input(|r| r.key_down(egui::Key::Space)))) && !has_multi_touch;
        let look_orbit = lmb && !look_pan && !look_fps;

        let mouselook_speed = 0.002;

        let right = self.rotation * Vec3::X;
        let up = self.rotation * Vec3::NEG_Y;
        let forward = self.rotation * Vec3::Z;

        if response.hovered() {
            if ui.input(|r| r.modifiers.ctrl) {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Move);
            } else if ui.input(|r| r.key_down(egui::Key::Space)) {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            } else {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
        }

        if look_pan {
            let drag_mult = self.focus_distance / response.rect.width().max(response.rect.height());

            if let Some(multi_touch) = multi_touch {
                // Use multi-touch translation for two-finger panning
                let translation = multi_touch.translation_delta;
                self.position -= right * translation.x * drag_mult;
                self.position += up * translation.y * drag_mult;
            } else {
                // Use mouse drag for single-pointer panning
                self.position -= right * mouse_delta.x * drag_mult;
                self.position += up * mouse_delta.y * drag_mult;
            }
            ui.ctx().set_cursor_icon(egui::CursorIcon::Move);
        } else if look_fps {
            let axis = response.drag_delta();
            let yaw = Quat::from_axis_angle(self.roll * Vec3::NEG_Y, -axis.x * mouselook_speed);
            let pitch = Quat::from_rotation_x(-axis.y * mouselook_speed);
            self.rotation = yaw * self.rotation * pitch;
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        } else if look_orbit {
            let delta_yaw = mouse_delta.x * mouselook_speed;
            let delta_pitch = mouse_delta.y * mouselook_speed;
            self.orbit_velocity = glam::vec2(delta_yaw, delta_pitch);
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }

        (self.position, self.rotation) = smooth_orbit(
            self.position,
            self.rotation,
            self.roll,
            self.orbit_velocity.x,
            self.orbit_velocity.y,
            &self.settings.clamping,
            delta_time,
            self.focus_distance,
        );

        let fly_moment_lambda = 0.8;

        let move_speed = 25.0
            * self.settings.speed_scale.unwrap_or(1.0)
            * if ui.input(|r| r.modifiers.shift) {
                4.0
            } else {
                1.0
            };

        if ui.input(|r| r.key_down(egui::Key::W) || r.key_down(egui::Key::ArrowUp)) {
            self.fly_velocity = exp_lerp3(
                self.fly_velocity,
                Vec3::Z * move_speed,
                delta_time,
                fly_moment_lambda,
            );
        }
        if ui.input(|r| r.key_down(egui::Key::A) || r.key_down(egui::Key::ArrowLeft)) {
            self.fly_velocity = exp_lerp3(
                self.fly_velocity,
                -Vec3::X * move_speed,
                delta_time,
                fly_moment_lambda,
            );
        }
        if ui.input(|r| r.key_down(egui::Key::S) || r.key_down(egui::Key::ArrowDown)) {
            self.fly_velocity = exp_lerp3(
                self.fly_velocity,
                -Vec3::Z * move_speed,
                delta_time,
                fly_moment_lambda,
            );
        }
        if ui.input(|r| r.key_down(egui::Key::D) || r.key_down(egui::Key::ArrowRight)) {
            self.fly_velocity = exp_lerp3(
                self.fly_velocity,
                Vec3::X * move_speed,
                delta_time,
                fly_moment_lambda,
            );
        }

        // Move _down_ with Q
        if ui.input(|r| r.key_down(egui::Key::Q)) {
            self.fly_velocity = exp_lerp3(
                self.fly_velocity,
                -Vec3::Y * move_speed,
                delta_time,
                fly_moment_lambda,
            );
        }
        // Move up with E
        if ui.input(|r| r.key_down(egui::Key::E)) {
            self.fly_velocity = exp_lerp3(
                self.fly_velocity,
                Vec3::Y * move_speed,
                delta_time,
                fly_moment_lambda,
            );
        }

        if ui.input(|r| r.key_down(egui::Key::Z)) {
            let roll = Quat::from_axis_angle(forward, move_speed * 0.025 * delta_time);
            self.rotation = roll * self.rotation;
            self.roll = roll * self.roll;
        }
        if ui.input(|r| r.key_down(egui::Key::X)) {
            self.rotation = self.roll.inverse() * self.rotation;
            self.roll = Quat::IDENTITY;
        }
        if ui.input(|r| r.key_down(egui::Key::C)) {
            let roll = Quat::from_axis_angle(forward, -move_speed * 0.025 * delta_time);
            self.rotation = roll * self.rotation;
            self.roll = roll * self.roll;
        }

        let delta = self.fly_velocity * delta_time;
        self.position += delta.x * right + delta.y * up + delta.z * forward;

        // Damp velocities towards zero.
        self.orbit_velocity = exp_lerp2(self.orbit_velocity, Vec2::ZERO, delta_time, 8.0);
        self.fly_velocity = exp_lerp3(self.fly_velocity, Vec3::ZERO, delta_time, 7.0);

        // Handle scroll wheel: move back, and adjust focus distance.
        let scrolled = ui.input(|r| r.smooth_scroll_delta.y);
        let scroll_speed = 0.001;

        let old_pivot = self.position + self.rotation * Vec3::Z * self.focus_distance;

        // Handle pinch-to-zoom from multi-touch
        let mut zoom_delta = 0.0;
        if let Some(multi_touch) = multi_touch {
            // Convert zoom factor to distance change - try reversing the direction
            let zoom_factor = multi_touch.zoom_delta;
            if zoom_factor != 1.0 {
                zoom_delta = (zoom_factor - 1.0) * self.focus_distance * 2.0;
            }
        }

        // Scroll speed depends on how far zoomed out we are.
        self.focus_distance -= scrolled * scroll_speed * self.focus_distance + zoom_delta;
        self.focus_distance = self.focus_distance.max(0.01);

        self.focus_distance = smooth_clamp(
            self.focus_distance,
            self.settings.clamping.min_focus_distance,
            self.settings.clamping.max_focus_distance,
            delta_time,
            50.5,
        );

        self.position = old_pivot - (self.rotation * Vec3::Z * self.focus_distance);
    }

    pub fn local_to_world(&self) -> glam::Affine3A {
        glam::Affine3A::from_rotation_translation(self.rotation, self.position)
    }

    pub fn stop_movement(&mut self) {
        self.orbit_velocity = Vec2::ZERO;
        self.fly_velocity = Vec3::ZERO;
    }
}
