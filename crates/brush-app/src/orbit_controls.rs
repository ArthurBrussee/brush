use core::f32;

use egui::Response;
use glam::{Quat, Vec2, Vec3};

pub struct CameraController {
    pub position: Vec3,
    pub rotation: Quat,
    roll: Quat,
    focus_distance: f32,
    fly_velocity: Vec3,
    orbit_velocity: Vec2,
}

pub fn smooth_orbit(
    position: Vec3,
    rotation: Quat,
    base_roll: Quat,
    delta_x: f32,
    delta_y: f32,
    distance: f32,
) -> (Vec3, Quat) {
    // Calculate focal point (where we're looking at)
    let focal_point = position + rotation * Vec3::Z * distance;

    // Create rotation quaternions in camera's local space
    let pitch = Quat::from_axis_angle(rotation * Vec3::X, delta_x);
    let yaw = Quat::from_axis_angle(base_roll * Vec3::Y, delta_y);

    // Apply yaw in world space, pitch in local space
    let new_rotation = (yaw * pitch * rotation).normalize();

    // Calculate new position by backing up from focal point
    let new_position = focal_point - new_rotation * Vec3::Z * distance;

    (new_position, new_rotation)
}

fn exp_lerp2(a: Vec2, b: Vec2, dt: f32, lambda: f32) -> Vec2 {
    let lerp_exp = (-lambda * dt).exp();
    a * lerp_exp + b * (1.0 - lerp_exp)
}

fn exp_lerp3(a: Vec3, b: Vec3, dt: f32, lambda: f32) -> Vec3 {
    let lerp_exp = (-lambda * dt).exp();
    a * lerp_exp + b * (1.0 - lerp_exp)
}

impl CameraController {
    pub fn new(start_focus_distance: f32) -> Self {
        Self {
            position: -Vec3::Z * start_focus_distance,
            rotation: Quat::IDENTITY,
            roll: Quat::IDENTITY,
            focus_distance: start_focus_distance,
            fly_velocity: Vec3::ZERO,
            orbit_velocity: Vec2::ZERO,
        }
    }

    pub fn tick(&mut self, response: &Response, ui: &egui::Ui) {
        let delta_time = ui.input(|r| r.predicted_dt);

        let lmb = response.dragged_by(egui::PointerButton::Primary);
        let rmb = response.dragged_by(egui::PointerButton::Secondary);
        let mmb = response.dragged_by(egui::PointerButton::Middle);

        let look_pan = mmb || lmb && ui.input(|r| r.modifiers.ctrl);
        let look_fps = rmb || lmb && ui.input(|r| r.key_down(egui::Key::Space));
        let look_orbit = lmb;

        let mouselook_speed = 0.002;

        let right = self.rotation * Vec3::X;
        let up = self.rotation * Vec3::Y;
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
            self.position -= right * response.drag_delta().x * drag_mult;
            self.position -= up * response.drag_delta().y * drag_mult;

            ui.ctx().set_cursor_icon(egui::CursorIcon::Move);
        } else if look_fps {
            let axis = response.drag_delta();
            let yaw = Quat::from_axis_angle(self.roll * Vec3::Y, axis.x * mouselook_speed);
            let pitch = Quat::from_rotation_x(-axis.y * mouselook_speed);
            self.rotation = yaw * self.rotation * pitch;
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        } else if look_orbit {
            let dx = response.drag_delta().x * mouselook_speed;
            let dy = -response.drag_delta().y * mouselook_speed;

            self.orbit_velocity = glam::vec2(dy, dx);
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }

        (self.position, self.rotation) = smooth_orbit(
            self.position,
            self.rotation,
            self.roll,
            self.orbit_velocity.x,
            self.orbit_velocity.y,
            self.focus_distance,
        );

        let fly_moment_lambda = 0.8;

        let move_speed = 30.0
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

        if ui.input(|r| r.modifiers.alt) {
            // Roll with alt + Q&E.
            if ui.input(|r| r.key_down(egui::Key::Q)) {
                let roll = Quat::from_axis_angle(forward, move_speed * 0.025 * delta_time);
                self.rotation = roll * self.rotation;
                self.roll = roll * self.roll;
            }
            if ui.input(|r| r.key_down(egui::Key::E)) {
                let roll = Quat::from_axis_angle(forward, -move_speed * 0.025 * delta_time);
                self.rotation = roll * self.rotation;
                self.roll = roll * self.roll;
            }
        } else {
            // Move _down_ with Q
            if ui.input(|r| r.key_down(egui::Key::Q)) {
                self.fly_velocity = exp_lerp3(
                    self.fly_velocity,
                    Vec3::Y * move_speed,
                    delta_time,
                    fly_moment_lambda,
                );
            }
            // Move up with E
            if ui.input(|r| r.key_down(egui::Key::E)) {
                self.fly_velocity = exp_lerp3(
                    self.fly_velocity,
                    -Vec3::Y * move_speed,
                    delta_time,
                    fly_moment_lambda,
                );
            }
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

        // Scroll speed depends on how far zoomed out we are.
        self.focus_distance -= scrolled * scroll_speed * self.focus_distance;
        self.focus_distance = self.focus_distance.max(0.01);

        self.position = old_pivot - (self.rotation * Vec3::Z * self.focus_distance);
    }

    pub fn local_to_world(&self) -> glam::Affine3A {
        glam::Affine3A::from_rotation_translation(self.rotation, self.position)
    }
}
