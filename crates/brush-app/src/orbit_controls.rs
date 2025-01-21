use core::f32;

use egui::Response;
use glam::{Quat, Vec3};

pub struct CameraController {
    pub position: Vec3,
    pub rotation: Quat,

    roll: Quat,
    focus_distance: f32,
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

impl CameraController {
    pub fn new(start_focus_distance: f32) -> Self {
        Self {
            position: -Vec3::Z * start_focus_distance,
            rotation: Quat::IDENTITY,
            roll: Quat::IDENTITY,
            focus_distance: start_focus_distance,
        }
    }

    pub fn tick(&mut self, response: &Response, ui: &egui::Ui) {
        let move_speed = 5.0
            * ui.input(|r| r.predicted_dt)
            * if ui.input(|r| r.modifiers.shift) {
                4.0
            } else {
                1.0
            };

        let lmb = response.dragged_by(egui::PointerButton::Primary);

        let is_panning = lmb && ui.input(|r| r.key_down(egui::Key::Space));
        let is_orbiting = lmb && ui.input(|r| r.modifiers.alt);
        let is_flythrough = lmb && !is_panning && !is_orbiting;

        let mouselook_speed = 0.0025;

        let right = self.rotation * Vec3::X;
        let up = self.rotation * Vec3::Y;
        let forward = self.rotation * Vec3::Z;

        if is_orbiting {
            let dx = response.drag_delta().x * mouselook_speed;
            let dy = -response.drag_delta().y * mouselook_speed;

            (self.position, self.rotation) = smooth_orbit(
                self.position,
                self.rotation,
                self.roll,
                dy,
                dx,
                self.focus_distance,
            );
        } else if is_panning {
            let drag_mult = 0.1;
            self.position -= right * response.drag_delta().x * drag_mult * move_speed;
            self.position -= up * response.drag_delta().y * drag_mult * move_speed;
        } else if is_flythrough {
            let axis = response.drag_delta();
            let yaw = Quat::from_axis_angle(self.roll * Vec3::Y, axis.x * mouselook_speed);
            let pitch = Quat::from_rotation_x(-axis.y * mouselook_speed);
            self.rotation = yaw * self.rotation * pitch;
        }

        // In Unity, this is only enabled when the camera is in flythrough mode.
        // for our purposes... just enable it.
        if ui.input(|r| r.key_down(egui::Key::W) || r.key_down(egui::Key::ArrowUp)) {
            self.position += forward * move_speed;
        }
        if ui.input(|r| r.key_down(egui::Key::A) || r.key_down(egui::Key::ArrowLeft)) {
            self.position -= right * move_speed;
        }
        if ui.input(|r| r.key_down(egui::Key::S) || r.key_down(egui::Key::ArrowDown)) {
            self.position -= forward * move_speed;
        }
        if ui.input(|r| r.key_down(egui::Key::D) || r.key_down(egui::Key::ArrowRight)) {
            self.position += right * move_speed;
        }

        // Move _down_ with Q
        if ui.input(|r| r.key_down(egui::Key::Q)) {
            self.position += up * move_speed;
        }
        // Move up with E
        if ui.input(|r| r.key_down(egui::Key::E)) {
            self.position -= up * move_speed;
        }

        // Roll with alt + Q&E.
        if ui.input(|r| r.key_down(egui::Key::R)) {
            let roll = Quat::from_axis_angle(forward, move_speed * 0.05);
            self.rotation = roll * self.rotation;
            self.roll = roll * self.roll;
        }
        if ui.input(|r| r.key_down(egui::Key::T)) {
            let roll = Quat::from_axis_angle(forward, -move_speed * 0.05);
            self.rotation = roll * self.rotation;
            self.roll = roll * self.roll;
        }
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
