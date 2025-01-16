use core::f32;
use std::ops::Range;
use web_time::Duration;

use egui::Response;
use glam::{Affine3A, Quat, Vec2, Vec3, Vec3A};

pub struct PlayerController {
    position: Vec3,
    rotation: Quat,
}

impl PlayerController {
    pub fn new() -> Self {
        PlayerController {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }

    pub fn tick(&mut self, dt: Duration, controls: &Response, ui: &egui::Ui) {
        if controls.dragged_by(egui::PointerButton::Secondary)
            || controls.dragged_by(egui::PointerButton::Primary) && ui.input(|r| r.modifiers.alt)
        {
        } else if controls.dragged_by(egui::PointerButton::Primary) {
            let axis = controls.drag_delta();

            let mouselook_speed = 0.0025;
            // First, handle yaw (left/right rotation around Y axis)
            let yaw = Quat::from_rotation_y(axis.x * mouselook_speed);
            // Then handle pitch (up/down rotation around X axis)
            let pitch = Quat::from_rotation_x(-axis.y * mouselook_speed);

            // Apply yaw to the current rotation
            self.rotation = yaw * self.rotation * pitch;
        }

        let move_speed = 0.05
            * if ui.input(|r| r.modifiers.shift) {
                4.0
            } else {
                1.0
            };

        if ui.input(|r| r.key_down(egui::Key::W)) {
            self.position += self.rotation * Vec3::Z * move_speed;
        }
        if ui.input(|r| r.key_down(egui::Key::A)) {
            self.position -= self.rotation * Vec3::X * move_speed;
        }
        if ui.input(|r| r.key_down(egui::Key::S)) {
            self.position -= self.rotation * Vec3::Z * move_speed;
        }
        if ui.input(|r| r.key_down(egui::Key::D)) {
            self.position += self.rotation * Vec3::X * move_speed;
        }
    }

    pub fn local_to_world(&self) -> glam::Affine3A {
        glam::Affine3A::from_rotation_translation(self.rotation, self.position)
    }
}

pub struct OrbitControls {
    pub position: Vec3A,
    pub rotation: Quat,

    pub focus: Vec3A,
    pub dirty: bool,

    pan_momentum: Vec2,
    rotate_momentum: Vec2,

    radius_range: Range<f32>,
    yaw_range: Range<f32>,
    pitch_range: Range<f32>,
}

impl OrbitControls {
    pub fn new(
        radius: f32,
        radius_range: Range<f32>,
        yaw_range: Range<f32>,
        pitch_range: Range<f32>,
    ) -> Self {
        Self {
            position: -Vec3A::Z * radius,
            rotation: Quat::IDENTITY,
            focus: Vec3A::ZERO,
            pan_momentum: Vec2::ZERO,
            rotate_momentum: Vec2::ZERO,
            dirty: false,
            radius_range,
            yaw_range,
            pitch_range,
        }
    }

    pub fn radius(&self) -> f32 {
        (self.position - self.focus).length()
    }

    fn clamp_smooth(val: f32, range: Range<f32>) -> f32 {
        let mut val = val;
        if val < range.start {
            val = val * 0.5 + range.start * 0.5;
        }

        if val > range.end {
            val = val * 0.5 + range.end * 0.5;
        }
        val
    }

    pub fn pan_orbit_camera(
        &mut self,
        pan: Vec2,
        rotate: Vec2,
        scroll: f32,
        window: Vec2,
        delta_time: f32,
    ) -> bool {
        let (yaw, pitch, roll) = self.rotation.to_euler(glam::EulerRot::YXZ);

        let mut radius = self.radius();

        // Adjust momentum with the new input
        self.pan_momentum += pan;
        self.rotate_momentum += rotate;

        // Apply damping to the momentum
        let damping = 0.0005f32.powf(delta_time);
        self.pan_momentum *= damping;
        self.rotate_momentum *= damping;

        // Update velocities based on momentum
        let pan_velocity = self.pan_momentum * delta_time;
        let rotate_velocity = self.rotate_momentum * delta_time;

        let delta_x = rotate_velocity.x * std::f32::consts::PI * 2.0 / window.x;
        let delta_y = rotate_velocity.y * std::f32::consts::PI / window.y;

        let yaw = Self::clamp_smooth(yaw + delta_x, self.yaw_range.clone());
        let pitch = Self::clamp_smooth(pitch - delta_y, self.pitch_range.clone());

        self.rotation =
            Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch) * Quat::from_rotation_z(roll);

        let scaled_pan = pan_velocity * Vec2::new(1.0 / window.x, 1.0 / window.y);

        let right = self.rotation * Vec3A::X * -scaled_pan.x;
        let up = self.rotation * Vec3A::Y * -scaled_pan.y;
        let translation = (right + up) * radius;

        self.focus += translation;
        radius -= scroll * radius * 0.2;
        radius = Self::clamp_smooth(radius, self.radius_range.clone());
        self.position = self.focus + self.rotation * Vec3A::new(0.0, 0.0, -radius);

        scroll.abs() > 0.0
            || pan.length_squared() > 0.0
            || rotate.length_squared() > 0.0
            || self.pan_momentum.length_squared() > 0.001
            || self.rotate_momentum.length_squared() > 0.001
            || self.dirty
    }

    pub(crate) fn transform(&self) -> Affine3A {
        Affine3A::from_rotation_translation(self.rotation, self.position.into())
    }
}
