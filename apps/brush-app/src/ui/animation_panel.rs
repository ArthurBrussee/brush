use brush_anim::{AnimationConfig, Interpolation};
use egui::{Ui, WidgetText};
use web_time::Instant;

#[cfg(not(target_family = "wasm"))]
use crate::ui::animation_export::Exporter;
use crate::ui::panels::AppPane;
use crate::ui::ui_process::UiProcess;

#[cfg(not(target_family = "wasm"))]
use std::sync::{Arc, Mutex};

const MARKER_RADIUS: f32 = 10.0;

#[derive(Default)]
pub struct AnimationPanel {
    /// Serializable animation state (keyframes, timeline, camera).
    config: AnimationConfig,
    current_frame: usize,
    /// When playing: (wall-clock start, frame playback started from).
    playback: Option<(Instant, usize)>,
    /// Last frame applied to the camera, so we only update on change.
    applied_frame: Option<usize>,
    #[cfg(not(target_family = "wasm"))]
    exporter: Exporter,
    #[cfg(not(target_family = "wasm"))]
    config_io: ConfigIo,
}

impl AppPane for AnimationPanel {
    fn title(&self) -> WidgetText {
        "Animation".into()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        !process.current_splats().is_empty() && !process.is_training()
    }

    fn ui(&mut self, ui: &mut Ui, process: &UiProcess) {
        #[cfg(not(target_family = "wasm"))]
        if let Some(config) = self.config_io.take_loaded() {
            self.config = config;
            self.apply_config_to_viewer(process);
            self.playback = None;
            self.applied_frame = None;
        }

        let min_frames = self.config.min_frames();
        self.config.num_frames = self.config.num_frames.max(min_frames);

        let last_frame = self.config.num_frames.saturating_sub(1);
        self.current_frame = self.current_frame.min(last_frame);
        self.advance_playback(ui, last_frame);

        self.frame_slider(ui, last_frame);
        self.keyframe_timeline(ui, last_frame);

        ui.add_space(8.0);
        self.toolbar(ui, min_frames, process);

        if self.applied_frame != Some(self.current_frame) {
            self.apply_camera_for_frame(process, self.current_frame);
            self.applied_frame = Some(self.current_frame);
        }

        #[cfg(not(target_family = "wasm"))]
        self.export_popup(ui, process);
    }
}

impl AnimationPanel {
    /// Advances `current_frame` from wall-clock time so playback speed tracks
    /// the FPS regardless of the display refresh rate, looping at the end.
    fn advance_playback(&mut self, ui: &Ui, last_frame: usize) {
        let Some((start, start_frame)) = self.playback else {
            return;
        };
        let advanced = (start.elapsed().as_secs_f32() * self.config.fps as f32) as usize;
        self.current_frame = (start_frame + advanced) % (last_frame + 1);
        ui.ctx().request_repaint();
    }

    fn frame_slider(&mut self, ui: &mut Ui, last_frame: usize) {
        ui.label("Current frame");
        ui.spacing_mut().slider_width = ui.available_width();
        ui.add(
            egui::Slider::new(&mut self.current_frame, 0..=last_frame)
                .integer()
                .show_value(false),
        );
    }

    fn toolbar(&mut self, ui: &mut Ui, min_frames: usize, process: &UiProcess) {
        ui.horizontal(|ui| {
            let playing = self.playback.is_some();
            if ui
                .add_enabled(!playing, egui::Button::new("▶"))
                .on_hover_text("Play the animation")
                .clicked()
            {
                self.playback = Some((Instant::now(), self.current_frame));
            }
            if ui
                .add_enabled(playing, egui::Button::new("⏹"))
                .on_hover_text("Stop the animation")
                .clicked()
            {
                self.playback = None;
            }

            ui.separator();

            if ui
                .button("➕")
                .on_hover_text("Add keyframe at the current frame")
                .clicked()
            {
                let camera = process.current_camera();
                self.config
                    .set_keyframe(self.current_frame, camera.position, camera.rotation);
            }
            let has_here = self
                .config
                .keyframes
                .iter()
                .any(|k| k.frame == self.current_frame);
            if ui
                .add_enabled(has_here, egui::Button::new("➖"))
                .on_hover_text("Remove the keyframe at the current frame")
                .clicked()
            {
                self.config
                    .keyframes
                    .retain(|k| k.frame != self.current_frame);
            }

            ui.separator();

            let prev_interp = self.config.interpolation;
            egui::ComboBox::from_id_salt("interpolation")
                .selected_text(self.config.interpolation.label())
                .show_ui(ui, |ui| {
                    for method in Interpolation::ALL {
                        ui.selectable_value(&mut self.config.interpolation, method, method.label());
                    }
                });
            // Re-apply the camera so the new method takes effect immediately.
            if self.config.interpolation != prev_interp {
                self.applied_frame = None;
            }

            #[cfg(not(target_family = "wasm"))]
            {
                ui.separator();
                self.config_buttons(ui, process);
            }

            // FPS / frame-count editors, pushed to the right boundary. In a
            // right-to-left layout the first widget sits furthest right.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.config.num_frames)
                        .range(min_frames..=100_000)
                        .speed(1),
                );
                ui.label("Frames");
                ui.add(
                    egui::DragValue::new(&mut self.config.fps)
                        .range(1..=60)
                        .speed(1),
                );
                ui.label("FPS");
            });
        });
    }

    /// Save / load / export buttons (native only, since they touch the
    /// filesystem and, for export, ffmpeg).
    #[cfg(not(target_family = "wasm"))]
    fn config_buttons(&mut self, ui: &mut Ui, process: &UiProcess) {
        if ui
            .button("Save…")
            .on_hover_text("Save the animation to a config file")
            .clicked()
        {
            self.sync_scene_into_config(process);
            self.config_io.save(self.config.clone());
        }
        if ui
            .button("Load…")
            .on_hover_text("Load an animation config file")
            .clicked()
        {
            self.config_io.load(ui.ctx().clone());
        }
        if ui
            .add_enabled(
                !self.config.keyframes.is_empty(),
                egui::Button::new("Export…"),
            )
            .on_hover_text("Export the animation as an MP4 video")
            .clicked()
        {
            self.exporter.open();
        }
    }

    /// Captures the current viewer camera/scene state into the config so it can
    /// be reproduced on export or load.
    #[cfg(not(target_family = "wasm"))]
    fn sync_scene_into_config(&mut self, process: &UiProcess) {
        let camera = process.current_camera();
        self.config.fov_x = camera.fov_x;
        self.config.fov_y = camera.fov_y;
        self.config.center_uv = camera.center_uv;
        self.config.model_local_to_world = process.model_local_to_world();
        let settings = process.get_cam_settings();
        self.config.background = settings.background.unwrap_or(glam::Vec3::ZERO);
        self.config.splat_scale = settings.splat_scale;
    }

    /// Pushes the loaded config's scene snapshot into the viewer, so live
    /// playback matches what the config was authored with (and what export
    /// produces). The inverse of [`Self::sync_scene_into_config`].
    #[cfg(not(target_family = "wasm"))]
    fn apply_config_to_viewer(&self, process: &UiProcess) {
        process.set_model_local_to_world(self.config.model_local_to_world);
        process.set_cam_fov(self.config.fov_y);
        let mut settings = process.get_cam_settings();
        settings.background = Some(self.config.background);
        settings.splat_scale = self.config.splat_scale;
        process.set_cam_settings(&settings);
    }

    /// Drives the live camera to the interpolated pose for `frame`.
    fn apply_camera_for_frame(&self, process: &UiProcess, frame: usize) {
        if let Some((position, rotation)) = self.config.pose_at_frame(frame) {
            process.set_cam_transform(position, rotation);
        }
    }

    /// Draws the export popup and starts an export when requested.
    #[cfg(not(target_family = "wasm"))]
    fn export_popup(&mut self, ui: &Ui, process: &UiProcess) {
        let Some(pending) = self.exporter.draw(ui) else {
            return;
        };
        let Some(splats) = process.current_splats().latest() else {
            return;
        };

        self.sync_scene_into_config(process);
        let cameras = self.config.render_cameras(pending.width, pending.height);
        self.exporter.start(
            ui.ctx().clone(),
            crate::ui::animation_export::ExportJob {
                splats,
                cameras,
                fps: self.config.fps,
                background: self.config.background,
                splat_scale: self.config.splat_scale,
                width: pending.width,
                height: pending.height,
                path: pending.path,
            },
        );
    }

    fn keyframe_timeline(&mut self, ui: &mut Ui, last_frame: usize) {
        let layout = TimelineLayout::allocate(ui, last_frame);
        self.draw_track(ui, &layout);

        let (to_move, to_remove) = self.draw_markers(ui, &layout);
        if let Some((idx, new_frame)) = to_move {
            self.config.keyframes[idx].frame = new_frame;
        }
        if let Some(idx) = to_remove {
            self.config.keyframes.remove(idx);
        }
    }

    /// Draws the baseline, the playhead, and the current frame number.
    fn draw_track(&self, ui: &Ui, layout: &TimelineLayout) {
        let painter = ui.painter_at(layout.rect);
        let visuals = ui.visuals();

        painter.line_segment(
            [
                egui::pos2(layout.track_left, layout.marker_y),
                egui::pos2(layout.track_right(), layout.marker_y),
            ],
            egui::Stroke::new(2.0, visuals.widgets.inactive.bg_fill),
        );

        let playhead_x = layout.frame_to_x(self.current_frame);
        painter.line_segment(
            [
                egui::pos2(playhead_x, layout.marker_y),
                egui::pos2(playhead_x, layout.rect.bottom()),
            ],
            egui::Stroke::new(1.5, visuals.selection.bg_fill),
        );
        painter.text(
            egui::pos2(playhead_x, layout.rect.top()),
            egui::Align2::CENTER_TOP,
            self.current_frame.to_string(),
            egui::FontId::proportional(11.0),
            visuals.selection.bg_fill,
        );
    }

    /// Draws each keyframe marker and handles dragging/removal. Returns the
    /// pending `(index, new_frame)` move and/or index to remove, applied by the
    /// caller to avoid mutating while iterating.
    fn draw_markers(
        &self,
        ui: &Ui,
        layout: &TimelineLayout,
    ) -> (Option<(usize, usize)>, Option<usize>) {
        let painter = ui.painter_at(layout.rect);
        let mut to_move = None;
        let mut to_remove = None;

        for (idx, kf) in self.config.keyframes.iter().enumerate() {
            let frame = kf.frame;
            let center = egui::pos2(layout.frame_to_x(frame), layout.marker_y);
            let hit = egui::Rect::from_center_size(center, egui::Vec2::splat(MARKER_RADIUS * 2.5));
            // Identity is keyed on the stable index, not the frame, so a drag
            // survives the frame value changing underneath it.
            let resp = ui.interact(
                hit,
                ui.id().with(("keyframe", idx)),
                egui::Sense::click_and_drag(),
            );

            let color = if resp.hovered() || resp.dragged() {
                ui.visuals().widgets.hovered.fg_stroke.color
            } else {
                ui.visuals().selection.bg_fill
            };
            painter.circle_filled(center, MARKER_RADIUS, color);
            painter.circle_stroke(
                center,
                MARKER_RADIUS,
                egui::Stroke::new(1.0, ui.visuals().extreme_bg_color),
            );
            painter.text(
                egui::pos2(center.x, layout.marker_y + MARKER_RADIUS + 2.0),
                egui::Align2::CENTER_TOP,
                frame.to_string(),
                egui::FontId::proportional(11.0),
                ui.visuals().text_color(),
            );

            if resp.dragged()
                && let Some(pointer) = resp.interact_pointer_pos()
            {
                let new_frame = layout.x_to_frame(pointer.x);
                let occupied = self
                    .config
                    .keyframes
                    .iter()
                    .enumerate()
                    .any(|(i, k)| i != idx && k.frame == new_frame);
                if new_frame != frame && !occupied {
                    to_move = Some((idx, new_frame));
                }
            }

            resp.clone().context_menu(|ui| {
                if ui.button("Remove keyframe").clicked() {
                    to_remove = Some(idx);
                    ui.close();
                }
            });

            resp.on_hover_text(format!(
                "Frame {frame} — drag to move, right-click to remove"
            ));
        }

        (to_move, to_remove)
    }
}

/// Geometry for mapping frames to/from x positions on the timeline.
struct TimelineLayout {
    rect: egui::Rect,
    track_left: f32,
    track_width: f32,
    marker_y: f32,
    span: f32,
}

impl TimelineLayout {
    const PAD: f32 = 8.0;
    const HEIGHT: f32 = 44.0;

    fn allocate(ui: &mut Ui, last_frame: usize) -> Self {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), Self::HEIGHT),
            egui::Sense::hover(),
        );
        let track_left = rect.left() + Self::PAD;
        Self {
            track_left,
            track_width: (rect.right() - Self::PAD - track_left).max(1.0),
            // Leave room for the current frame above and keyframe numbers below.
            marker_y: rect.top() + 20.0,
            span: last_frame.max(1) as f32,
            rect,
        }
    }

    fn track_right(&self) -> f32 {
        self.track_left + self.track_width
    }

    fn frame_to_x(&self, frame: usize) -> f32 {
        self.track_left + (frame as f32 / self.span) * self.track_width
    }

    fn x_to_frame(&self, x: f32) -> usize {
        (((x - self.track_left) / self.track_width) * self.span)
            .round()
            .clamp(0.0, self.span) as usize
    }
}

/// Background file-dialog driver for saving/loading the animation config. The
/// dialogs (via `rrfd`) are async, so they run on a worker thread and hand the
/// loaded config back through a shared slot, picked up on the next frame.
#[cfg(not(target_family = "wasm"))]
struct ConfigIo {
    actor: brush_async::Actor,
    loaded: Arc<Mutex<Option<AnimationConfig>>>,
}

#[cfg(not(target_family = "wasm"))]
impl Default for ConfigIo {
    fn default() -> Self {
        Self {
            actor: brush_async::Actor::new("animation-config-io"),
            loaded: Arc::new(Mutex::new(None)),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl ConfigIo {
    /// Opens a save dialog and writes `config` as JSON to the chosen path.
    fn save(&self, config: AnimationConfig) {
        self.actor
            .run(move || async move {
                let json = match config.to_json() {
                    Ok(json) => json,
                    Err(e) => {
                        log::error!("Failed to serialize animation config: {e}");
                        return;
                    }
                };
                if let Err(e) = rrfd::save_file("animation.json", json.into_bytes()).await {
                    log::error!("Failed to save animation config: {e}");
                }
            })
            .detach();
    }

    /// Opens a file picker and stashes the parsed config for the panel to pick up.
    fn load(&self, ctx: egui::Context) {
        let loaded = self.loaded.clone();
        self.actor
            .run(move || async move {
                match load_config_file().await {
                    Ok(config) => {
                        *loaded.lock().expect("poisoned") = Some(config);
                        ctx.request_repaint();
                    }
                    Err(e) => log::error!("Failed to load animation config: {e}"),
                }
            })
            .detach();
    }

    fn take_loaded(&self) -> Option<AnimationConfig> {
        self.loaded.lock().expect("poisoned").take()
    }
}

#[cfg(not(target_family = "wasm"))]
async fn load_config_file() -> anyhow::Result<AnimationConfig> {
    use tokio::io::AsyncReadExt;

    let picked = rrfd::pick_file().await?;
    let mut contents = String::new();
    let mut reader = picked.reader;
    reader.read_to_string(&mut contents).await?;
    Ok(AnimationConfig::from_json(&contents)?)
}
