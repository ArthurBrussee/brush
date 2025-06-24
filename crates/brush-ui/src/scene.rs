use brush_dataset::splat_export;
use brush_process::message::ProcessMessage;
use core::f32;
use egui::{Area, epaint::mutex::RwLock as EguiRwLock};
use std::sync::Arc;

use brush_render::{
    MainBackend,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect, Slider};
use glam::{UVec2, Vec3};
use tokio_with_wasm::alias as tokio_wasm;
use tracing::trace_span;
use web_time::Instant;

use crate::{
    BrushUiProcess, UiMode, app::CameraSettings, burn_texture::BurnTexture, draw_checkerboard,
    panels::AppPanel, size_for_splat_view,
};

#[derive(Debug, Clone, PartialEq)]
struct RenderState {
    size: UVec2,
    cam: Camera,
    frame: f32,
    splat_scale: Option<f32>,
}

struct ErrorDisplay {
    headline: String,
    context: Vec<String>,
}

pub struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    view_splats: Vec<Splats<MainBackend>>,
    frame_count: u32,
    frame: f32,

    // Ui state.
    live_update: bool,
    paused: bool,
    err: Option<ErrorDisplay>,
    ui_mode: UiMode,

    // Keep track of what was last rendered.
    last_state: Option<RenderState>,
}

impl ScenePanel {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: Arc<EguiRwLock<Renderer>>,
        ui_mode: UiMode,
    ) -> Self {
        Self {
            backbuffer: BurnTexture::new(renderer, device, queue),
            last_draw: None,
            err: None,
            view_splats: vec![],
            live_update: true,
            paused: false,
            last_state: None,
            ui_mode,
            frame_count: 0,
            frame: 0.0,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        process: &dyn BrushUiProcess,
        splats: Option<Splats<MainBackend>>,
    ) -> egui::Rect {
        let size = size_for_splat_view(ui, self.ui_mode == UiMode::Full);

        let mut size = size.floor();

        let view = process.selected_view();

        if let Some(view) = view {
            let aspect_ratio = view.image.aspect_ratio();
            if size.x / size.y > aspect_ratio {
                size.x = size.y * aspect_ratio;
            } else {
                size.y = size.x / aspect_ratio;
            }
        }
        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);

        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2::new(size.x as f32, size.y as f32),
            egui::Sense::drag(),
        );

        process.tick_controls(&response, ui);

        // Get camera after modifying the controls.
        let mut camera = process.current_camera();
        let settings = process.get_cam_settings();

        let focal_y = fov_to_focal(camera.fov_y, size.y) as f32;
        camera.fov_x = focal_to_fov(focal_y as f64, size.x);

        let state = RenderState {
            size,
            cam: camera.clone(),
            frame: self.frame,
            splat_scale: settings.splat_scale,
        };

        let dirty = self.last_state != Some(state.clone());

        if dirty {
            self.last_state = Some(state);
            // Check again next frame, as there might be more to animate.
            ui.ctx().request_repaint();
        }

        if let Some(splats) = splats {
            // If this viewport is re-rendering.
            if size.x > 8 && size.y > 8 && dirty {
                let _span = trace_span!("Render splats").entered();
                // Could add an option for background color.
                let (img, _) = splats.render(&camera, size, Vec3::ZERO, settings.splat_scale);
                self.backbuffer.update_texture(img);
            }
        }

        ui.scope(|ui| {
            let mut background = false;

            let view = process.selected_view();
            if let Some(view) = view {
                // if training views have alpha, show a background checker. Masked images
                // should still use a black background.
                if view.image.has_alpha() && !view.image.is_masked() {
                    background = true;
                    draw_checkerboard(ui, rect, Color32::WHITE);
                }
            }

            // If a scene is opaque, it assumes a black background.
            if !background {
                ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
            }

            if let Some(id) = self.backbuffer.id() {
                ui.painter().image(
                    id,
                    rect,
                    Rect {
                        min: egui::pos2(0.0, 0.0),
                        max: egui::pos2(1.0, 1.0),
                    },
                    Color32::WHITE,
                );
            }
        });

        rect
    }
}

impl AppPanel for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, process: &dyn BrushUiProcess) {
        match message {
            ProcessMessage::NewSource => {
                self.last_draw = None;
                self.view_splats = vec![];
                self.frame_count = 0;
                self.frame = 0.0;
                self.live_update = true;
                self.paused = false;
                self.err = None;
                self.backbuffer.reset();
                self.last_state = None;
            }
            ProcessMessage::ViewSplats {
                up_axis,
                splats,
                frame,
                total_frames,
            } => {
                // Training does also handle this but in the dataset.
                if !process.is_training() {
                    if let Some(up_axis) = up_axis {
                        process.set_model_up(*up_axis);
                    }
                }

                self.view_splats.truncate(*frame as usize);
                self.view_splats.push(*splats.clone());
                self.frame_count = *total_frames;

                // Mark redraw as dirty if we're live updating.
                if self.live_update {
                    self.last_state = None;
                }
            }
            ProcessMessage::TrainStep { splats, .. } => {
                let splats = *splats.clone();
                self.view_splats = vec![splats];
                // Mark redraw as dirty if we're live updating.
                if self.live_update {
                    self.last_state = None;
                }
            }
            _ => {}
        }
    }

    fn on_error(&mut self, error: &anyhow::Error, _: &dyn BrushUiProcess) {
        let headline = error.to_string();
        let context = error
            .chain()
            .skip(1)
            .map(|cause| format!("{cause}"))
            .collect();
        self.err = Some(ErrorDisplay { headline, context });
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &dyn BrushUiProcess) {
        let cur_time = Instant::now();

        let delta_time = self.last_draw.map_or(0.0, |x| x.elapsed().as_secs_f32());
        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show.
        if !process.is_training()
            && self.view_splats.is_empty()
            && self.err.is_none()
            && self.ui_mode == UiMode::Full
        {
            ui.heading("Load a ply file or dataset to get started.");
            ui.add_space(5.0);

            if cfg!(debug_assertions) {
                ui.scope(|ui| {
                    ui.visuals_mut().override_text_color = Some(Color32::LIGHT_BLUE);
                    ui.heading(
                        "Note: running in debug mode, compile with --release for best performance",
                    );
                });

                ui.add_space(10.0);
            }

            #[cfg(target_family = "wasm")]
            ui.scope(|ui| {
                ui.visuals_mut().override_text_color = Some(Color32::YELLOW);

                ui.label(
                    r#"
Note: In browser training can be slower. For bigger training runs consider using the native app."#,
                );
            });

            return;
        }

        if let Some(err) = self.err.as_ref() {
            ui.heading(format!("❌ {}", err.headline));

            ui.indent("err_context", |ui| {
                for c in &err.context {
                    ui.label(format!("• {c}"));
                    ui.add_space(2.0);
                }
            });
        } else {
            const FPS: f32 = 24.0;

            if !self.paused {
                self.frame += delta_time;
            }
            if self.view_splats.len() as u32 != self.frame_count {
                let max_t = (self.view_splats.len() - 1) as f32 / FPS;
                self.frame = self.frame.min(max_t);
            }
            let frame = (self.frame * FPS)
                .rem_euclid(self.frame_count as f32)
                .floor() as usize;

            let splats = self.view_splats.get(frame).cloned();
            let rect = self.draw_splats(ui, process, splats.clone());

            if process.is_loading() {
                let id = ui.auto_id_with("loading_bar");
                Area::new(id)
                    .order(egui::Order::Foreground)
                    .fixed_pos(rect.min)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::new()
                            .fill(egui::Color32::from_rgba_premultiplied(20, 20, 20, 150))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("Loading...").heading());
                                    ui.spinner();
                                });
                            });
                    });
            }

            if self.view_splats.len() > 1 && self.view_splats.len() as u32 == self.frame_count {
                let id = ui.auto_id_with("play_pause_button");
                Area::new(id)
                    .order(egui::Order::Foreground)
                    .fixed_pos(egui::pos2(rect.max.x - 40.0, rect.min.y + 6.0))
                    .show(ui.ctx(), |ui| {
                        let bg_color = if self.paused {
                            egui::Color32::from_rgba_premultiplied(0, 0, 0, 64)
                        } else {
                            egui::Color32::from_rgba_premultiplied(30, 80, 200, 120)
                        };

                        egui::Frame::new()
                            .fill(bg_color)
                            .corner_radius(egui::CornerRadius::same(16))
                            .inner_margin(egui::Margin::same(4))
                            .show(ui, |ui| {
                                let icon = if self.paused { "⏵" } else { "⏸" };
                                let mut button = egui::Button::new(
                                    egui::RichText::new(icon).size(18.0).color(Color32::WHITE),
                                );

                                if !self.paused {
                                    button = button.fill(egui::Color32::from_rgb(60, 120, 220));
                                }

                                if ui.add(button).clicked() {
                                    self.paused = !self.paused;
                                }
                            });
                    });
            }

            ui.horizontal(|ui| {
                if process.is_training() {
                    ui.add_space(15.0);

                    let label = if self.paused {
                        "⏸ paused"
                    } else {
                        "⏵ training"
                    };

                    if ui.selectable_label(!self.paused, label).clicked() {
                        self.paused = !self.paused;
                        process.set_train_paused(self.paused);
                    }

                    ui.add_space(15.0);

                    ui.scope(|ui| {
                        ui.style_mut().visuals.selection.bg_fill = Color32::DARK_RED;
                        if ui
                            .selectable_label(self.live_update, "🔴 Live update splats")
                            .clicked()
                        {
                            self.live_update = !self.live_update;
                        }
                    });

                    ui.add_space(15.0);

                    if let Some(splats) = splats {
                        if ui.button("⬆ Export").clicked() {
                            let fut = async move {
                                let data = splat_export::splat_to_ply(splats).await;

                                let data = match data {
                                    Ok(data) => data,
                                    Err(e) => {
                                        log::error!("Failed to serialize file: {e}");
                                        return;
                                    }
                                };

                                // Not sure where/how to show this error if any.
                                let _ = rrfd::save_file("export.ply", data)
                                    .await
                                    .inspect_err(|e| log::error!("Failed to save file: {e}"));
                            };

                            tokio_wasm::task::spawn(fut);
                        }
                    }
                }

                if self.ui_mode == UiMode::Full {
                    ui.add_space(15.0);

                    // Splat scale slider
                    let mut settings = process.get_cam_settings();
                    let mut scale = settings.splat_scale.unwrap_or(1.0);

                    ui.label("Splat Scale:");
                    let response = ui.add(
                        Slider::new(&mut scale, 0.01..=2.0)
                            .logarithmic(true)
                            .show_value(true)
                            .custom_formatter(|val, _| format!("{val:.1}x")),
                    );

                    if response.changed() {
                        settings.splat_scale = Some(scale);
                        process.set_cam_settings(settings);
                    }

                    ui.add_space(15.0);

                    // FOV slider
                    ui.label("Field of View:");
                    let current_camera = process.current_camera();
                    let mut fov_degrees = current_camera.fov_y.to_degrees() as f32;
                    let response = ui.add(
                        Slider::new(&mut fov_degrees, 10.0..=140.0)
                            .suffix("°")
                            .show_value(true)
                            .custom_formatter(|val, _| format!("{val:.0}°")),
                    );

                    if response.changed() {
                        process.set_cam_settings(CameraSettings {
                            fov_y: fov_degrees.to_radians() as f64,
                            ..process.get_cam_settings()
                        });
                    }

                    ui.selectable_label(false, "Controls")
                        .on_hover_ui_at_pointer(|ui| {
                            ui.heading("Controls");

                            ui.label("• Left click and drag to orbit");
                            ui.label(
                                "• Right click, or left click + spacebar, and drag to look around.",
                            );
                            ui.label("• Middle click, or left click + control, and drag to pan");
                            ui.label("• Scroll to zoom");
                            ui.label("• WASD to fly, Q&E to move up & down.");
                            ui.label("• Z&C to roll, X to reset roll");
                            ui.label("• Shift to move faster");
                        });
                }
            });
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
