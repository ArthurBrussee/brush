use brush_dataset::splat_export;
use brush_process::message::ProcessMessage;
use burn::prelude::Backend;
use core::f32;
use egui::{Area, Frame, epaint::mutex::RwLock as EguiRwLock};
use std::sync::Arc;

use brush_render::{
    MainBackend,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect, Slider, collapsing_header::CollapsingState};
use glam::UVec2;
use tokio_with_wasm::alias as tokio_wasm;
use tracing::trace_span;
use web_time::Instant;

use crate::{
    UiMode, app::CameraSettings, burn_texture::BurnTexture, draw_checkerboard, panels::AppPane,
    ui_process::UiProcess,
};

#[derive(Clone, PartialEq)]
struct RenderState {
    size: UVec2,
    cam: Camera,
    frame: f32,
    settings: CameraSettings,
}

struct ErrorDisplay {
    headline: String,
    context: Vec<String>,
}

pub struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    view_splats: Vec<Splats<MainBackend>>,

    fully_loaded: bool,
    frame_count: u32,
    frame: f32,

    // Ui state.
    live_update: bool,
    paused: bool,
    err: Option<ErrorDisplay>,

    // Keep track of what was last rendered.
    last_state: Option<RenderState>,
}

impl ScenePanel {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        Self {
            backbuffer: BurnTexture::new(renderer, device, queue),
            last_draw: None,
            err: None,
            view_splats: vec![],
            live_update: true,
            paused: false,
            last_state: None,
            frame_count: 0,
            frame: 0.0,
            fully_loaded: false,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        process: &UiProcess,
        splats: Option<Splats<MainBackend>>,
        interactive: bool,
    ) -> egui::Rect {
        let mut size = ui.available_size();
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

        if interactive {
            process.tick_controls(&response, ui);
        }

        // Get camera after modifying the controls.
        let mut camera = process.current_camera();

        let total_transform = process.model_local_to_world() * camera.local_to_world();
        let (_, rotation, position) = total_transform.to_scale_rotation_translation();
        camera.position = position;
        camera.rotation = rotation;

        let settings = process.get_cam_settings();

        let focal_y = fov_to_focal(camera.fov_y, size.y) as f32;
        camera.fov_x = focal_to_fov(focal_y as f64, size.x);

        let state = RenderState {
            size,
            cam: camera.clone(),
            frame: self.frame,
            settings: settings.clone(),
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
                let (img, _) =
                    splats.render(&camera, size, settings.background, settings.splat_scale);
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

    fn controls_box(
        &mut self,
        ui: &egui::Ui,
        process: &UiProcess,
        splats: Option<Splats<MainBackend>>,
        rect: egui::Rect,
    ) {
        // Controls window in bottom right
        let id = ui.id().with("controls_box");
        egui::Area::new(id)
            .kind(egui::UiKind::Window)
            .current_pos(egui::pos2(rect.min.x, rect.min.y))
            .movable(false)
            .show(ui.ctx(), |ui| {
                // Add transparent background frame. This has the same settings as a
                let style = ui.style_mut();
                let fill = style.visuals.window_fill;
                style.visuals.window_fill =
                    Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), 200);
                let frame = Frame::window(style);

                frame.show(ui, |ui| {
                    if process.is_loading() {
                        ui.horizontal(|ui| {
                            ui.label("Loading...");
                            ui.spinner();
                        });
                        return;
                    }

                    // Custom title bar using egui's CollapsingState
                    let state = CollapsingState::load_with_default_open(
                        ui.ctx(),
                        ui.id().with("controls_collapse"),
                        false,
                    );

                    // Show a header
                    state
                        .show_header(ui, |ui| {
                            ui.label(egui::RichText::new("Controls").strong());

                            ui.add_space(5.0);

                            // Help button
                            let help_button = egui::Button::new(
                                egui::RichText::new("?").size(10.0).color(Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(60, 120, 200))
                            .corner_radius(6.0)
                            .min_size(egui::vec2(14.0, 14.0));

                            ui.add(help_button).on_hover_ui_at_pointer(|ui| {
                                ui.set_max_width(280.0);
                                ui.heading("Controls");
                                ui.separator();
                                ui.label("• Left click and drag to orbit");
                                ui.label("• Right click + drag to look around");
                                ui.label("• Middle click + drag to pan");
                                ui.label("• Scroll to zoom");
                                ui.label("• WASD to fly, Q&E up/down");
                                ui.label("• Z&C to roll, X to reset roll");
                                ui.label("• Shift to move faster");
                            });
                        })
                        .body_unindented(|ui| {
                            ui.set_max_width(180.0);
                            ui.spacing_mut().item_spacing.y = 6.0;

                            // Training controls
                            if process.is_training() {
                                let label = if self.paused {
                                    "⏸ Paused"
                                } else {
                                    "⏵ Training"
                                };

                                if ui.selectable_label(!self.paused, label).clicked() {
                                    self.paused = !self.paused;
                                    process.set_train_paused(self.paused);
                                }

                                ui.scope(|ui| {
                                    ui.style_mut().visuals.selection.bg_fill =
                                        Color32::from_rgb(120, 40, 40);
                                    if ui
                                        .selectable_label(self.live_update, "🔴 Live update")
                                        .clicked()
                                    {
                                        self.live_update = !self.live_update;
                                    }
                                });

                                if let Some(splats) = splats
                                    && ui.small_button("⬆ Export").clicked()
                                {
                                    export_current_splat(splats);
                                }

                                ui.add_space(4.0);
                                ui.separator();
                                ui.add_space(4.0);
                            }

                            // Background color picker
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Background").size(12.0));
                                let mut settings = process.get_cam_settings();
                                let mut bg_color = egui::Color32::from_rgb(
                                    (settings.background.x * 255.0) as u8,
                                    (settings.background.y * 255.0) as u8,
                                    (settings.background.z * 255.0) as u8,
                                );
                                if ui.color_edit_button_srgba(&mut bg_color).changed() {
                                    settings.background = glam::vec3(
                                        bg_color.r() as f32 / 255.0,
                                        bg_color.g() as f32 / 255.0,
                                        bg_color.b() as f32 / 255.0,
                                    );
                                    process.set_cam_settings(&settings);
                                }
                            });

                            ui.add_space(4.0);

                            // FOV slider
                            ui.label(egui::RichText::new("Field of View").size(12.0));
                            let current_camera = process.current_camera();
                            let mut fov_degrees = current_camera.fov_y.to_degrees() as f32;

                            let response = ui.add(
                                Slider::new(&mut fov_degrees, 10.0..=140.0)
                                    .suffix("°")
                                    .show_value(true)
                                    .custom_formatter(|val, _| format!("{val:.0}°")),
                            );

                            if response.changed() {
                                process.set_cam_fov(fov_degrees.to_radians() as f64);
                            }

                            // Splat scale slider
                            ui.label(egui::RichText::new("Splat Scale").size(12.0));
                            let mut settings = process.get_cam_settings();
                            let mut scale = settings.splat_scale.unwrap_or(1.0);

                            let response = ui.add(
                                Slider::new(&mut scale, 0.01..=2.0)
                                    .logarithmic(true)
                                    .show_value(true)
                                    .custom_formatter(|val, _| format!("{val:.1}x")),
                            );

                            if response.changed() {
                                settings.splat_scale = Some(scale);
                                process.set_cam_settings(&settings);
                            }

                            ui.add_space(4.0);
                        });
                });
            });
    }
}

fn export_current_splat<B: Backend>(splat: Splats<B>) {
    tokio_wasm::task::spawn(async move {
        let data = splat_export::splat_to_ply(splat).await;

        let data = match data {
            Ok(data) => data,
            Err(e) => {
                log::error!("Failed to serialize file: {e}");
                return;
            }
        };

        let _ = rrfd::save_file("export.ply", data).await.inspect_err(|e| {
            log::error!("Failed to save file: {e}");
        });
    });
}

impl ScenePanel {
    fn reset_splats(&mut self) {
        self.last_draw = None;
        self.last_state = None;
        self.view_splats = vec![];
        self.frame_count = 0;
        self.frame = 0.0;
    }
}

impl AppPane for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, process: &UiProcess) {
        match message {
            ProcessMessage::NewSource => {
                self.live_update = true;
                self.err = None;
            }
            ProcessMessage::StartLoading { training } => {
                // If training reset. Otherwise, keep existing splats until new ones are fully loaded.
                if *training {
                    self.reset_splats();
                }
            }
            ProcessMessage::ViewSplats {
                up_axis,
                splats,
                frame,
                total_frames,
                progress,
            } => {
                if !process.is_training()
                    && let Some(up_axis) = up_axis
                {
                    process.set_model_up(*up_axis);
                }

                self.frame_count = *total_frames;
                let done_loading = *progress >= 1.0;

                // For animated splats (total_frames > 1), always show streaming
                if *total_frames > 1 {
                    // Clear existing splats for animations to show streaming
                    if *frame == 0 {
                        self.view_splats.clear();
                    }
                    self.view_splats
                        .resize(*frame as usize + 1, splats.as_ref().clone());
                } else {
                    // Static splat - only replace when fully loaded (progress = 1.0) or if we haven't fully loaded a splat
                    // yet.
                    if done_loading || !self.fully_loaded {
                        self.view_splats = vec![splats.as_ref().clone()];
                    }
                }

                if done_loading {
                    self.fully_loaded = true;
                }

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

    fn on_error(&mut self, error: &anyhow::Error, _: &UiProcess) {
        let headline = error.to_string();
        let context = error
            .chain()
            .skip(1)
            .map(|cause| format!("{cause}"))
            .collect();
        self.err = Some(ErrorDisplay { headline, context });
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        let cur_time = Instant::now();

        let delta_time = self.last_draw.map_or(0.0, |x| x.elapsed().as_secs_f32());
        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show.
        if !process.is_training()
            && self.view_splats.is_empty()
            && self.err.is_none()
            && process.ui_mode() == UiMode::Default
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

            let interactive =
                matches!(process.ui_mode(), UiMode::Default | UiMode::FullScreenSplat);
            let rect = self.draw_splats(ui, process, splats.clone(), interactive);

            // Floating play/pause button if needed.
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

                        Frame::new()
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

            if interactive {
                self.controls_box(ui, process, splats, rect);
            }
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
