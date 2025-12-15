#[cfg(feature = "training")]
use brush_process::message::TrainMessage;

use brush_process::message::ProcessMessage;
use core::f32;
use egui::{Align2, Area, Frame, Pos2, Ui, epaint::mutex::RwLock as EguiRwLock};
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use brush_render::{
    MainBackend,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect, Slider, collapsing_header::CollapsingState};
use glam::{UVec2, Vec3};
use tracing::trace_span;
use web_time::Instant;

use crate::{
    UiMode,
    app::CameraSettings,
    burn_texture::BurnTexture,
    draw_checkerboard,
    panels::AppPane,
    ui_process::{BackgroundStyle, UiProcess},
    widget_3d::Widget3D,
};

#[derive(Clone, PartialEq)]
struct RenderState {
    size: UVec2,
    cam: Camera,
    frame: f32,
    settings: CameraSettings,
    grid_opacity: f32,
}

struct ErrorDisplay {
    headline: String,
    context: Vec<String>,
}

impl ErrorDisplay {
    fn new(error: &anyhow::Error) -> Self {
        let headline = error.to_string();
        let context = error
            .chain()
            .skip(1)
            .map(|cause| format!("{cause}"))
            .collect();
        Self { headline, context }
    }

    fn draw(&self, ui: &mut egui::Ui) {
        ui.heading(format!("❌ {}", self.headline));
        ui.indent("err_context", |ui| {
            for c in &self.context {
                ui.label(format!("• {c}"));
                ui.add_space(2.0);
            }
        });
    }
}

#[cfg(feature = "training")]
async fn export(splat: Splats<MainBackend>) -> Result<(), anyhow::Error> {
    let data = brush_serde::splat_to_ply(splat).await?;
    rrfd::save_file("export.ply", data).await?;
    Ok(())
}

fn box_ui<R>(
    id: &str,
    ui: &egui::Ui,
    pivot: Align2,
    pos: Pos2,
    add_contents: impl FnOnce(&mut Ui) -> R,
) {
    // Controls window in bottom right
    let id = ui.id().with(id);
    egui::Area::new(id)
        .kind(egui::UiKind::Window)
        .pivot(pivot)
        .current_pos(pos)
        .movable(false)
        .show(ui.ctx(), |ui| {
            let style = ui.style_mut();
            let fill = style.visuals.window_fill;
            style.visuals.window_fill =
                Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), 200);
            let frame = Frame::window(style);

            frame.show(ui, add_contents);
        });
}

pub struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    view_splats: Vec<Splats<MainBackend>>,

    fully_loaded: bool,
    frame_count: u32,
    frame: f32,

    // Splat info for display
    num_splats: u32,
    sh_degree: u32,

    // Ui state.
    live_update: bool,
    paused: bool,
    err: Option<ErrorDisplay>,
    warnings: Vec<ErrorDisplay>,

    export_channel: (
        UnboundedSender<anyhow::Error>,
        UnboundedReceiver<anyhow::Error>,
    ),

    // Keep track of what was last rendered.
    last_state: Option<RenderState>,

    // 3D widgets for visualization
    widget_3d: Option<Widget3D>,
}

impl ScenePanel {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        let channel = tokio::sync::mpsc::unbounded_channel();

        // Create Widget3D for 3D overlay rendering
        let widget_3d = Some(Widget3D::new(device.clone(), queue.clone()));

        Self {
            backbuffer: BurnTexture::new(renderer, device, queue),
            last_draw: None,
            err: None,
            warnings: vec![],
            view_splats: vec![],
            live_update: true,
            paused: false,
            last_state: None,
            frame_count: 0,
            frame: 0.0,
            fully_loaded: false,
            num_splats: 0,
            sh_degree: 0,
            export_channel: channel,
            widget_3d,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        process: &UiProcess,
        splats: Option<Splats<MainBackend>>,
        interactive: bool,
    ) -> egui::Rect {
        let size = ui.available_size();
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

        let view_eff = (camera.world_to_local() * process.model_local_to_world()).inverse();
        let (_, rotation, position) = view_eff.to_scale_rotation_translation();
        camera.position = position;
        camera.rotation = rotation;

        let settings = process.get_cam_settings();

        // Adjust FOV so that the scene view shows at least what's visible in the dataset view.
        // The camera has original fov_x and fov_y from the dataset. We need to ensure
        // the viewport shows at least that much in both dimensions.
        let camera_aspect = (camera.fov_x / 2.0).tan() / (camera.fov_y / 2.0).tan();
        let viewport_aspect = size.x as f64 / size.y as f64;

        if viewport_aspect > camera_aspect {
            // Viewport is wider than camera - keep fov_y, expand fov_x
            let focal_y = fov_to_focal(camera.fov_y, size.y);
            camera.fov_x = focal_to_fov(focal_y, size.x);
        } else {
            // Viewport is taller than camera - keep fov_x, expand fov_y
            let focal_x = fov_to_focal(camera.fov_x, size.x);
            camera.fov_y = focal_to_fov(focal_x, size.y);
        }

        let grid_opacity = process.get_grid_opacity();

        let state = RenderState {
            size,
            cam: camera.clone(),
            frame: self.frame,
            settings: settings.clone(),
            grid_opacity,
        };

        let dirty = self.last_state != Some(state.clone());

        if dirty {
            self.last_state = Some(state);
            // Check again next frame, as there might be more to animate.
            ui.ctx().request_repaint();
        }

        if let Some(splats) = splats {
            let pixel_size = glam::uvec2(
                (size.x as f32 * ui.ctx().pixels_per_point().round()) as u32,
                (size.y as f32 * ui.ctx().pixels_per_point().round()) as u32,
            );
            // If this viewport is re-rendering.
            if pixel_size.x > 8 && pixel_size.y > 8 && dirty {
                let _span = trace_span!("Render splats").entered();
                // Could add an option for background color.
                let (img, _) = splats.render(
                    &camera,
                    pixel_size,
                    settings.background.unwrap_or(Vec3::ZERO),
                    settings.splat_scale,
                );

                self.backbuffer.update_texture(img);

                if let Some(widget_3d) = &mut self.widget_3d
                    && let Some(texture) = self.backbuffer.texture()
                {
                    widget_3d.render_to_texture(
                        &camera,
                        process.model_local_to_world(),
                        pixel_size,
                        texture,
                        grid_opacity,
                    );
                }
            }
        }

        ui.scope(|ui| {
            // if training views have alpha, show a background checker. Masked images
            // should still use a black background.
            match process.background_style() {
                BackgroundStyle::Checkerboard => {
                    draw_checkerboard(ui, rect, Color32::WHITE);
                }
                BackgroundStyle::Black => {
                    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                }
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
        _splats: Option<Splats<MainBackend>>,
        pos: egui::Pos2,
    ) {
        let inner = |ui: &mut egui::Ui| {
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
                        ui.set_max_width(200.0);
                        Self::draw_controls_help(ui);
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

                        #[cfg(feature = "training")]
                        if let Some(splats) = _splats
                            && ui.small_button("⬆ Export").clicked()
                        {
                            let sender = self.export_channel.0.clone();
                            let ctx = ui.ctx().clone();
                            tokio_with_wasm::alias::task::spawn(async move {
                                if let Err(e) = export(splats).await {
                                    let _ = sender.send(e.context("Failed to export splat"));
                                    ctx.request_repaint();
                                }
                            });
                        }

                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                    }

                    // Background color picker
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Background").size(12.0));
                        let mut settings = process.get_cam_settings();
                        let mut bg_color = settings.background.map_or(egui::Color32::BLACK, |b| {
                            egui::Color32::from_rgb(
                                (b.x * 255.0) as u8,
                                (b.y * 255.0) as u8,
                                (b.z * 255.0) as u8,
                            )
                        });

                        if ui.color_edit_button_srgba(&mut bg_color).changed() {
                            settings.background = Some(glam::vec3(
                                bg_color.r() as f32 / 255.0,
                                bg_color.g() as f32 / 255.0,
                                bg_color.b() as f32 / 255.0,
                            ));
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

                    // Grid toggle
                    ui.horizontal(|ui| {
                        let mut enabled = process.get_cam_settings().grid_enabled.unwrap_or(false);
                        if ui.checkbox(&mut enabled, "Show Grid").changed() {
                            settings.grid_enabled = Some(enabled);
                            process.set_cam_settings(&settings);
                        }
                    });

                    if self.num_splats > 0 {
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);

                        let label_color = Color32::from_rgb(140, 140, 140);
                        let value_color = Color32::from_rgb(200, 200, 200);

                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Splats").size(11.0).color(label_color));
                            ui.label(
                                egui::RichText::new(format!("{}", self.num_splats))
                                    .size(11.0)
                                    .color(value_color),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("SH Degree")
                                    .size(11.0)
                                    .color(label_color),
                            );
                            ui.label(
                                egui::RichText::new(format!("{}", self.sh_degree))
                                    .size(11.0)
                                    .color(value_color),
                            );
                        });
                    }

                    ui.add_space(4.0);
                });
        };

        box_ui("controls_box", ui, Align2::LEFT_TOP, pos, inner);
    }

    fn draw_play_pause(&mut self, ui: &egui::Ui, rect: Rect) {
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
    }

    fn draw_warnings(&mut self, ui: &egui::Ui, pos: Pos2) {
        if self.warnings.is_empty() {
            return;
        }

        let inner = |ui: &mut egui::Ui| {
            ui.set_max_width(300.0);
            ui.set_max_height(200.0);

            // Warning header with icon
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("⚠").size(16.0).color(Color32::YELLOW));
                ui.label(
                    egui::RichText::new("Warnings")
                        .strong()
                        .color(Color32::YELLOW),
                );

                ui.add_space(10.0);

                if ui.button("clear").clicked() {
                    self.warnings.clear();
                }
            });

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);

            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;

                    for warning in &self.warnings {
                        ui.scope(|ui| {
                            ui.visuals_mut().override_text_color =
                                Some(Color32::from_rgb(255, 220, 120));
                            warning.draw(ui);
                        });
                        ui.add_space(4.0);
                    }
                });
        };

        box_ui("warnings_box", ui, Align2::RIGHT_TOP, pos, inner);
    }
}

impl ScenePanel {
    fn reset_splats(&mut self) {
        self.last_draw = None;
        self.last_state = None;
        self.view_splats = vec![];
        self.frame_count = 0;
        self.frame = 0.0;
        self.num_splats = 0;
        self.sh_degree = 0;
    }

    fn draw_controls_help(ui: &mut egui::Ui) {
        let key_color = Color32::from_rgb(130, 170, 220);
        let action_color = Color32::from_rgb(140, 140, 140);
        let title_color = Color32::from_rgb(200, 200, 200);

        let controls = [
            ("Left drag", "Orbit"),
            ("Right drag", "Look around"),
            ("Middle drag", "Pan"),
            ("Scroll", "Zoom"),
            ("WASD / QE", "Fly"),
            ("Shift", "Move faster"),
            ("F", "Fullscreen"),
        ];

        Frame::new()
            .fill(Color32::from_rgba_unmultiplied(40, 40, 45, 200))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::symmetric(18, 14))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Controls")
                            .size(14.0)
                            .strong()
                            .color(title_color),
                    );
                    ui.add_space(10.0);

                    egui::Grid::new("controls_grid")
                        .num_columns(2)
                        .spacing([16.0, 6.0])
                        .show(ui, |ui| {
                            for (key, action) in controls {
                                ui.label(egui::RichText::new(key).size(13.0).color(key_color));
                                ui.label(
                                    egui::RichText::new(action).size(13.0).color(action_color),
                                );
                                ui.end_row();
                            }
                        });
                });
            });
    }
}

impl AppPane for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, process: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
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

                // Track splat info
                self.num_splats = splats.num_splats();
                self.sh_degree = splats.sh_degree();

                // Mark redraw as dirty if we're live updating.
                if self.live_update {
                    self.last_state = None;
                }
            }
            #[cfg(feature = "training")]
            ProcessMessage::TrainMessage(TrainMessage::TrainStep { splats, .. }) => {
                let splats = *splats.clone();
                self.num_splats = splats.num_splats();
                self.sh_degree = splats.sh_degree();
                self.view_splats = vec![splats];
                // Mark redraw as dirty if we're live updating.
                if self.live_update {
                    self.last_state = None;
                }
            }
            ProcessMessage::Warning { error } => {
                self.warnings.push(ErrorDisplay::new(error));
            }
            _ => {}
        }
    }

    fn on_error(&mut self, error: &anyhow::Error, _: &UiProcess) {
        self.err = Some(ErrorDisplay::new(error));
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        if let Some(err) = self.err.as_ref() {
            err.draw(ui);
            return;
        }

        // Handle export errors
        while let Ok(err) = self.export_channel.1.try_recv() {
            self.warnings.push(ErrorDisplay::new(&err));
        }

        let cur_time = Instant::now();

        let delta_time = self.last_draw.map_or(0.0, |x| x.elapsed().as_secs_f32());
        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show.
        if !process.is_training()
            && self.view_splats.is_empty()
            && process.ui_mode() == UiMode::Default
        {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.30);

                ui.horizontal(|ui| {
                    let box_color = Color32::from_rgba_unmultiplied(40, 40, 45, 200);
                    let title_color = Color32::from_rgb(200, 200, 200);
                    let text_color = Color32::from_rgb(150, 150, 150);

                    // Center the two boxes
                    ui.add_space((ui.available_width() - 500.0).max(0.0) / 2.0);

                    // Getting started box
                    Frame::new()
                        .fill(box_color)
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(18, 14))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(
                                    egui::RichText::new("Getting Started")
                                        .size(14.0)
                                        .strong()
                                        .color(title_color),
                                );
                                ui.add_space(10.0);
                                ui.label(
                                    egui::RichText::new("Load a .ply splat file")
                                        .size(13.0)
                                        .color(text_color),
                                );
                                ui.label(
                                    egui::RichText::new("or a dataset to train")
                                        .size(13.0)
                                        .color(text_color),
                                );
                                ui.add_space(8.0);
                                ui.label(
                                    egui::RichText::new("Use the status bar above")
                                        .size(12.0)
                                        .color(Color32::from_rgb(120, 120, 120)),
                                );
                            });
                        });

                    ui.add_space(20.0);

                    // Controls box
                    Self::draw_controls_help(ui);
                });

                if cfg!(debug_assertions) {
                    ui.add_space(24.0);
                    ui.label(
                        egui::RichText::new("⚠ Debug build - use --release for best performance")
                            .size(11.0)
                            .color(Color32::from_rgb(180, 140, 60)),
                    );
                }
            });
            return;
        }

        const FPS: f32 = 24.0;

        if !self.paused {
            self.frame += delta_time;

            if self.view_splats.len() as u32 != self.frame_count {
                let max_t = (self.view_splats.len() - 1) as f32 / FPS;
                self.frame = self.frame.min(max_t);
            }
        }

        let frame = (self.frame * FPS)
            .rem_euclid(self.frame_count as f32)
            .floor() as usize;

        let splats = self.view_splats.get(frame).cloned();
        let interactive = matches!(process.ui_mode(), UiMode::Default | UiMode::FullScreenSplat);
        let rect = self.draw_splats(ui, process, splats.clone(), interactive);

        if interactive {
            // Floating play/pause button if needed.
            self.draw_play_pause(ui, rect);
            self.controls_box(ui, process, splats, egui::pos2(rect.min.x, rect.min.y));

            let pos = egui::pos2(ui.available_rect_before_wrap().max.x, rect.min.y);
            self.draw_warnings(ui, pos);
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
