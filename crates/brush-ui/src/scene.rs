#[cfg(feature = "training")]
use crate::settings_popup::SettingsPopup;
#[cfg(feature = "training")]
use brush_process::message::TrainMessage;
#[cfg(feature = "training")]
use std::sync::Mutex;

use brush_process::{create_process, message::ProcessMessage};
use brush_vfs::DataSource;
use core::f32;
use egui::{
    Align2, Button, Frame, RichText, containers::Popup, epaint::mutex::RwLock as EguiRwLock,
};
use std::sync::Arc;

use brush_render::{
    MainBackend,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
    render_splats,
};
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect, Slider};
use glam::{UVec2, Vec3};
use tracing::trace_span;
use web_time::Instant;

use serde::{Deserialize, Serialize};

/// Controls how often the viewport re-renders during training.
#[derive(Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderUpdateMode {
    /// Don't re-render during training
    Off,
    /// Re-render every 100 iterations
    Low,
    /// Re-render every 5 iterations (default)
    #[default]
    Live,
}

impl RenderUpdateMode {
    /// Returns the iteration interval for this mode, or None if rendering is disabled.
    #[cfg(feature = "training")]
    fn update_interval(&self) -> Option<u32> {
        match self {
            Self::Off => None,
            Self::Low => Some(100),
            Self::Live => Some(5),
        }
    }

    fn to_index(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Low => 1,
            Self::Live => 2,
        }
    }
}

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
    settings: CameraSettings,
    grid_opacity: f32,
    frame: u32,
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

#[derive(Default, Serialize, Deserialize)]
pub struct ScenePanel {
    #[serde(skip)]
    pub(crate) backbuffer: Option<BurnTexture>,
    #[serde(skip)]
    pub(crate) last_draw: Option<Instant>,
    #[serde(skip)]
    has_splats: bool,
    /// Current frame for animated sequences (as float for smooth interpolation).
    #[serde(skip)]
    frame: f32,
    /// Total number of frames in the current sequence.
    #[serde(skip)]
    frame_count: u32,
    /// Whether animation playback is paused.
    #[serde(skip)]
    paused: bool,
    #[serde(skip)]
    err: Option<ErrorDisplay>,
    #[serde(skip)]
    warnings: Vec<ErrorDisplay>,
    /// Number of warnings that have been seen by the user.
    #[serde(skip)]
    seen_warning_count: usize,
    #[serde(skip)]
    last_state: Option<RenderState>,
    #[serde(skip)]
    widget_3d: Option<Widget3D>,
    #[serde(skip)]
    source_name: Option<String>,
    #[serde(skip)]
    source_type: Option<DataSource>,
    // Loading UI state
    #[serde(skip)]
    url: String,
    #[serde(skip)]
    show_url_dialog: bool,
    /// Controls how often the viewport re-renders during training.
    #[serde(skip)]
    render_update_mode: RenderUpdateMode,
    /// Tracks the last iteration we rendered at, for Low mode.
    #[serde(skip)]
    last_rendered_iter: u32,
    #[cfg(feature = "training")]
    #[serde(skip)]
    settings_popup: Option<Arc<Mutex<SettingsPopup>>>,
}

impl ScenePanel {
    fn draw_load_buttons(&mut self, ui: &mut egui::Ui) -> Option<DataSource> {
        let button_height = 28.0;
        let button_color = Color32::from_rgb(70, 130, 180);
        let mut load_option = None;

        ui.horizontal(|ui| {
            if ui
                .add(
                    Button::new(RichText::new("📁 File").size(13.0))
                        .min_size(egui::vec2(80.0, button_height))
                        .fill(button_color)
                        .stroke(egui::Stroke::NONE),
                )
                .clicked()
            {
                load_option = Some(DataSource::PickFile);
            }

            let can_pick_dir = !cfg!(target_os = "android");
            if can_pick_dir
                && ui
                    .add(
                        Button::new(RichText::new("📂 Directory").size(13.0))
                            .min_size(egui::vec2(100.0, button_height))
                            .fill(button_color)
                            .stroke(egui::Stroke::NONE),
                    )
                    .clicked()
            {
                load_option = Some(DataSource::PickDirectory);
            }

            let can_url = !cfg!(target_os = "android");
            if can_url
                && ui
                    .add(
                        Button::new(RichText::new("🔗 URL").size(13.0))
                            .min_size(egui::vec2(70.0, button_height))
                            .fill(button_color)
                            .stroke(egui::Stroke::NONE),
                    )
                    .clicked()
            {
                self.show_url_dialog = true;
            }
        });

        load_option
    }

    fn draw_url_dialog(&mut self, ui: &egui::Ui) -> Option<DataSource> {
        let mut load_option = None;

        if self.show_url_dialog {
            egui::Window::new("Load from URL")
                .resizable(false)
                .collapsible(false)
                .default_pos(ui.ctx().screen_rect().center())
                .pivot(Align2::CENTER_CENTER)
                .show(ui.ctx(), |ui| {
                    ui.vertical(|ui| {
                        ui.label("Enter URL:");
                        ui.add_space(5.0);

                        let url_response = ui.add(
                            egui::TextEdit::singleline(&mut self.url)
                                .desired_width(300.0)
                                .hint_text("e.g., splat.com/example.ply"),
                        );

                        ui.add_space(10.0);

                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() && !self.url.trim().is_empty() {
                                load_option = Some(DataSource::Url(self.url.clone()));
                                self.show_url_dialog = false;
                            }
                            if ui.button("Cancel").clicked() {
                                self.show_url_dialog = false;
                            }
                        });

                        if url_response.lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter))
                            && !self.url.trim().is_empty()
                        {
                            load_option = Some(DataSource::Url(self.url.clone()));
                            self.show_url_dialog = false;
                        }
                    });
                });
        }

        load_option
    }

    fn start_loading(&self, source: DataSource, process: &UiProcess) {
        process.connect_to_process(create_process(
            source,
            #[cfg(feature = "training")]
            {
                let settings = self.settings_popup.clone().unwrap();
                async move |initial| {
                    let fut = settings.lock().unwrap().start_pick(initial);
                    fut.await
                }
            },
        ));
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
            settings: settings.clone(),
            grid_opacity,
            frame: self.frame as u32,
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
                let (img, _, _) = render_splats(
                    &splats,
                    &camera,
                    pixel_size,
                    settings.background.unwrap_or(Vec3::ZERO),
                    settings.splat_scale,
                );

                if let Some(backbuffer) = &mut self.backbuffer {
                    backbuffer.update_texture(img);
                }

                if let Some(widget_3d) = &mut self.widget_3d
                    && let Some(backbuffer) = &self.backbuffer
                    && let Some(texture) = backbuffer.texture()
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

            if let Some(backbuffer) = &self.backbuffer
                && let Some(id) = backbuffer.id()
            {
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

    fn draw_play_pause(&mut self, ui: &egui::Ui, rect: Rect) {
        // Only show play/pause if we have a multi-frame sequence that's fully loaded
        if self.frame_count > 1 {
            let id = ui.auto_id_with("play_pause_button");
            egui::Area::new(id)
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
                            let icon = if self.paused { "\u{25B6}" } else { "\u{23F8}" };
                            let mut button =
                                Button::new(RichText::new(icon).size(18.0).color(Color32::WHITE));

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

    fn draw_warnings_popup(&mut self, ui: &mut egui::Ui, popup_id: egui::Id) {
        ui.set_min_width(280.0);
        ui.set_max_width(400.0);
        ui.set_max_height(300.0);

        // Warning header
        ui.horizontal(|ui| {
            ui.label(RichText::new("⚠").size(16.0).color(Color32::YELLOW));
            ui.label(
                RichText::new(format!("Warnings ({})", self.warnings.len()))
                    .strong()
                    .color(Color32::YELLOW),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let clear_button = Button::new(
                    RichText::new("Clear")
                        .size(11.0)
                        .color(Color32::from_rgb(180, 180, 180)),
                )
                .fill(Color32::from_rgb(60, 60, 65))
                .corner_radius(4.0);

                if ui.add(clear_button).clicked() {
                    self.warnings.clear();
                    self.seen_warning_count = 0;
                    // Close the popup to prevent click from hitting other UI elements
                    Popup::close_id(ui.ctx(), popup_id);
                }
            });
        });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        if self.warnings.is_empty() {
            ui.label(
                RichText::new("No warnings")
                    .italics()
                    .color(Color32::from_rgb(140, 140, 140)),
            );
        } else {
            let has_new = self.warnings.len() > self.seen_warning_count;
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .max_height(250.0)
                .stick_to_bottom(has_new)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;

                    for (i, warning) in self.warnings.iter().enumerate() {
                        let is_new = i >= self.seen_warning_count;
                        let color = if is_new {
                            Color32::from_rgb(255, 200, 80)
                        } else {
                            Color32::from_rgb(200, 180, 120)
                        };

                        ui.horizontal(|ui| {
                            if is_new {
                                ui.label(RichText::new("•").color(Color32::YELLOW));
                            }
                            ui.label(RichText::new(&warning.headline).color(color));
                        });
                    }
                });
        }
    }
}

impl ScenePanel {
    fn reset(&mut self) {
        self.last_draw = None;
        self.last_state = None;
        self.has_splats = false;
        self.frame = 0.0;
        self.frame_count = 0;
        self.paused = false;
        self.last_rendered_iter = 0;
        self.warnings.clear();
        self.seen_warning_count = 0;
    }

    fn draw_controls_help(ui: &mut egui::Ui, min_width: Option<f32>) {
        let key_color = Color32::from_rgb(140, 180, 220);
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
            .inner_margin(egui::Margin::symmetric(24, 20))
            .show(ui, |ui| {
                if let Some(w) = min_width {
                    ui.set_min_width(w);
                }
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new("Controls")
                            .size(16.0)
                            .strong()
                            .color(title_color),
                    );
                    ui.add_space(12.0);

                    egui::Grid::new("controls_grid")
                        .num_columns(2)
                        .spacing([16.0, 8.0])
                        .show(ui, |ui| {
                            for (key, action) in controls {
                                ui.label(RichText::new(key).size(14.0).color(key_color));
                                ui.label(RichText::new(action).size(14.0).color(action_color));
                                ui.end_row();
                            }
                        });
                });
            });
    }

    fn draw_controls_content(ui: &mut egui::Ui, process: &UiProcess) {
        ui.spacing_mut().item_spacing.y = 6.0;

        // FOV slider
        ui.label(RichText::new("Field of View").size(12.0));
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
        ui.label(RichText::new("Splat Scale").size(12.0));
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

        ui.add_space(6.0);

        // Grid toggle
        let mut settings = process.get_cam_settings();
        let mut enabled = settings.grid_enabled.unwrap_or(false);
        if ui.checkbox(&mut enabled, "Show Grid").changed() {
            settings.grid_enabled = Some(enabled);
            process.set_cam_settings(&settings);
        }

        ui.label(RichText::new("Background").size(12.0));

        ui.separator();

        ui.horizontal(|ui| {
            let mut settings = process.get_cam_settings();
            let mut bg_color = settings.background.map_or(egui::Color32::BLACK, |b| {
                egui::Color32::from_rgb(
                    (b.x * 255.0) as u8,
                    (b.y * 255.0) as u8,
                    (b.z * 255.0) as u8,
                )
            });

            if egui::widgets::color_picker::color_picker_color32(
                ui,
                &mut bg_color,
                egui::color_picker::Alpha::Opaque,
            ) {
                settings.background = Some(glam::vec3(
                    bg_color.r() as f32 / 255.0,
                    bg_color.g() as f32 / 255.0,
                    bg_color.b() as f32 / 255.0,
                ));
                process.set_cam_settings(&settings);
            }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(6.0);

        if ui.button("Reset Layout").clicked() {
            process.request_reset_layout();
        }
    }
}

impl AppPane for ScenePanel {
    fn title(&self) -> egui::WidgetText {
        match (&self.source_type, &self.source_name) {
            (Some(t), Some(n)) => {
                let mut job = egui::text::LayoutJob::default();
                job.append(
                    n,
                    0.0,
                    egui::TextFormat {
                        color: Color32::WHITE,
                        ..Default::default()
                    },
                );
                job.append(
                    &format!("  |  {t}"),
                    0.0,
                    egui::TextFormat {
                        color: Color32::from_rgb(140, 140, 140),
                        ..Default::default()
                    },
                );
                job.into()
            }
            (None, Some(n)) => n.clone().into(),
            _ => "Scene".into(),
        }
    }

    fn top_bar_right_ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        // Only show reset button if we have content loaded
        let has_content = self.has_splats || process.is_training();

        if has_content {
            // New button - stands out with red background
            let new_button = Button::new(
                RichText::new("New")
                    .size(12.0)
                    .strong()
                    .color(Color32::WHITE),
            )
            .fill(egui::Color32::from_rgb(180, 70, 70))
            .corner_radius(6.0)
            .min_size(egui::vec2(50.0, 18.0));

            if ui
                .add(new_button)
                .on_hover_text("Start over with a new file")
                .clicked()
            {
                if process.is_training() {
                    // Store in egui memory that we want to show the confirm dialog
                    ui.ctx().memory_mut(|mem| {
                        mem.data
                            .insert_temp(egui::Id::new("show_reset_confirm"), true);
                    });
                } else {
                    process.reset_session();
                }
            }

            ui.add_space(6.0);
        }

        let help_button = Button::new(RichText::new("?").size(14.0).color(Color32::WHITE))
            .fill(egui::Color32::from_rgb(70, 130, 180))
            .corner_radius(6.0)
            .min_size(egui::vec2(22.0, 18.0));

        let help_response = ui.add(help_button).on_hover_text("Controls");

        Popup::from_toggle_button_response(&help_response)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                Self::draw_controls_help(ui, None);
            });

        ui.add_space(6.0);

        // Settings dropdown
        let gear_button = Button::new(RichText::new("⚙").size(14.0).color(Color32::WHITE))
            .fill(egui::Color32::from_rgb(70, 70, 75))
            .corner_radius(6.0)
            .min_size(egui::vec2(22.0, 18.0));

        let response = ui.add(gear_button).on_hover_text("Settings");

        Popup::from_toggle_button_response(&response)
            .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
            .show(|ui| {
                ui.set_min_width(220.0);
                Self::draw_controls_content(ui, process);
            });

        if !self.warnings.is_empty() {
            ui.add_space(6.0);

            let unseen_count = self.warnings.len().saturating_sub(self.seen_warning_count);
            let has_unseen = unseen_count > 0;

            let button_color = if has_unseen {
                egui::Color32::from_rgb(220, 160, 40) // Brighter yellow/orange for new warnings
            } else {
                egui::Color32::from_rgb(90, 85, 70) // Subtle for seen warnings
            };

            let label = format!("⚠ {}", self.warnings.len());

            let warnings_button = Button::new(RichText::new(label).size(12.0).strong().color(
                if has_unseen {
                    Color32::BLACK
                } else {
                    Color32::from_rgb(180, 170, 130)
                },
            ))
            .fill(button_color)
            .corner_radius(6.0)
            .min_size(egui::vec2(44.0, 18.0));

            let response = ui.add(warnings_button).on_hover_text(if has_unseen {
                format!("{unseen_count} new warning(s)")
            } else {
                "View warnings".into()
            });

            // Auto-open popup when there are new warnings
            let popup_id = response.id.with("popup");
            if has_unseen {
                Popup::open_id(ui.ctx(), popup_id);
            }

            let was_open = Popup::is_id_open(ui.ctx(), popup_id);

            let popup = Popup::from_toggle_button_response(&response)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);

            popup.show(|ui| {
                self.draw_warnings_popup(ui, popup_id);
            });

            // Check if popup just closed
            let is_open = Popup::is_id_open(ui.ctx(), popup_id);
            if was_open && !is_open {
                self.seen_warning_count = self.warnings.len();
            }
        }

        if process.is_training() {
            ui.add_space(6.0);

            // Render update mode slider
            let mut idx = self.render_update_mode.to_index() as f32;
            let old_idx = idx as usize;
            let is_live = self.render_update_mode == RenderUpdateMode::Live;

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                ui.label(
                    RichText::new("View update")
                        .size(10.0)
                        .color(Color32::from_rgb(140, 140, 140)),
                );

                // Style the slider with red color
                ui.style_mut().visuals.widgets.inactive.fg_stroke.color =
                    Color32::from_rgb(180, 60, 60);
                ui.style_mut().visuals.widgets.hovered.fg_stroke.color =
                    Color32::from_rgb(220, 80, 80);
                ui.style_mut().visuals.widgets.active.fg_stroke.color =
                    Color32::from_rgb(220, 60, 60);
                ui.style_mut().visuals.selection.bg_fill = Color32::from_rgb(180, 60, 60);

                ui.add(
                    Slider::new(&mut idx, 0.0..=2.0)
                        .step_by(1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );

                // Current mode label with recording icon when live
                let mode_label = match self.render_update_mode {
                    RenderUpdateMode::Off => "Off",
                    RenderUpdateMode::Low => "Low",
                    RenderUpdateMode::Live => "Live",
                };
                let label_color = if is_live {
                    Color32::from_rgb(220, 60, 60)
                } else {
                    Color32::from_rgb(140, 140, 140)
                };
                if is_live {
                    ui.label(RichText::new("⏺").size(10.0).color(label_color));
                }
                ui.label(RichText::new(mode_label).size(10.0).color(label_color));
            });

            let new_idx = idx.round() as usize;
            if new_idx != old_idx {
                let old_mode = self.render_update_mode;
                self.render_update_mode = match new_idx {
                    0 => RenderUpdateMode::Off,
                    1 => RenderUpdateMode::Low,
                    _ => RenderUpdateMode::Live,
                };
                // If enabling rendering from Off, force a redraw
                if old_mode == RenderUpdateMode::Off {
                    self.last_state = None;
                }
            }
        }
    }

    fn init(
        &mut self,
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: Arc<EguiRwLock<Renderer>>,
        _burn_device: burn_wgpu::WgpuDevice,
        _adapter_info: wgpu::AdapterInfo,
    ) {
        self.widget_3d = Some(Widget3D::new(device.clone(), queue.clone()));
        self.backbuffer = Some(BurnTexture::new(renderer, device, queue));

        // Create the settings popup now that we have the base_path
        #[cfg(feature = "training")]
        {
            self.settings_popup = Some(Arc::new(Mutex::new(SettingsPopup::new())));
        }
    }

    fn on_message(&mut self, message: &ProcessMessage, process: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                self.err = None;
                self.source_name = None;
                self.source_type = None;
                self.reset();
            }

            ProcessMessage::StartLoading {
                name,
                source,
                training,
                base_path,
            } => {
                // If training reset. Otherwise, keep existing state until new splats are loaded.
                if *training {
                    self.reset();
                }
                self.source_name = Some(name.clone());
                self.source_type = Some(source.clone());

                #[cfg(feature = "training")]
                {
                    self.settings_popup
                        .as_ref()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .base_path = base_path.clone();
                }
                let _ = base_path;
            }
            ProcessMessage::SplatsUpdated {
                up_axis,
                frame,
                total_frames,
            } => {
                self.has_splats = true;
                self.frame_count = *total_frames;

                // For non-training updates (e.g., loading), always redraw
                if !process.is_training() {
                    self.last_state = None;

                    // When training, datasets handle this.
                    if let Some(up_axis) = up_axis {
                        process.set_model_up(*up_axis);
                    }

                    // For single-frame or still loading, keep frame at current loaded frame
                    if *total_frames <= 1 || *frame < *total_frames - 1 {
                        self.frame = *frame as f32;
                    }
                }
            }
            #[cfg(feature = "training")]
            ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter, .. }) => {
                // Check if we should redraw based on render update mode
                if let Some(interval) = self.render_update_mode.update_interval() {
                    // Check if enough iterations have passed since last render
                    if *iter >= self.last_rendered_iter + interval || self.last_rendered_iter == 0 {
                        self.last_rendered_iter = *iter;
                        self.last_state = None;
                    }
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
        // Track the scene rect for centering popups
        let scene_rect = ui.available_rect_before_wrap();

        if let Some(err) = &self.err {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    Frame::new()
                        .fill(Color32::from_rgba_unmultiplied(60, 30, 30, 220))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(24, 20))
                        .show(ui, |ui| {
                            err.draw(ui);
                        });
                });
            });
        }

        let cur_time = Instant::now();
        let delta_time = self.last_draw.map_or(0.0, |t| t.elapsed().as_secs_f32());
        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show - show load buttons
        let show_welcome = !process.is_training()
            && !self.has_splats
            && process.ui_mode() != UiMode::EmbeddedViewer;

        if show_welcome {
            let box_color = Color32::from_rgba_unmultiplied(40, 40, 45, 200);
            let text_color = Color32::from_rgb(150, 150, 150);
            let box_width = 320.0;

            // Center content vertically and horizontally
            ui.vertical(|ui| {
                ui.add_space(ui.available_height() * 0.25);

                ui.horizontal(|ui| {
                    ui.add_space((ui.available_width() - box_width - 48.0).max(0.0) / 2.0);

                    ui.vertical(|ui| {
                        Frame::new()
                            .fill(box_color)
                            .corner_radius(8.0)
                            .inner_margin(egui::Margin::symmetric(24, 20))
                            .show(ui, |ui| {
                                ui.set_min_width(box_width);
                                ui.vertical(|ui| {
                                    ui.label(
                                        RichText::new(
                                            "Load a .ply splat file or a dataset to train",
                                        )
                                        .size(14.0)
                                        .color(text_color),
                                    );
                                    ui.add_space(16.0);

                                    // Load buttons
                                    let load_option = self.draw_load_buttons(ui);
                                    if let Some(source) = load_option {
                                        self.start_loading(source, process);
                                    }
                                });
                            });

                        ui.add_space(20.0);

                        // Controls help box - same width as getting started box
                        Self::draw_controls_help(ui, Some(box_width));

                        if cfg!(debug_assertions) {
                            ui.add_space(24.0);
                            ui.label(
                                RichText::new("Debug build - use --release for best performance")
                                    .size(14.0)
                                    .strong()
                                    .color(Color32::from_rgb(220, 160, 60)),
                            );
                        }
                    });
                });
            });

            // Draw URL dialog if open
            if let Some(source) = self.draw_url_dialog(ui) {
                self.start_loading(source, process);
            }
        } else {
            // Animate frame if we have a multi-frame sequence and not paused
            if self.frame_count > 1 && !self.paused {
                // Advance frame by deltatime (30 fps playback)
                self.frame += delta_time * 30.0;
                // Loop back to start
                if self.frame >= self.frame_count as f32 {
                    self.frame = 0.0;
                }
                // Keep animating
                ui.ctx().request_repaint();
            }

            // Get the splat for the current frame
            let splats = process.current_splats().and_then(|sv| {
                let frame_idx = self.frame as usize;
                sv.get(frame_idx)
            });

            let interactive =
                matches!(process.ui_mode(), UiMode::Default | UiMode::FullScreenSplat);
            let rect = self.draw_splats(ui, process, splats, interactive);

            if interactive {
                self.draw_play_pause(ui, rect);
            }
        }

        // Draw settings popup if loading (at end so it draws over everything)
        #[cfg(feature = "training")]
        {
            if let Some(popup) = &mut self.settings_popup
                && process.is_loading()
                && process.is_training()
            {
                let mut popup = popup.lock().unwrap();
                popup.ui(ui, scene_rect.center());
            }
        }

        // Reset confirmation dialog - check egui memory for the flag
        let show_reset_confirm = ui.ctx().memory(|mem| {
            mem.data
                .get_temp::<bool>(egui::Id::new("show_reset_confirm"))
                .unwrap_or(false)
        });

        if show_reset_confirm {
            egui::Window::new("Unsaved Training")
                .resizable(false)
                .collapsible(false)
                .default_pos(scene_rect.center())
                .pivot(Align2::CENTER_CENTER)
                .show(ui.ctx(), |ui| {
                    ui.vertical(|ui| {
                        ui.label("You have unsaved training progress.");
                        ui.label("Are you sure you want to close?");
                        ui.add_space(12.0);

                        ui.horizontal(|ui| {
                            if ui
                                .add(
                                    Button::new("Close Anyway")
                                        .fill(Color32::from_rgb(150, 60, 60)),
                                )
                                .clicked()
                            {
                                ui.ctx().memory_mut(|mem| {
                                    mem.data
                                        .insert_temp(egui::Id::new("show_reset_confirm"), false);
                                });
                                process.reset_session();
                            }

                            if ui.button("Cancel").clicked() {
                                ui.ctx().memory_mut(|mem| {
                                    mem.data
                                        .insert_temp(egui::Id::new("show_reset_confirm"), false);
                                });
                            }
                        });
                    });
                });
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
