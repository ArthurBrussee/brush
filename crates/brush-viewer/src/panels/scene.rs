use brush_dataset::splat_export;
use brush_ui::burn_texture::BurnTexture;
use burn_wgpu::Wgpu;
use egui::epaint::mutex::RwLock as EguiRwLock;
use std::{sync::Arc, time::Duration};

use brush_render::gaussian_splats::Splats;
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect};
use glam::Vec2;
use tokio_with_wasm::alias as tokio;
use tracing::trace_span;
use web_time::Instant;

use crate::{
    train_loop::TrainMessage,
    viewer::{ViewerContext, ViewerMessage},
    ViewerPanel,
};

pub(crate) struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    view_splats: Vec<Splats<Wgpu>>,
    frame: f32,
    err: Option<Arc<anyhow::Error>>,

    is_loading: bool,

    is_training: bool,
    live_update: bool,
    paused: bool,

    last_size: glam::UVec2,
    dirty: bool,

    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    renderer: Arc<EguiRwLock<Renderer>>,
}

impl ScenePanel {
    pub(crate) fn new(
        queue: Arc<wgpu::Queue>,
        device: Arc<wgpu::Device>,
        renderer: Arc<EguiRwLock<Renderer>>,
    ) -> Self {
        Self {
            frame: 0.0,
            backbuffer: BurnTexture::new(device.clone(), queue.clone()),
            last_draw: None,
            err: None,
            view_splats: vec![],
            live_update: true,
            paused: false,
            dirty: true,
            last_size: glam::UVec2::ZERO,
            is_loading: false,
            is_training: false,
            queue,
            device,
            renderer,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut ViewerContext,
        splats: &Splats<Wgpu>,
        delta_time: web_time::Duration,
    ) {
        let mut size = ui.available_size();
        let focal = context.camera.focal(glam::uvec2(1, 1));
        let aspect_ratio = focal.y / focal.x;
        if size.x / size.y > aspect_ratio {
            size.x = size.y * aspect_ratio;
        } else {
            size.y = size.x / aspect_ratio;
        }
        // Round to 64 pixels. Necesarry for buffer sizes to align.
        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);

        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2::new(size.x as f32, size.y as f32),
            egui::Sense::drag(),
        );

        let mouse_delta = glam::vec2(response.drag_delta().x, response.drag_delta().y);

        let (pan, rotate) = if response.dragged_by(egui::PointerButton::Primary) {
            (Vec2::ZERO, mouse_delta)
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            (mouse_delta, Vec2::ZERO)
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };

        let scrolled = ui.input(|r| r.smooth_scroll_delta).y;

        self.dirty |= context.controls.pan_orbit_camera(
            &mut context.camera,
            pan * 5.0,
            rotate * 5.0,
            scrolled * 0.01,
            glam::vec2(rect.size().x, rect.size().y),
            delta_time.as_secs_f32(),
        );

        self.dirty |= self.last_size != size;
        context.controls.dirty = false;

        // If this viewport is re-rendering.
        if ui.ctx().has_requested_repaint() && size.x > 0 && size.y > 0 && self.dirty {
            let _span = trace_span!("Render splats").entered();
            let (img, _) = splats.render(&context.camera, size, true);
            self.backbuffer.update_texture(img, self.renderer.clone());
            self.dirty = false;
            self.last_size = size;
        }

        if let Some(id) = self.backbuffer.id() {
            ui.scope(|ui| {
                if context
                    .dataset
                    .train
                    .views
                    .first()
                    .map(|view| view.image.color().has_alpha())
                    .unwrap_or(false)
                {
                    // if training views have alpha, show a background checker.
                    brush_ui::draw_checkerboard(ui, rect);
                } else {
                    // If a scene is opaque, it assumes a black background.
                    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                };

                ui.painter().image(
                    id,
                    rect,
                    Rect {
                        min: egui::pos2(0.0, 0.0),
                        max: egui::pos2(1.0, 1.0),
                    },
                    Color32::WHITE,
                );
            });
        }
    }
}

impl ViewerPanel for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: &ViewerMessage, _: &mut ViewerContext) {
        if self.live_update {
            self.dirty = true;
        }

        match message {
            ViewerMessage::NewSource => {
                self.view_splats = vec![];
                self.paused = false;
                self.is_loading = false;
                self.is_training = false;
                self.err = None;
            }
            ViewerMessage::DoneLoading { training: _ } => {
                self.is_loading = false;
            }
            ViewerMessage::StartLoading { training } => {
                self.is_training = *training;
                self.is_loading = true;
            }
            ViewerMessage::ViewSplats { splats, frame } => {
                if self.live_update {
                    self.view_splats.truncate(*frame);
                    log::info!("Received splat at {frame}");
                    self.view_splats.push(*splats.clone());
                }
            }
            ViewerMessage::TrainStep {
                splats,
                stats: _,
                iter: _,
                timestamp: _,
            } => {
                if self.live_update {
                    self.view_splats = vec![*splats.clone()];
                }
            }
            ViewerMessage::Error(e) => {
                self.err = Some(e.clone());
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut ViewerContext) {
        let cur_time = Instant::now();
        let delta_time = self
            .last_draw
            .map(|last| cur_time - last)
            .unwrap_or(Duration::from_millis(10));

        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show.
        if !self.is_loading && self.view_splats.is_empty() && self.err.is_none() {
            ui.heading("Load a ply file or dataset to get started.");
            ui.add_space(5.0);
            ui.label(
                r#"
Load a pretrained .ply file to view it

Or load a dataset to train on. These are zip files with:
    - a transforms.json and images, like the nerfstudio dataset format.
    - COLMAP data, containing the `images` & `sparse` folder."#,
            );

            ui.add_space(10.0);

            #[cfg(target_family = "wasm")]
            ui.scope(|ui| {
                ui.visuals_mut().override_text_color = Some(Color32::YELLOW);
                ui.heading("Note: Running in browser is still experimental");

                ui.label(
                    r#"
In browser training is slower, and lower quality than the native app.

For bigger training runs consider using the native app."#,
                );
            });

            return;
        }

        if let Some(err) = self.err.as_ref() {
            ui.label("Error: ".to_owned() + &err.to_string());
        } else if !self.view_splats.is_empty() {
            const FPS: usize = 24;
            let frame = ((self.frame * FPS as f32).floor() as usize) % self.view_splats.len();
            let splats = self.view_splats[frame].clone();

            self.draw_splats(ui, context, &splats, delta_time);

            if self.view_splats.len() > 1 {
                self.dirty = true;

                let label = if self.paused {
                    "⏸ paused"
                } else {
                    "⏵ playing"
                };

                if ui.selectable_label(!self.paused, label).clicked() {
                    self.paused = !self.paused;
                }

                if !self.paused {
                    self.frame += delta_time.as_secs_f32();
                    self.dirty = true;
                }
            }

            if self.is_training {
                ui.horizontal(|ui| {
                    ui.add_space(15.0);

                    let label = if self.paused {
                        "⏸ paused"
                    } else {
                        "⏵ training"
                    };

                    if ui.selectable_label(!self.paused, label).clicked() {
                        self.paused = !self.paused;
                        context.send_train_message(TrainMessage::Paused(self.paused));
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

                    if ui.button("⬆ Export").clicked() {
                        let splats = splats.clone();

                        let fut = async move {
                            let file = rrfd::save_file("export.ply").await;

                            // Not sure where/how to show this error if any.
                            match file {
                                Err(e) => {
                                    log::error!("Failed to save file: {e}");
                                }
                                Ok(file) => {
                                    let data = splat_export::splat_to_ply(splats).await;

                                    let data = match data {
                                        Ok(data) => data,
                                        Err(e) => {
                                            log::error!("Failed to serialize file: {e}");
                                            return;
                                        }
                                    };

                                    if let Err(e) = file.write(&data).await {
                                        log::error!("Failed to write file: {e}");
                                    }
                                }
                            }
                        };

                        tokio::task::spawn(fut);
                    }
                });
            }

            // Also redraw next frame, need to check if we're still animating.
            if self.dirty {
                ui.ctx().request_repaint();
            }
        }
    }
}
