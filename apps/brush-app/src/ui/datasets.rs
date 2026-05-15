use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use brush_dataset::{
    Dataset,
    scene::{Scene, SceneView, ViewType},
};
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_render::AlphaMode;
use egui::{Color32, Rect, Slider, TextureOptions, pos2};

use brush_async::Actor;

use crate::ui::{
    UiMode, draw_checkerboard,
    panels::AppPane,
    ui_process::{BackgroundStyle, TexHandle, UiProcess},
};

fn selected_scene(t: ViewType, dataset: &Dataset) -> &Scene {
    match t {
        ViewType::Train => &dataset.train,
        _ => {
            if let Some(eval_scene) = dataset.eval.as_ref() {
                eval_scene
            } else {
                &dataset.train
            }
        }
    }
}

pub struct DatasetPanel {
    view_type: ViewType,
    cur_dataset: Dataset,

    current_view_index: Cell<Option<usize>>,
    loading_start: Option<web_time::Instant>,

    loading: Arc<AtomicBool>,
    view_send: tokio::sync::watch::Sender<Option<(SceneView, egui::Context)>>,
    loaded_view_rec: tokio::sync::watch::Receiver<Option<(SceneView, TexHandle)>>,

    // Keep actor live.
    _actor: Actor,
}

impl Default for DatasetPanel {
    fn default() -> Self {
        let (view_send, mut view_rec) = tokio::sync::watch::channel(None);
        let (tex_send, tex_rec) = tokio::sync::watch::channel(None);

        let loading = Arc::new(AtomicBool::new(false));

        let loading_task = loading.clone();

        let actor = Actor::new("dataset-preview");
        actor.run(move || async move {
            loop {
                let view = view_rec
                    .wait_for(|x: &Option<(SceneView, egui::Context)>| x.is_some())
                    .await;

                let Ok(view) = view else {
                    return;
                };
                let (view, ctx) = view.as_ref().unwrap().clone();
                let image = view
                    .image
                    .load()
                    .await
                    .expect("Failed to load dataset image");

                loading_task.store(true, Ordering::SeqCst);

                let has_alpha = image.color().has_alpha();
                let img_size = [image.width() as usize, image.height() as usize];

                // Create blurred background: downscale 32x then blur
                let bg_width = (image.width() / 32).max(1);
                let bg_height = (image.height() / 32).max(1);
                let blurred = image
                    .resize(bg_width, bg_height, image::imageops::FilterType::Triangle)
                    .fast_blur(6.0);
                let blurred_size = [blurred.width() as usize, blurred.height() as usize];
                let blurred_img =
                    egui::ColorImage::from_rgb(blurred_size, &blurred.into_rgb8().into_vec());

                brush_async::yield_now().await;

                let color_img = if has_alpha {
                    let data = image.into_rgba8().into_vec();
                    egui::ColorImage::from_rgba_unmultiplied(img_size, &data)
                } else {
                    egui::ColorImage::from_rgb(img_size, &image.into_rgb8().into_vec())
                };

                let image_name = view.image.img_name();
                let egui_handle =
                    ctx.load_texture(image_name, color_img, TextureOptions::default());
                let blurred_handle = ctx.load_texture(
                    format!("{}_blurred", view.image.img_name()),
                    blurred_img,
                    TextureOptions::default(),
                );
                brush_async::yield_now().await;

                let handle = TexHandle {
                    handle: egui_handle,
                    has_alpha,
                    blurred_bg: Some(blurred_handle),
                };

                // If channel is gone, that's fine.
                let _ = tex_send.send(Some((view, handle)));
                loading_task.store(false, Ordering::SeqCst);

                ctx.request_repaint();
            }
        });

        Self {
            view_type: ViewType::Train,
            cur_dataset: Dataset::empty(),
            view_send,
            loaded_view_rec: tex_rec,
            current_view_index: Cell::new(None),
            loading_start: None,
            loading,
            _actor: actor,
        }
    }
}

impl DatasetPanel {
    pub fn set_selected_view(&mut self, view: &SceneView, ctx: &egui::Context) {
        let _ = self.view_send.send(Some((view.clone(), ctx.clone())));
        self.loading_start = Some(web_time::Instant::now());
    }

    pub fn is_selected_view_loading(&self) -> bool {
        self.loading.load(Ordering::SeqCst)
    }

    fn focus_picked(&self, process: &UiProcess) {
        let pick_scene = selected_scene(self.view_type, &self.cur_dataset);

        if let Some(idx) = self.current_view_index.get()
            && let Some(view) = pick_scene.views.get(idx)
        {
            process.focus_view(&view.camera);
        }
    }
}

impl AppPane for DatasetPanel {
    fn title(&self) -> egui::WidgetText {
        let Some((view, tex)) = self.loaded_view_rec.borrow().clone() else {
            return "Dataset".into();
        };

        let img_name = view.image.img_name();

        // Try to get image info from texture handle
        let mask_info = if tex.has_alpha {
            if view.image.alpha_mode() == AlphaMode::Transparent {
                "rgba"
            } else {
                "masked"
            }
        } else {
            "rgb"
        };

        let mut job = egui::text::LayoutJob::default();
        job.append(
            &img_name,
            0.0,
            egui::TextFormat {
                color: Color32::WHITE,
                ..Default::default()
            },
        );
        job.append(
            &format!(
                "  |  {}x{} {}",
                tex.handle.size()[0],
                tex.handle.size()[1],
                mask_info
            ),
            0.0,
            egui::TextFormat {
                color: Color32::from_rgb(140, 140, 140),
                ..Default::default()
            },
        );
        job.into()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default && process.is_training()
    }

    fn on_message(&mut self, message: &ProcessMessage, process: &UiProcess) {
        match message {
            ProcessMessage::NewProcess => {
                *self = Self::default();
            }
            ProcessMessage::TrainMessage(TrainMessage::Dataset { dataset }) => {
                if let Some(view) = dataset.train.views.first() {
                    process.focus_view(&view.camera);
                }
                self.cur_dataset = dataset.clone();
            }
            ProcessMessage::SplatsUpdated { up_axis, .. } => {
                // Training does also handle this but in the dataset.
                if process.is_training()
                    && let Some(up_axis) = up_axis
                {
                    process.set_model_up(*up_axis);
                    if let Some(view) = self.cur_dataset.train.views.first() {
                        process.focus_view(&view.camera);
                    }
                }
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        let mv = process.current_camera().world_to_local() * process.model_local_to_world();
        let pick_scene = selected_scene(self.view_type, &self.cur_dataset).clone();
        let mut nearest_view_ind = pick_scene.get_nearest_view(mv.inverse());

        let Some(nearest) = nearest_view_ind.as_mut() else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Waiting for training to start")
                        .size(14.0)
                        .color(Color32::from_rgb(140, 140, 140))
                        .italics(),
                );
            });
            return;
        };

        let Some((view, tex)) = self.loaded_view_rec.borrow().clone() else {
            self.set_selected_view(&pick_scene.views[*nearest], ui.ctx());
            return;
        };

        // Update image if dirty.
        let dirty = view.image != pick_scene.views[*nearest].image;

        if dirty {
            self.set_selected_view(&pick_scene.views[*nearest], ui.ctx());
        }

        // if training views have alpha, show a background checker. Masked images
        // should still use a black background.
        let background = if tex.has_alpha && view.image.alpha_mode() == AlphaMode::Transparent {
            BackgroundStyle::Checkerboard
        } else {
            BackgroundStyle::Black
        };
        process.set_background_style(background);

        let available = ui.available_size();
        let cursor_min = ui.cursor().min;
        let aspect_ratio = tex.handle.aspect_ratio();

        let mut size = available;
        if size.x / size.y > aspect_ratio {
            size.x = size.y * aspect_ratio;
        } else {
            size.y = size.x / aspect_ratio;
        }

        // Center the image in the available space
        let offset_x = (available.x - size.x) / 2.0;
        let offset_y = (available.y - size.y) / 2.0;
        let min = cursor_min + egui::vec2(offset_x, offset_y);
        let rect = egui::Rect::from_min_size(min, size);

        // Blurred background for letterbox areas
        let full_rect = egui::Rect::from_min_size(cursor_min, available);
        if let Some(blurred) = &tex.blurred_bg {
            ui.painter().image(
                blurred.id(),
                full_rect,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::from_gray(80),
            );
        } else {
            ui.painter()
                .rect_filled(full_rect, 0.0, Color32::from_gray(30));
        }

        if tex.has_alpha {
            if view.image.alpha_mode() == AlphaMode::Masked {
                draw_checkerboard(ui, rect, egui::Color32::DARK_RED);
            } else {
                draw_checkerboard(ui, rect, egui::Color32::WHITE);
            }
        }

        ui.painter().image(
            tex.handle.id(),
            rect,
            egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        if self.is_selected_view_loading() {
            if self
                .loading_start
                .is_some_and(|t| t.elapsed().as_secs_f32() > 0.1)
            {
                ui.painter().rect_filled(
                    rect,
                    0.0,
                    Color32::from_rgba_unmultiplied(200, 200, 220, 80),
                );
            }
        } else {
            // Clear loading start when done
            self.loading_start = None;
        }

        ui.allocate_rect(full_rect, egui::Sense::click());
        self.current_view_index.set(Some(*nearest));
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }

    fn top_bar_right_ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        let pick_scene = selected_scene(self.view_type, &self.cur_dataset);
        let view_count = pick_scene.views.len();

        if view_count == 0 {
            return;
        }

        let mut current_idx = self.current_view_index.get().unwrap_or(0);

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if self.cur_dataset.eval.is_some() {
                let gear_button =
                    egui::Button::new(egui::RichText::new("⚙").size(14.0).color(Color32::WHITE))
                        .fill(egui::Color32::from_rgb(70, 70, 75))
                        .corner_radius(6.0)
                        .min_size(egui::vec2(22.0, 18.0));

                let response = ui.add(gear_button);

                egui::containers::Popup::from_toggle_button_response(&response)
                    .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                    .show(|ui| {
                        ui.label("View");
                        for (t, l) in [(ViewType::Train, "train"), (ViewType::Eval, "eval")] {
                            if ui.selectable_label(self.view_type == t, l).clicked() {
                                self.view_type = t;
                                self.current_view_index.set(Some(0));
                                self.focus_picked(process);
                            }
                        }
                    });

                ui.add_space(6.0);
            }

            let nav_button = |ui: &mut egui::Ui, icon: &str| {
                ui.add(
                    egui::Button::new(
                        egui::RichText::new(icon)
                            .size(10.0)
                            .color(Color32::from_rgb(200, 200, 200)),
                    )
                    .fill(egui::Color32::from_rgb(60, 60, 65))
                    .corner_radius(6.0)
                    .min_size(egui::vec2(20.0, 18.0)),
                )
            };

            if nav_button(ui, "▶").clicked() {
                current_idx = (current_idx + 1) % view_count;
                self.current_view_index.set(Some(current_idx));
                self.focus_picked(process);
            }

            let mut idx = current_idx;
            if ui
                .add(
                    Slider::new(&mut idx, 0..=view_count - 1)
                        .suffix(format!("/ {view_count}"))
                        .custom_formatter(|num, _| format!("{}", num as usize + 1))
                        .custom_parser(|s| s.parse::<usize>().ok().map(|n| n as f64 - 1.0)),
                )
                .changed()
            {
                current_idx = idx;
                self.current_view_index.set(Some(current_idx));
                self.focus_picked(process);
            }

            if nav_button(ui, "◀").clicked() {
                current_idx = (current_idx + view_count - 1) % view_count;
                self.current_view_index.set(Some(current_idx));
                self.focus_picked(process);
            }
        });
    }
}
