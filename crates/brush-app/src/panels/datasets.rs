use crate::app::{AppContext, AppPanel};
use brush_dataset::scene::{Scene, SceneView, ViewType};
use brush_process::process_loop::ProcessMessage;
use egui::{Color32, Slider, TextureHandle, TextureOptions, pos2};
use tokio::sync::oneshot::Receiver;

struct SelectedView {
    index: usize,
    view_type: ViewType,
    handle: Receiver<TextureHandle>,
}

fn selected_scene(t: ViewType, context: &AppContext) -> &Scene {
    if let Some(eval_scene) = context.dataset.eval.as_ref() {
        match t {
            ViewType::Train => &context.dataset.train,
            _ => eval_scene,
        }
    } else {
        &context.dataset.train
    }
}

impl SelectedView {
    fn get_view<'a>(&'a self, context: &'a AppContext) -> &'a SceneView {
        &selected_scene(self.view_type, context).views[self.index]
    }
}

pub(crate) struct DatasetPanel {
    view_type: ViewType,
    selected_view: Option<SelectedView>,
    last_handle: Option<TextureHandle>,
}

impl DatasetPanel {
    pub(crate) fn new() -> Self {
        Self {
            view_type: ViewType::Train,
            selected_view: None,
            last_handle: None,
        }
    }
}

impl AppPanel for DatasetPanel {
    fn title(&self) -> String {
        "Dataset".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, context: &mut AppContext) {
        match message {
            ProcessMessage::NewSource => {
                *self = Self::new();
            }
            ProcessMessage::Dataset { dataset } => {
                if let Some(view) = dataset.train.views.first() {
                    context.focus_view(view);
                }
                context.dataset = dataset.clone();
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut AppContext) {
        let pick_scene = selected_scene(self.view_type, context).clone();

        let mut nearest_view_ind = pick_scene.get_nearest_view(context.camera.local_to_world());

        if let Some(nearest) = nearest_view_ind.as_mut() {
            // Update image if dirty.
            // For some reason (bug in egui), this _has_ to be before drawing the image.
            // Otherwise egui releases the image too early and wgpu crashes.
            let mut dirty = self.selected_view.is_none();

            if let Some(view) = self.selected_view.as_ref() {
                dirty |= view.index != *nearest;
                dirty |= view.view_type != self.view_type;
            }

            if dirty {
                let (sender, handle) = tokio::sync::oneshot::channel();

                let ctx = ui.ctx().clone();

                // Clone the arc to send to the task.
                let views = pick_scene.views.clone();
                let cur_nearest = *nearest;

                tokio_with_wasm::alias::spawn(async move {
                    let view = &views[cur_nearest];

                    // When selecting images super rapidly, might happen, don't waste resources loading.
                    if sender.is_closed() {
                        return;
                    }
                    let image = view
                        .image
                        .load()
                        .await
                        .expect("Failed to load dataset image");
                    if sender.is_closed() {
                        return;
                    }
                    let img_size = [image.width() as usize, image.height() as usize];
                    let color_img = if image.color().has_alpha() {
                        let data = image.to_rgba8().into_vec();
                        egui::ColorImage::from_rgba_unmultiplied(img_size, &data)
                    } else {
                        egui::ColorImage::from_rgb(img_size, &image.to_rgb8().into_vec())
                    };

                    // If channel is gone, that's fine.
                    let _ = sender.send(ctx.load_texture(
                        "nearest_view_tex",
                        color_img,
                        TextureOptions::default(),
                    ));
                    // Show updated texture asap.
                    ctx.request_repaint();
                });

                self.selected_view = Some(SelectedView {
                    index: cur_nearest,
                    view_type: self.view_type,
                    handle,
                });
            }

            let view_count = pick_scene.views.len();

            if let Some(selected) = self.selected_view.as_mut() {
                if let Ok(texture_handle) = selected.handle.try_recv() {
                    self.last_handle = Some(texture_handle);
                }

                if let Some(texture_handle) = &mut self.last_handle {
                    let selected_view = selected.get_view(context);

                    let size = brush_ui::size_for_splat_view(ui);
                    let mut size = size.floor();
                    let aspect_ratio =
                        selected_view.image.width() as f32 / selected_view.image.height() as f32;

                    if size.x / size.y > aspect_ratio {
                        size.x = size.y * aspect_ratio;
                    } else {
                        size.y = size.x / aspect_ratio;
                    }
                    let min = ui.cursor().min;
                    let rect = egui::Rect::from_min_size(min, size);

                    if selected_view.image.color().has_alpha() {
                        if selected_view.image.is_masked() {
                            brush_ui::draw_checkerboard(ui, rect, egui::Color32::DARK_RED);
                        } else {
                            brush_ui::draw_checkerboard(ui, rect, egui::Color32::WHITE);
                        }
                    }

                    ui.painter().image(
                        texture_handle.id(),
                        rect,
                        egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );

                    if !selected.handle.is_terminated() {
                        ui.painter().rect_filled(
                            rect,
                            0.0,
                            Color32::from_rgba_unmultiplied(100, 100, 120, 30),
                        );
                    }

                    ui.allocate_rect(rect, egui::Sense::click());
                }

                ui.horizontal(|ui| {
                    let mut interacted = false;
                    if ui.button("⏪").clicked() {
                        *nearest = (*nearest + view_count - 1) % view_count;
                        interacted = true;
                    }
                    if ui
                        .add(
                            Slider::new(nearest, 0..=view_count - 1)
                                .suffix(format!("/ {view_count}"))
                                .custom_formatter(|num, _| format!("{}", num as usize + 1))
                                .custom_parser(|s| s.parse::<usize>().ok().map(|n| n as f64 - 1.0)),
                        )
                        .dragged()
                    {
                        interacted = true;
                    }
                    if ui.button("⏩").clicked() {
                        *nearest = (*nearest + 1) % view_count;
                        interacted = true;
                    }

                    ui.add_space(10.0);

                    if context.dataset.eval.is_some() {
                        for (t, l) in [ViewType::Train, ViewType::Eval]
                            .into_iter()
                            .zip(["train", "eval"])
                        {
                            if ui.selectable_label(self.view_type == t, l).clicked() {
                                self.view_type = t;
                                *nearest = 0;
                                interacted = true;
                            };
                        }
                    }

                    if interacted {
                        context.focus_view(&pick_scene.views[*nearest]);
                    }

                    ui.add_space(10.0);

                    let selected_view = selected.get_view(context);
                    let mask_info = if selected_view.image.color().has_alpha() {
                        if !selected_view.image.is_masked() {
                            "rgb + alpha transparency"
                        } else {
                            "rgb, masked"
                        }
                    } else {
                        "rgb"
                    };

                    let info = format!(
                        "{} ({}x{} {})",
                        selected_view.image.path.to_string_lossy(),
                        selected_view.image.width(),
                        selected_view.image.height(),
                        mask_info
                    );
                    ui.label(info);
                });
            }
        }

        if context.loading() && context.training() {
            ui.label("Loading...");
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
