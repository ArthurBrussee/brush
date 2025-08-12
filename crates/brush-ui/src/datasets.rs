use crate::{UiMode, draw_checkerboard, panels::AppPane, ui_process::UiProcess};
use brush_dataset::{
    Dataset,
    scene::{Scene, SceneView, ViewType},
};
use brush_process::message::ProcessMessage;
use egui::{
    Color32, Frame, Slider, TextureHandle, TextureOptions, collapsing_header::CollapsingState, pos2,
};
use tokio::sync::oneshot::{Receiver, channel};

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

struct SelectedView {
    index: usize,
    view_type: ViewType,
    handle: Receiver<TextureHandle>,
}

impl SelectedView {
    fn get_view<'b>(&self, dataset: &'b Dataset) -> &'b SceneView {
        let scene = selected_scene(self.view_type, dataset);
        &scene.views[self.index]
    }
}

pub struct DatasetPanel {
    view_type: ViewType,
    selected_view: Option<SelectedView>,
    last_handle: Option<TextureHandle>,
    cur_dataset: Dataset,
}

impl DatasetPanel {
    pub(crate) fn new() -> Self {
        Self {
            view_type: ViewType::Train,
            selected_view: None,
            last_handle: None,
            cur_dataset: Dataset::empty(),
        }
    }

    fn dataset_controls_box(
        &mut self,
        ui: &egui::Ui,
        process: &UiProcess,
        rect: egui::Rect,
        nearest_view_ind: &mut usize,
        pick_scene: &Scene,
    ) {
        // Controls window in top left
        let id = ui.id().with("dataset_controls_box");
        egui::Area::new(id)
            .kind(egui::UiKind::Window)
            .current_pos(egui::pos2(rect.min.x, rect.min.y))
            .movable(false)
            .show(ui.ctx(), |ui| {
                // Add transparent background frame
                let style = ui.style_mut();
                let fill = style.visuals.window_fill;
                style.visuals.window_fill =
                    Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), 200);
                let frame = Frame::window(style);

                frame.show(ui, |ui| {
                    if let Some(selected) = self.selected_view.as_ref() {
                        let selected_view = selected.get_view(&self.cur_dataset);
                        let image_name = selected_view
                            .image
                            .path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("Unknown");

                        // Custom title bar using egui's CollapsingState
                        let state = CollapsingState::load_with_default_open(
                            ui.ctx(),
                            ui.id().with("dataset_controls_collapse"),
                            false,
                        );

                        // Show a header with image name
                        state
                            .show_header(ui, |ui| {
                                ui.label(egui::RichText::new(image_name).strong());
                            })
                            .body_unindented(|ui| {
                                ui.set_max_width(200.0);
                                ui.spacing_mut().item_spacing.y = 6.0;

                                let view_count = pick_scene.views.len();

                                // Navigation buttons and slider
                                ui.horizontal(|ui| {
                                    let mut interacted = false;
                                    if ui.button("⏪").clicked() {
                                        *nearest_view_ind =
                                            (*nearest_view_ind + view_count - 1) % view_count;
                                        interacted = true;
                                    }
                                    if ui
                                        .add(
                                            Slider::new(nearest_view_ind, 0..=view_count - 1)
                                                .suffix(format!("/ {view_count}"))
                                                .custom_formatter(|num, _| {
                                                    format!("{}", num as usize + 1)
                                                })
                                                .custom_parser(|s| {
                                                    s.parse::<usize>().ok().map(|n| n as f64 - 1.0)
                                                }),
                                        )
                                        .dragged()
                                    {
                                        interacted = true;
                                    }
                                    if ui.button("⏩").clicked() {
                                        *nearest_view_ind = (*nearest_view_ind + 1) % view_count;
                                        interacted = true;
                                    }

                                    if interacted {
                                        process.focus_view(&pick_scene.views[*nearest_view_ind]);
                                    }
                                });

                                ui.add_space(4.0);

                                // View type selector (train/eval)
                                if self.cur_dataset.eval.is_some() {
                                    ui.horizontal(|ui| {
                                        for (t, l) in [ViewType::Train, ViewType::Eval]
                                            .into_iter()
                                            .zip(["train", "eval"])
                                        {
                                            if ui.selectable_label(self.view_type == t, l).clicked()
                                            {
                                                self.view_type = t;
                                                *nearest_view_ind = 0;
                                                process.focus_view(
                                                    &pick_scene.views[*nearest_view_ind],
                                                );
                                            };
                                        }
                                    });

                                    ui.add_space(4.0);
                                }

                                // Image info
                                let mask_info = if selected_view.image.has_alpha() {
                                    if !selected_view.image.is_masked() {
                                        "rgb + alpha transparency"
                                    } else {
                                        "rgb, masked"
                                    }
                                } else {
                                    "rgb"
                                };

                                ui.label(
                                    egui::RichText::new(format!(
                                        "{}x{} {}",
                                        selected_view.image.width(),
                                        selected_view.image.height(),
                                        mask_info
                                    ))
                                    .size(11.0),
                                );
                            });
                    }
                });
            });
    }
}

impl AppPane for DatasetPanel {
    fn title(&self) -> String {
        "Dataset".to_owned()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        process.ui_mode() == UiMode::Default && process.is_training()
    }

    fn on_message(&mut self, message: &ProcessMessage, process: &UiProcess) {
        match message {
            ProcessMessage::NewSource => {
                *self = Self::new();
            }
            ProcessMessage::Dataset { dataset } => {
                if let Some(view) = dataset.train.views.first() {
                    process.focus_view(view);
                }
                self.cur_dataset = dataset.clone();
            }
            ProcessMessage::ViewSplats { up_axis, .. } => {
                // Training does also handle this but in the dataset.
                if process.is_training()
                    && let Some(up_axis) = up_axis
                {
                    process.set_model_up(*up_axis);

                    if let Some(view) = self.cur_dataset.train.views.first() {
                        process.focus_view(view);
                    }
                }
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        let pick_scene = selected_scene(self.view_type, &self.cur_dataset).clone();
        let total_transform =
            process.model_local_to_world() * process.current_camera().local_to_world();
        let mut nearest_view_ind = pick_scene.get_nearest_view(total_transform);

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
                let (sender, handle) = channel();

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
                        let data = image.into_rgba8().into_vec();
                        egui::ColorImage::from_rgba_unmultiplied(img_size, &data)
                    } else {
                        egui::ColorImage::from_rgb(img_size, &image.into_rgb8().into_vec())
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

            if let Some(selected) = self.selected_view.as_mut() {
                if let Ok(texture_handle) = selected.handle.try_recv() {
                    self.last_handle = Some(texture_handle);
                }

                if let Some(texture_handle) = &mut self.last_handle {
                    let selected_view = selected.get_view(&self.cur_dataset);

                    let mut size = ui.available_size();
                    let aspect_ratio = texture_handle.aspect_ratio();

                    if size.x / size.y > aspect_ratio {
                        size.x = size.y * aspect_ratio;
                    } else {
                        size.y = size.x / aspect_ratio;
                    }
                    let min = ui.cursor().min;
                    let rect = egui::Rect::from_min_size(min, size);

                    if selected_view.image.has_alpha() {
                        if selected_view.image.is_masked() {
                            draw_checkerboard(ui, rect, egui::Color32::DARK_RED);
                        } else {
                            draw_checkerboard(ui, rect, egui::Color32::WHITE);
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

                    // Draw the controls overlay in the top left
                    self.dataset_controls_box(ui, process, rect, nearest, &pick_scene);
                }
            }
        }

        if process.is_loading() && process.is_training() {
            ui.label("Loading...");
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
