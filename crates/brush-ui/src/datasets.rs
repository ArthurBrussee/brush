use crate::{
    UiMode, draw_checkerboard,
    panels::AppPane,
    ui_process::{BackgroundStyle, TexHandle, UiProcess},
};
use brush_dataset::{
    Dataset,
    scene::{Scene, SceneView, ViewType},
};
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_render::AlphaMode;
use egui::{
    Color32, Frame, Rect, Slider, TextureOptions, collapsing_header::CollapsingState, pos2,
};

use tokio_with_wasm::alias as tokio_wasm;

#[derive(Clone)]
pub struct SelectedView {
    pub view: Option<SceneView>,
    pub tex: tokio::sync::watch::Receiver<Option<TexHandle>>,
}

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
    selected_view: SelectedView,
    sender: tokio::sync::watch::Sender<Option<TexHandle>>,
    loading_task: Option<tokio_wasm::task::JoinHandle<()>>,
}

impl DatasetPanel {
    pub(crate) fn new() -> Self {
        let (sender, tex) = tokio::sync::watch::channel(None);

        Self {
            view_type: ViewType::Train,
            cur_dataset: Dataset::empty(),
            loading_task: None,
            sender,
            selected_view: SelectedView { view: None, tex },
        }
    }

    pub fn set_selected_view(&mut self, view: &SceneView, ctx: &egui::Context) {
        let view_send = view.clone();

        if let Some(task) = self.loading_task.take() {
            task.abort();
        }

        let sender = self.sender.clone();
        let ctx = ctx.clone();

        self.loading_task = Some(tokio_with_wasm::alias::spawn(async move {
            // When selecting images super rapidly, might happen, don't waste resources loading.
            let image = view_send
                .image
                .load()
                .await
                .expect("Failed to load dataset image");

            if sender.is_closed() {
                return;
            }

            // Yield in case we're cancelled.
            tokio_wasm::task::yield_now().await;

            let has_alpha = image.color().has_alpha();
            let img_size = [image.width() as usize, image.height() as usize];

            // Create blurred background: downscale 32x then blur
            let bg_width = (image.width() / 32).max(1);
            let bg_height = (image.height() / 32).max(1);
            let blurred = image
                .resize(bg_width, bg_height, image::imageops::FilterType::Triangle)
                .blur(6.0);
            let blurred_size = [blurred.width() as usize, blurred.height() as usize];
            let blurred_img =
                egui::ColorImage::from_rgb(blurred_size, &blurred.into_rgb8().into_vec());

            // Yield in case we're cancelled.
            tokio_wasm::task::yield_now().await;

            let color_img = if has_alpha {
                let data = image.into_rgba8().into_vec();
                egui::ColorImage::from_rgba_unmultiplied(img_size, &data)
            } else {
                egui::ColorImage::from_rgb(img_size, &image.into_rgb8().into_vec())
            };

            let image_name = view_send.image.img_name();
            let egui_handle = ctx.load_texture(image_name, color_img, TextureOptions::default());
            let blurred_handle = ctx.load_texture(
                format!("{}_blurred", view_send.image.img_name()),
                blurred_img,
                TextureOptions::default(),
            );

            // Yield in case we're cancelled.
            tokio_wasm::task::yield_now().await;

            // If channel is gone, that's fine.
            let _ = sender.send(Some(TexHandle {
                handle: egui_handle,
                has_alpha,
                blurred_bg: Some(blurred_handle),
            }));
            // Show updated texture asap.
            ctx.request_repaint();
        }));
        self.selected_view.view = Some(view.clone());
    }

    pub fn is_selected_view_loading(&self) -> bool {
        self.loading_task.as_ref().is_some_and(|t| !t.is_finished())
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
            ProcessMessage::NewProcess => {
                *self = Self::new();
            }
            ProcessMessage::TrainMessage(TrainMessage::Dataset { dataset }) => {
                if let Some(view) = dataset.train.views.first() {
                    process.focus_view(&view.camera);
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
                        process.focus_view(&view.camera);
                    }
                }
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, process: &UiProcess) {
        let pick_scene = selected_scene(self.view_type, &self.cur_dataset).clone();
        let mv = process.current_camera().world_to_local() * process.model_local_to_world();
        let mut nearest_view_ind = pick_scene.get_nearest_view(mv.inverse());

        if let Some(nearest) = nearest_view_ind.as_mut() {
            // Update image if dirty.
            let dirty = self
                .selected_view
                .view
                .as_ref()
                .is_none_or(|view| view.image != pick_scene.views[*nearest].image);

            if dirty {
                self.set_selected_view(&pick_scene.views[*nearest], ui.ctx());
            }

            if let Some(selected_view) = &self.selected_view.view {
                let last_handle = self.selected_view.tex.borrow();

                if let Some(texture_handle) = last_handle.as_ref() {
                    // if training views have alpha, show a background checker. Masked images
                    // should still use a black background.
                    let background = if texture_handle.has_alpha
                        && selected_view.image.alpha_mode() == AlphaMode::Transparent
                    {
                        BackgroundStyle::Checkerboard
                    } else {
                        BackgroundStyle::Black
                    };
                    process.set_background_style(background);

                    let available = ui.available_size();
                    let cursor_min = ui.cursor().min;
                    let aspect_ratio = texture_handle.handle.aspect_ratio();

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
                    if let Some(blurred) = &texture_handle.blurred_bg {
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

                    if texture_handle.has_alpha {
                        if selected_view.image.alpha_mode() == AlphaMode::Masked {
                            draw_checkerboard(ui, rect, egui::Color32::DARK_RED);
                        } else {
                            draw_checkerboard(ui, rect, egui::Color32::WHITE);
                        }
                    }

                    // Draw the main image on top
                    ui.painter().image(
                        texture_handle.handle.id(),
                        rect,
                        egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );

                    if self.is_selected_view_loading() {
                        ui.painter().rect_filled(
                            rect,
                            0.0,
                            Color32::from_rgba_unmultiplied(200, 200, 220, 80),
                        );
                    }

                    ui.allocate_rect(full_rect, egui::Sense::click());

                    // Controls window in top left of the panel (not offset with image)
                    let id = ui.id().with("dataset_controls_box");
                    egui::Area::new(id)
                        .kind(egui::UiKind::Window)
                        .current_pos(egui::pos2(cursor_min.x, cursor_min.y))
                        .movable(false)
                        .show(ui.ctx(), |ui| {
                            // Add transparent background frame
                            let style = ui.style_mut();
                            let fill = style.visuals.window_fill;
                            style.visuals.window_fill =
                                Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), 200);
                            let frame = Frame::window(style);

                            frame.show(ui, |ui| {
                                // Custom title bar using egui's CollapsingState
                                let state = CollapsingState::load_with_default_open(
                                    ui.ctx(),
                                    ui.id().with("dataset_controls_collapse"),
                                    false,
                                );

                                // Show a header with image name
                                state
                                    .show_header(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new(texture_handle.handle.name())
                                                .strong(),
                                        );
                                    })
                                    .body_unindented(|ui| {
                                        ui.set_max_width(200.0);
                                        ui.spacing_mut().item_spacing.y = 6.0;

                                        let view_count = pick_scene.views.len();

                                        // Navigation buttons and slider
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
                                                        .custom_formatter(|num, _| {
                                                            format!("{}", num as usize + 1)
                                                        })
                                                        .custom_parser(|s| {
                                                            s.parse::<usize>()
                                                                .ok()
                                                                .map(|n| n as f64 - 1.0)
                                                        }),
                                                )
                                                .dragged()
                                            {
                                                interacted = true;
                                            }
                                            if ui.button("⏩").clicked() {
                                                *nearest = (*nearest + 1) % view_count;
                                                interacted = true;
                                            }

                                            if interacted {
                                                process
                                                    .focus_view(&pick_scene.views[*nearest].camera);
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
                                                    if ui
                                                        .selectable_label(self.view_type == t, l)
                                                        .clicked()
                                                    {
                                                        self.view_type = t;
                                                        *nearest = 0;
                                                        process.focus_view(
                                                            &pick_scene.views[*nearest].camera,
                                                        );
                                                    };
                                                }
                                            });

                                            ui.add_space(4.0);
                                        }

                                        // Image info

                                        let mask_info = if texture_handle.has_alpha {
                                            if selected_view.image.alpha_mode()
                                                == AlphaMode::Transparent
                                            {
                                                "rgb, alpha transparency"
                                            } else {
                                                "rgb, masked"
                                            }
                                        } else {
                                            "rgb"
                                        };

                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{}x{} {}",
                                                texture_handle.handle.size()[0],
                                                texture_handle.handle.size()[1],
                                                mask_info
                                            ))
                                            .size(11.0),
                                        );
                                    });
                            });
                        });
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
