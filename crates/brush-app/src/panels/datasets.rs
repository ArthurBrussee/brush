use crate::app::{AppContext, AppPanel};
use brush_process::process_loop::ProcessMessage;
use brush_train::scene::{Scene, ViewType};
use egui::{pos2, Slider, TextureHandle, TextureOptions};

pub(crate) struct DatasetPanel {
    view_type: ViewType,
    selected_view: Option<(usize, ViewType, TextureHandle)>,
    loading: bool,
}

impl DatasetPanel {
    pub(crate) fn new() -> Self {
        Self {
            view_type: ViewType::Train,
            selected_view: None,
            loading: false,
        }
    }
}

impl DatasetPanel {
    fn selected_scene(&self, context: &AppContext) -> Scene {
        if let Some(eval_scene) = context.dataset.eval.as_ref() {
            match self.view_type {
                ViewType::Train => context.dataset.train.clone(),
                _ => eval_scene.clone(),
            }
        } else {
            context.dataset.train.clone()
        }
    }
}

impl AppPanel for DatasetPanel {
    fn title(&self) -> String {
        "Dataset".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, context: &mut AppContext) {
        match message {
            ProcessMessage::NewSource | ProcessMessage::DoneLoading { .. } => {
                self.loading = false;
            }
            ProcessMessage::StartLoading { training } => {
                self.loading = *training;
            }
            ProcessMessage::Dataset { data: d } => {
                // Set train view to last loaded camera.
                if let Some(view) = d.train.views.last() {
                    context.focus_view(&view.camera);
                }
                context.dataset = d.clone();
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut AppContext) {
        let mut nearest_view_ind = self
            .selected_scene(context)
            .get_nearest_view(&context.camera);

        if let Some(nearest) = nearest_view_ind.as_mut() {
            // Update image if dirty.
            // For some reason (bug in egui), this _has_ to be before drawing the image.
            // Otherwise egui releases the image too early and wgpu crashes.
            let mut dirty = self.selected_view.is_none();

            if let Some(view) = self.selected_view.as_ref() {
                dirty |= view.0 != *nearest;
                dirty |= view.1 != self.view_type;
            }

            if dirty {
                let image = &self.selected_scene(context).views[*nearest].image;
                let img_size = [image.width() as usize, image.height() as usize];
                let color_img = if image.color().has_alpha() {
                    egui::ColorImage::from_rgba_unmultiplied(img_size, &image.to_rgba8().into_vec())
                } else {
                    egui::ColorImage::from_rgb(img_size, &image.to_rgb8().into_vec())
                };

                self.selected_view = Some((
                    *nearest,
                    self.view_type,
                    ui.ctx()
                        .load_texture("nearest_view_tex", color_img, TextureOptions::default()),
                ));
            }

            let view_count = self.selected_scene(context).views.len();

            if let Some(selected) = self.selected_view.as_ref() {
                let img_size = selected.2.size();
                let size = brush_ui::size_for_splat_view(ui);

                let size = egui::Image::new(&selected.2).shrink_to_fit().calc_size(
                    size,
                    Some(egui::vec2(img_size[0] as f32, img_size[1] as f32)),
                );
                let min = ui.cursor().min;
                let rect = egui::Rect::from_min_size(min, size);

                brush_ui::draw_checkerboard(ui, rect);
                ui.painter().image(
                    selected.2.id(),
                    rect,
                    egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

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
                    let cam = &self.selected_scene(context).views[*nearest].camera;
                    context.focus_view(cam);
                }

                ui.add_space(10.0);

                let views = &self.selected_scene(context).views;
                let info = format!(
                    "{} ({}x{})",
                    views[*nearest].name,
                    views[*nearest].image.width(),
                    views[*nearest].image.height()
                );
                ui.label(info);
            });
        }

        if self.loading {
            ui.label("Loading...");
        }
    }
}
