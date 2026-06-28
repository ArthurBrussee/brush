use std::collections::VecDeque;

use brush_dataset::{
    Dataset,
    scene::{LoadImage, Scene, SceneView, ViewType},
};
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_render::AlphaMode;
use egui::{Color32, Slider, TextureOptions, pos2};
use image::GenericImageView;
use tokio::sync::oneshot;

use brush_async::Actor;

use crate::ui::{
    UiMode, draw_checkerboard,
    panels::AppPane,
    ui_process::{BackgroundStyle, TexHandle, UiProcess},
};

const TEX_CACHE_LIMIT: usize = 16;

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

enum LoadState {
    Pending(oneshot::Receiver<TexHandle>),
    Ready(TexHandle),
}

struct PreviewJob {
    view: SceneView,
    ctx: egui::Context,
    preview_edge: u32,
    reply: oneshot::Sender<TexHandle>,
}

struct PreviewLoader {
    jobs: async_channel::Sender<PreviewJob>,
    _workers: Vec<Actor>,
    cache: VecDeque<(LoadImage, LoadState)>,
    target_res: u32,
}

impl PreviewLoader {
    fn new() -> Self {
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        let (jobs, rx) = async_channel::unbounded::<PreviewJob>();
        let workers = (0..workers)
            .map(|i| {
                let actor = Actor::new(&format!("dataset-preview-{i}"));
                let rx = rx.clone();
                actor
                    .run(move || async move {
                        while let Ok(job) = rx.recv().await {
                            if let Some(tex) =
                                load_preview(job.view, job.ctx.clone(), job.preview_edge).await
                            {
                                let _ = job.reply.send(tex);
                                job.ctx.request_repaint();
                            }
                        }
                    })
                    .detach();
                actor
            })
            .collect();
        Self {
            jobs,
            _workers: workers,
            cache: VecDeque::with_capacity(TEX_CACHE_LIMIT),
            target_res: 0,
        }
    }

    fn set_target_res(&mut self, edge: u32) {
        let (lo, hi) = (edge.min(self.target_res), edge.max(self.target_res));
        if hi * 10 > lo * 11 {
            self.target_res = edge;
            self.cache.clear();
        }
    }

    /// Get a ready texture for `view`, queuing a decode on a miss. Returns
    /// `Some` only once the texture is uploaded.
    fn request(&mut self, view: &SceneView, ctx: &egui::Context) -> Option<TexHandle> {
        if let Some(tex) = self.cache_get(&view.image) {
            return Some(tex);
        }
        self.spawn_load(view, ctx);
        None
    }

    /// Look up a target in the cache. Returns `Some(tex)` when ready.
    fn cache_get(&mut self, target: &LoadImage) -> Option<TexHandle> {
        let pos = self.cache.iter().position(|(img, _)| img == target)?;

        if let LoadState::Pending(rx) = &mut self.cache[pos].1 {
            match rx.try_recv() {
                Ok(tex) => self.cache[pos].1 = LoadState::Ready(tex),
                Err(oneshot::error::TryRecvError::Empty) => return None,
                Err(oneshot::error::TryRecvError::Closed) => {
                    // Load failed (sender dropped without sending). Drop the slot
                    // so the caller can re-spawn.
                    self.cache.remove(pos);
                    return None;
                }
            }
        }
        let LoadState::Ready(tex) = &self.cache[pos].1 else {
            unreachable!()
        };
        let tex = tex.clone();
        let entry = self.cache.remove(pos)?;
        self.cache.push_front(entry);
        Some(tex)
    }

    /// Queue a load on the shared pool.
    fn spawn_load(&mut self, view: &SceneView, ctx: &egui::Context) {
        if self.cache.iter().any(|(i, _)| *i == view.image) {
            return;
        }
        let (reply, rx) = oneshot::channel();
        let _ = self.jobs.try_send(PreviewJob {
            view: view.clone(),
            ctx: ctx.clone(),
            preview_edge: self.target_res,
            reply,
        });
        self.cache
            .push_front((view.image.clone(), LoadState::Pending(rx)));
        if self.cache.len() > TEX_CACHE_LIMIT {
            self.cache.pop_back();
        }
    }
}

pub struct DatasetPanel {
    view_type: ViewType,
    cur_dataset: Dataset,

    current_view_index: Option<usize>,
    loading_start: Option<web_time::Instant>,
    loader: PreviewLoader,

    displayed: Option<(SceneView, TexHandle)>,
}

impl Default for DatasetPanel {
    fn default() -> Self {
        Self {
            view_type: ViewType::Train,
            cur_dataset: Dataset::empty(),
            current_view_index: None,
            loading_start: None,
            displayed: None,
            loader: PreviewLoader::new(),
        }
    }
}

async fn load_preview(view: SceneView, ctx: egui::Context, preview_edge: u32) -> Option<TexHandle> {
    // The preview texture is capped to the panel size for GPU/memory reasons,
    // but report the resolution training actually uses (read from the header,
    // no full decode) so the panel doesn't claim a misleadingly small size.

    let preview_load = view.image.clone().with_max_resolution(preview_edge);
    let image = preview_load.load().await.ok()?;
    let train_size = image.dimensions();

    let has_alpha = image.color().has_alpha();
    let img_size = [image.width() as usize, image.height() as usize];

    let color_img = if has_alpha {
        let data = image.into_rgba8().into_vec();
        egui::ColorImage::from_rgba_unmultiplied(img_size, &data)
    } else {
        egui::ColorImage::from_rgb(img_size, &image.into_rgb8().into_vec())
    };

    // Use the full path as the egui texture key: basenames can collide across
    // subdirectories in a dataset.
    let tex_key = view.image.path().to_string_lossy().into_owned();
    let egui_handle = ctx.load_texture(tex_key, color_img, TextureOptions::default());

    Some(TexHandle {
        handle: egui_handle,
        has_alpha,
        train_size,
    })
}

impl DatasetPanel {
    fn focus_picked(&self, process: &UiProcess) {
        let pick_scene = selected_scene(self.view_type, &self.cur_dataset);

        if let Some(idx) = self.current_view_index
            && let Some(view) = pick_scene.views.get(idx)
        {
            process.focus_view(&view.camera);
        }
    }
}

impl AppPane for DatasetPanel {
    fn title(&self) -> egui::WidgetText {
        let Some((view, tex)) = self.displayed.as_ref() else {
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
                tex.train_size.0, tex.train_size.1, mask_info
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
                self.loader = PreviewLoader::new();
                self.displayed = None;
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

        let target_view = pick_scene.views[*nearest].clone();
        self.current_view_index = Some(*nearest);

        // Size previews to the panel in physical pixels, so we neither waste
        // memory on oversized textures nor visibly downscale on large/hi-DPI
        // windows.
        let needed =
            ((ui.available_size().max_elem() * ui.ctx().pixels_per_point()).ceil() as u32).max(32);
        self.loader.set_target_res(needed);

        // Hit → swap display. Miss → the loader has queued a decode; show the
        // stale image until it lands.
        if let Some(tex) = self.loader.request(&target_view, ui.ctx()) {
            self.displayed = Some((target_view.clone(), tex));
            self.loading_start = None;

            // Also load neighbours as those are often nearby / and makes the arrow
            // keys smoother.
            let n = pick_scene.views.len();
            if n > 1 {
                let next = (*nearest + 1) % n;
                let prev = (*nearest + n - 1) % n;
                self.loader.spawn_load(&pick_scene.views[next], ui.ctx());
                self.loader.spawn_load(&pick_scene.views[prev], ui.ctx());
            }
        } else if self.loading_start.is_none() {
            self.loading_start = Some(web_time::Instant::now());
        }

        let Some((view, tex)) = self.displayed.clone() else {
            return;
        };

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

        // Letterbox fill.
        let full_rect = egui::Rect::from_min_size(cursor_min, available);
        ui.painter()
            .rect_filled(full_rect, 0.0, Color32::from_gray(20));

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

        // Overlay only when what we're showing differs from what we want, and
        // the load has been in flight long enough to be worth surfacing.
        let loading_new = view.image != target_view.image;
        if !loading_new {
            self.loading_start = None;
        } else if self
            .loading_start
            .is_some_and(|t| t.elapsed().as_secs_f32() > 0.1)
        {
            ui.painter().rect_filled(
                rect,
                0.0,
                Color32::from_rgba_unmultiplied(200, 200, 220, 80),
            );
        }

        ui.allocate_rect(full_rect, egui::Sense::click());
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

        let mut current_idx = self.current_view_index.unwrap_or(0);

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
                                self.current_view_index = Some(0);
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
                self.current_view_index = Some(current_idx);
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
                self.current_view_index = Some(current_idx);
                self.focus_picked(process);
            }

            if nav_button(ui, "◀").clicked() {
                current_idx = (current_idx + view_count - 1) % view_count;
                self.current_view_index = Some(current_idx);
                self.focus_picked(process);
            }
        });
    }
}
