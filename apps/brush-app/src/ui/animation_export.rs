//! Export popup + background driver for rendering the animation to an MP4.
//! The actual render/encode lives in `brush_anim`; this owns the UI, the
//! worker thread, and progress reporting. Native only.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use brush_async::Actor;
use brush_render::{camera::Camera, gaussian_splats::Splats};
use egui::Align2;
use glam::Vec3;

/// Resolution presets offered in the popup.
const PRESETS: [(&str, u32, u32); 3] = [
    ("HD", 1280, 720),
    ("Full HD", 1920, 1080),
    ("UHD", 3840, 2160),
];

/// Everything needed to render and encode the video, gathered on the UI thread.
pub struct ExportJob {
    pub splats: Splats,
    pub cameras: Vec<Camera>,
    pub fps: usize,
    pub background: Vec3,
    pub splat_scale: Option<f32>,
    pub width: u32,
    pub height: u32,
    pub path: PathBuf,
}

#[derive(Default)]
struct Progress {
    rendered: AtomicUsize,
    total: AtomicUsize,
    running: AtomicBool,
    error: std::sync::Mutex<Option<String>>,
}

pub struct Exporter {
    open: bool,
    width: u32,
    height: u32,
    path: Option<PathBuf>,
    /// Output path chosen by the async save dialog, picked up on the next frame.
    picked_path: Arc<Mutex<Option<PathBuf>>>,
    actor: Actor,
    progress: Arc<Progress>,
}

impl Default for Exporter {
    fn default() -> Self {
        Self {
            open: false,
            width: 1920,
            height: 1080,
            path: None,
            picked_path: Arc::new(Mutex::new(None)),
            actor: Actor::new("animation-export"),
            progress: Arc::new(Progress::default()),
        }
    }
}

impl Exporter {
    pub fn open(&mut self) {
        self.open = true;
    }

    fn is_running(&self) -> bool {
        self.progress.running.load(Ordering::Relaxed)
    }

    /// Draws the popup. Returns a [`PendingExport`] when the user starts an
    /// export, so the caller can build the splats/cameras for it.
    pub fn draw(&mut self, ui: &egui::Ui) -> Option<PendingExport> {
        if !self.open {
            return None;
        }

        // Pick up a path chosen by the (async) save dialog.
        if let Some(path) = self.picked_path.lock().expect("poisoned").take() {
            self.path = Some(path);
        }

        let mut request = None;
        let mut open = self.open;
        egui::Window::new("Export animation")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                let running = self.is_running();

                ui.add_enabled_ui(!running, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        for (name, w, h) in PRESETS {
                            let selected = self.width == w && self.height == h;
                            if ui.selectable_label(selected, name).clicked() {
                                self.width = w;
                                self.height = h;
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Resolution:");
                        ui.add(egui::DragValue::new(&mut self.width).range(16..=7680));
                        ui.label("×");
                        ui.add(egui::DragValue::new(&mut self.height).range(16..=4320));
                    });

                    ui.horizontal(|ui| {
                        if ui.button("Choose file…").clicked() {
                            let slot = self.picked_path.clone();
                            let ctx = ui.ctx().clone();
                            self.actor
                                .run(move || async move {
                                    // A cancelled dialog just leaves the path unset.
                                    if let Ok(path) =
                                        rrfd::pick_save_path("animation.mp4", "MP4 video", &["mp4"])
                                            .await
                                    {
                                        *slot.lock().expect("poisoned") = Some(path);
                                        ctx.request_repaint();
                                    }
                                })
                                .detach();
                        }
                        match &self.path {
                            Some(p) => ui.label(p.display().to_string()),
                            None => ui.label(egui::RichText::new("No file selected").italics()),
                        };
                    });
                });

                ui.separator();

                if running {
                    let done = self.progress.rendered.load(Ordering::Relaxed);
                    let total = self.progress.total.load(Ordering::Relaxed).max(1);
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total as f32)
                            .text(format!("Rendering frame {done} / {total}")),
                    );
                } else {
                    if let Some(err) = self.progress.error.lock().expect("poisoned").as_ref() {
                        ui.colored_label(egui::Color32::LIGHT_RED, err);
                    } else if self.progress.total.load(Ordering::Relaxed) > 0 {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, "Export complete.");
                    }

                    let can_export = self.path.is_some();
                    if ui
                        .add_enabled(can_export, egui::Button::new("Export"))
                        .clicked()
                        && let Some(path) = self.path.clone()
                    {
                        request = Some(PendingExport {
                            width: self.width,
                            height: self.height,
                            path,
                        });
                    }
                }
            });
        self.open = open;

        request
    }

    /// Kicks off the background render + encode for `job`.
    pub fn start(&self, ctx: egui::Context, job: ExportJob) {
        self.progress.rendered.store(0, Ordering::Relaxed);
        self.progress
            .total
            .store(job.cameras.len(), Ordering::Relaxed);
        self.progress.running.store(true, Ordering::Relaxed);
        *self.progress.error.lock().expect("poisoned") = None;

        let progress = self.progress.clone();
        self.actor
            .run(move || async move {
                let result = brush_anim::render_to_mp4(
                    job.splats,
                    job.cameras,
                    job.fps,
                    job.background,
                    job.splat_scale,
                    job.width,
                    job.height,
                    &job.path,
                    |done, _total| {
                        progress.rendered.store(done, Ordering::Relaxed);
                        ctx.request_repaint();
                    },
                )
                .await;

                if let Err(e) = result {
                    *progress.error.lock().expect("poisoned") = Some(format!("{e:#}"));
                }
                progress.running.store(false, Ordering::Relaxed);
                ctx.request_repaint();
            })
            .detach();
    }
}

/// What the popup hands back when the user clicks Export. The caller turns this
/// into a full [`ExportJob`] by adding the splats and per-frame cameras.
pub struct PendingExport {
    pub width: u32,
    pub height: u32,
    pub path: PathBuf,
}
