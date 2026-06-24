//! Export popup + background driver for rendering the animation to an MP4.
//! The actual render/encode lives in `brush_anim`; this owns the UI, the
//! worker thread, and progress reporting. Native only.

use std::path::PathBuf;

use brush_anim::VideoSettings;
use brush_async::Actor;
use brush_render::{camera::Camera, gaussian_splats::Splats};
use egui::Align2;
use tokio::sync::{oneshot, watch};

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
    pub settings: VideoSettings,
    pub path: PathBuf,
}

/// What the background export is doing, published to the UI thread.
#[derive(Clone)]
enum ExportState {
    /// Nothing has been exported yet this session.
    Idle,
    Rendering {
        done: usize,
        total: usize,
    },
    Done,
    Failed(String),
}

pub struct Exporter {
    open: bool,
    width: u32,
    height: u32,
    path: Option<PathBuf>,
    /// A save dialog in flight; resolves to the chosen path, or closes without
    /// a value if the user cancelled.
    picked_path: Option<oneshot::Receiver<PathBuf>>,
    actor: Actor,
    /// Progress published by the worker. Each export installs a fresh channel;
    /// the initial one has no sender and just reads back `Idle`.
    state: watch::Receiver<ExportState>,
}

impl Default for Exporter {
    fn default() -> Self {
        let (_, state) = watch::channel(ExportState::Idle);
        Self {
            open: false,
            width: 1920,
            height: 1080,
            path: None,
            picked_path: None,
            actor: Actor::new("animation-export"),
            state,
        }
    }
}

impl Exporter {
    pub fn open(&mut self) {
        self.open = true;
    }

    /// Draws the popup. Returns a [`PendingExport`] when the user starts an
    /// export, so the caller can build the splats/cameras for it.
    pub fn draw(&mut self, ui: &egui::Ui) -> Option<PendingExport> {
        if !self.open {
            return None;
        }

        // Pick up a path chosen by the (async) save dialog.
        if let Some(rx) = &mut self.picked_path {
            match rx.try_recv() {
                Ok(path) => {
                    self.path = Some(path);
                    self.picked_path = None;
                }
                // Cancelled: the sender dropped without ever sending.
                Err(oneshot::error::TryRecvError::Closed) => self.picked_path = None,
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
        }

        let mut request = None;
        let mut open = self.open;
        egui::Window::new("Export animation")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut open)
            .show(ui.ctx(), |ui| {
                let state = self.state.borrow().clone();

                ui.add_enabled_ui(!matches!(state, ExportState::Rendering { .. }), |ui| {
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
                            let (tx, rx) = oneshot::channel();
                            self.picked_path = Some(rx);
                            let ctx = ui.ctx().clone();
                            self.actor
                                .run(move || async move {
                                    // A cancelled dialog just drops `tx`.
                                    if let Ok(path) =
                                        rrfd::pick_save_path("animation.mp4", "MP4 video", &["mp4"])
                                            .await
                                    {
                                        let _ = tx.send(path);
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

                if let ExportState::Rendering { done, total } = state {
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                            .text(format!("Rendering frame {done} / {total}")),
                    );
                } else {
                    match state {
                        ExportState::Failed(err) => {
                            ui.colored_label(egui::Color32::LIGHT_RED, err);
                        }
                        ExportState::Done => {
                            ui.colored_label(egui::Color32::LIGHT_GREEN, "Export complete.");
                        }
                        ExportState::Idle | ExportState::Rendering { .. } => {}
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
    pub fn start(&mut self, ctx: egui::Context, job: ExportJob) {
        let total = job.cameras.len();
        let (state, rx) = watch::channel(ExportState::Rendering { done: 0, total });
        self.state = rx;

        self.actor
            .run(move || async move {
                let result = brush_anim::render_to_mp4(
                    job.splats,
                    &job.cameras,
                    &job.settings,
                    &job.path,
                    |done, total| {
                        let _ = state.send(ExportState::Rendering { done, total });
                        ctx.request_repaint();
                    },
                )
                .await;

                let _ = state.send(match result {
                    Ok(()) => ExportState::Done,
                    Err(e) => ExportState::Failed(format!("{e:#}")),
                });
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
