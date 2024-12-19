use std::sync::Arc;

use brush_process::{process_loop::ProcessMessage, rerun_tools::VisualizeTools};

use crate::app::{AppContext, AppPanel};

pub(crate) struct RerunPanel {
    visualize: Option<Arc<VisualizeTools>>,
    eval_view_count: Option<usize>,
    log_train_stats_every: u32,
    visualize_splats_every: Option<u32>,
    ready_to_log_dataset: bool,
}

impl RerunPanel {
    pub(crate) fn new() -> Self {
        Self {
            visualize: None,
            eval_view_count: None,
            log_train_stats_every: 50,
            visualize_splats_every: None,
            ready_to_log_dataset: false,
        }
    }
}

impl RerunPanel {}

impl AppPanel for RerunPanel {
    fn title(&self) -> String {
        "Rerun".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, _context: &mut AppContext) {
        match message {
            ProcessMessage::StartLoading { training } => {
                if *training {
                    if self.visualize.is_some() {
                        self.visualize = Some(Arc::new(VisualizeTools::new()));
                    }
                } else {
                    self.visualize = None;
                }
            }
            ProcessMessage::DoneLoading { training } => {
                if *training {
                    self.ready_to_log_dataset = true;
                }
            }
            ProcessMessage::TrainStep {
                splats,
                stats,
                iter,
                timestamp: _,
            } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };
                if let Some(every) = self.visualize_splats_every {
                    if iter % every == 0 {
                        visualize.clone().log_splats(*splats.clone());
                    }
                }

                visualize.log_splat_stats(splats);

                // Log out train stats.
                if iter % self.log_train_stats_every == 0 {
                    visualize.log_train_stats(*iter, *stats.clone());
                }
            }
            ProcessMessage::RefineStep { stats, iter } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                visualize.log_refine_stats(*iter, stats);
            }
            ProcessMessage::EvalResult { iter, eval } => {
                let Some(visualize) = self.visualize.clone() else {
                    return;
                };

                visualize.log_eval_stats(*iter, eval.clone());
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut AppContext) {
        let Some(visualize) = self.visualize.clone() else {
            if ui.button("Enable rerun").clicked() {
                self.visualize = Some(Arc::new(VisualizeTools::new()));
            }

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.label("Stream data to ");
                ui.hyperlink_to("Rerun.io", "https://rerun.io");
                ui.label(" for visualization");
            });

            ui.label("Install the viewer to get started.");
            ui.label("Will open the viewer if it isn't open yet. Open the viewer before enabling rerun to keep data.");
            ui.label("Open the brush_blueprint.rbl in the rerun viewer for a good default layout.");
            return;
        };

        if self.ready_to_log_dataset {
            visualize.log_scene(context.dataset.train.clone());
            self.ready_to_log_dataset = false;
        }

        ui.horizontal(|ui| {
            ui.label("Log train stats");
            ui.add(
                egui::Slider::new(&mut self.log_train_stats_every, 1..=1000)
                    .prefix("every ")
                    .suffix(" steps"),
            );
        });

        let mut limit_eval_views = self.eval_view_count.is_some();
        ui.checkbox(&mut limit_eval_views, "Limit eval views");
        if limit_eval_views != self.eval_view_count.is_some() {
            self.eval_view_count = if limit_eval_views { Some(4) } else { None };
        }

        if let Some(count) = self.eval_view_count.as_mut() {
            ui.add(egui::Slider::new(count, 1..=100).text("Eval view count"));
        }

        let mut visualize_splats = self.visualize_splats_every.is_some();
        ui.checkbox(&mut visualize_splats, "Visualize splats");
        if visualize_splats != self.visualize_splats_every.is_some() {
            self.visualize_splats_every = if visualize_splats { Some(500) } else { None };
        }

        if let Some(every) = self.visualize_splats_every.as_mut() {
            ui.add(egui::Slider::new(every, 1..=5000).text("Visualize splats every"));
        }
    }
}
