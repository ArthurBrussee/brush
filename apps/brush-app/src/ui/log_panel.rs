use std::collections::VecDeque;
use std::sync::Mutex;
use web_time::{SystemTime, UNIX_EPOCH};

use egui::{Color32, CornerRadius, Frame, Margin, RichText, ScrollArea, Stroke};
use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::ui::panels::AppPane;
use crate::ui::ui_process::UiProcess;

const MAX_ENTRIES: usize = 1024;

#[derive(Clone)]
struct LogEntry {
    timestamp: u64, // UNIX seconds at log time
    level: Level,
    message: String,
}

static LOG_BUFFER: Mutex<VecDeque<LogEntry>> = Mutex::new(VecDeque::new());

/// A logger that pushes records into the in-memory ring buffer used by
/// [`LogPanel`] and forwards them to an underlying logger (`env_logger`,
/// `wasm_logger`, or whatever the platform installs).
struct ChainedLogger {
    inner: Box<dyn Log>,
}

impl Log for ChainedLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &Record<'_>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let mut buf = LOG_BUFFER.lock().expect("LOG_BUFFER poisoned");
        if buf.len() >= MAX_ENTRIES {
            buf.pop_front();
        }
        buf.push_back(LogEntry {
            timestamp,
            level: record.level(),
            message: format!("{}", record.args()),
        });
        drop(buf);
        self.inner.log(record);
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

/// Install a logger that captures records for [`LogPanel`] in addition to
/// forwarding them to `inner`.
pub fn install_global_logger(inner: Box<dyn Log>, max_level: LevelFilter) {
    log::set_max_level(max_level);
    let _ = log::set_boxed_logger(Box::new(ChainedLogger { inner }));
}

/// On wasm we don't have access to the private `wasm_logger::WasmLogger`,
/// so this is a small inline replacement that writes to `console.log`.
#[cfg(target_family = "wasm")]
pub struct ConsoleLogger;

#[cfg(target_family = "wasm")]
impl Log for ConsoleLogger {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &Record<'_>) {
        let line = format!("[{}] {}", record.level(), record.args());
        web_sys::console::log_1(&line.into());
    }
    fn flush(&self) {}
}

fn format_clock(unix_secs: u64) -> String {
    // UTC HH:MM:SS — local time would need an OS-specific tz lookup that
    // pulls in chrono/time. For a log panel UTC is fine.
    let s = unix_secs % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

fn level_tag(level: Level) -> (&'static str, Color32) {
    match level {
        Level::Error => ("ERROR", Color32::from_rgb(232, 110, 110)),
        Level::Warn => ("WARN ", Color32::from_rgb(232, 184, 96)),
        Level::Info => ("INFO ", Color32::from_rgb(122, 178, 232)),
        Level::Debug => ("DEBUG", Color32::from_rgb(150, 165, 185)),
        Level::Trace => ("TRACE", Color32::from_rgb(120, 130, 145)),
    }
}

#[derive(Default)]
pub(crate) struct LogPanel;

impl AppPane for LogPanel {
    fn title(&self) -> egui::WidgetText {
        "Log".into()
    }

    fn is_visible(&self, process: &UiProcess) -> bool {
        // Hidden until something interesting is happening — keeps the default
        // landing layout focused on Stats / Settings.
        process.is_loading() || process.is_training()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _process: &UiProcess) {
        // Keep the lock short; clone-out, render outside the guard.
        let entries: Vec<LogEntry> = {
            let buf = LOG_BUFFER.lock().expect("LOG_BUFFER poisoned");
            buf.iter().cloned().collect()
        };

        let mono = egui::FontId::monospace(11.5);
        let timestamp_color = Color32::from_rgb(110, 120, 135);
        let message_color = Color32::from_rgb(212, 218, 226);

        Frame::new()
            .fill(Color32::from_rgb(14, 16, 22))
            .stroke(Stroke::new(1.0, Color32::from_rgb(36, 40, 50)))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.style_mut().spacing.item_spacing.y = 1.0;
                ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in entries {
                            let (tag, tag_color) = level_tag(entry.level);
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 6.0;
                                ui.label(
                                    RichText::new(format_clock(entry.timestamp))
                                        .font(mono.clone())
                                        .color(timestamp_color),
                                );
                                ui.label(
                                    RichText::new(tag)
                                        .font(mono.clone())
                                        .color(tag_color)
                                        .strong(),
                                );
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(&entry.message)
                                            .font(mono.clone())
                                            .color(message_color),
                                    )
                                    .selectable(true)
                                    .wrap(),
                                );
                            });
                        }
                    });
            });
    }

    fn inner_margin(&self) -> f32 {
        6.0
    }
}
