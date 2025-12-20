use crate::{UiMode, app::CameraSettings, camera_controls::CameraController};
use anyhow::Result;
use brush_process::{config::TrainStreamConfig, message::ProcessMessage, process::create_process};
use brush_render::camera::Camera;
use brush_vfs::DataSource;
use burn_cubecl::cubecl::config::{GlobalConfig, streaming::StreamingConfig};
use burn_wgpu::WgpuDevice;
use egui::{Response, TextureHandle};
use glam::{Affine3A, Quat, Vec3};
use std::sync::RwLock;
use tokio::sync::{self, oneshot::Receiver};
use tokio_stream::StreamExt;
use tokio_with_wasm::alias as tokio_wasm;

#[derive(Debug, Clone)]
enum ControlMessage {
    Paused(bool),
}

#[derive(Debug, Clone)]
struct DeviceContext {
    pub device: WgpuDevice,
    pub ctx: egui::Context,
}

struct RunningProcess {
    messages: sync::mpsc::Receiver<Result<ProcessMessage, anyhow::Error>>,
    control: sync::mpsc::UnboundedSender<ControlMessage>,
    send_device: Option<sync::oneshot::Sender<DeviceContext>>,
}

/// A thread-safe wrapper around the UI process.
/// This allows the UI process to be accessed from multiple threads.
///
/// Mixing a sync lock and async code is asking for trouble, but there's no other good way in egui currently.
/// The "precondition" to avoid deadlocks, is to only holds locks _within the trait functions_. As long as you don't ever hold them
/// over an await point, things shouldn't be able to deadlock.
pub struct UiProcess(RwLock<UiProcessInner>);

impl Default for UiProcess {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BackgroundStyle {
    Black,
    Checkerboard,
}

impl UiProcess {
    pub fn new() -> Self {
        // Set the global config once on startup.
        GlobalConfig::set(GlobalConfig {
            streaming: StreamingConfig {
                max_streams: 1,
                ..Default::default()
            },
            ..Default::default()
        });

        Self(RwLock::new(UiProcessInner::new()))
    }

    pub(crate) fn background_style(&self) -> BackgroundStyle {
        self.read().background_style
    }

    #[allow(unused)]
    pub(crate) fn set_background_style(&self, style: BackgroundStyle) {
        self.write().background_style = style;
    }
}

impl UiProcess {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, UiProcessInner> {
        self.0.read().expect("RwLock poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, UiProcessInner> {
        self.0.write().expect("RwLock poisoned")
    }
}

pub struct TexHandle {
    pub handle: TextureHandle,
    pub has_alpha: bool,
    pub blurred_bg: Option<TextureHandle>,
}

impl UiProcess {
    pub fn is_loading(&self) -> bool {
        self.read().is_loading
    }

    pub fn is_training(&self) -> bool {
        self.read().is_training
    }

    pub fn tick_controls(&self, response: &Response, ui: &egui::Ui) {
        self.write().controls.tick(response, ui);
    }

    pub fn model_local_to_world(&self) -> glam::Affine3A {
        self.read().controls.model_local_to_world
    }

    pub fn current_camera(&self) -> Camera {
        let inner = self.read();
        // Keep controls & camera position in sync.
        let mut cam = inner.camera.clone();
        cam.position = inner.controls.position;
        cam.rotation = inner.controls.rotation;
        cam
    }

    pub fn set_train_paused(&self, paused: bool) {
        self.write().train_paused = paused;
        if let Some(process) = self.read().running_process.as_ref() {
            let _ = process.control.send(ControlMessage::Paused(paused));
        }
    }

    pub fn is_train_paused(&self) -> bool {
        self.read().train_paused
    }

    pub fn get_cam_settings(&self) -> CameraSettings {
        self.read().controls.settings.clone()
    }

    pub fn get_grid_opacity(&self) -> f32 {
        let inner = self.read();
        if inner.controls.settings.grid_enabled.is_some_and(|g| g) {
            1.0 // Grid fully visible when enabled
        } else {
            inner.controls.get_grid_opacity() // Use fade timer when disabled
        }
    }

    pub fn set_cam_settings(&self, settings: &CameraSettings) {
        let mut inner = self.write();
        inner.controls.settings = settings.clone();
        inner.splat_scale = settings.splat_scale;
    }

    pub fn set_cam_transform(&self, position: Vec3, rotation: Quat) {
        self.write().set_camera_transform(position, rotation);
        self.read().repaint();
    }

    pub fn set_focal_point(&self, focal_point: Vec3, focus_distance: f32, rotation: Quat) {
        self.write()
            .set_focal_point(focal_point, focus_distance, rotation);
        self.read().repaint();
    }

    pub fn set_cam_fov(&self, fov_y: f64) {
        self.write().camera.fov_y = fov_y;
        self.read().repaint();
    }

    pub fn focus_view(&self, cam: &Camera) {
        // Also focus this view.
        let mut inner = self.write();
        inner.camera = cam.clone();
        inner.controls.stop_movement();

        // We want to set the view matrix such that MV == view view matrix.
        // new_view_mat * model_mat == view_view_mat
        // new_view_mat = view_view_mat * model_mat.inverse()
        let new_view_mat = cam.world_to_local() * inner.controls.model_local_to_world.inverse();

        let view_local_to_world = new_view_mat.inverse();
        let (_, rot, translate) = view_local_to_world.to_scale_rotation_translation();
        inner.controls.position = translate;
        inner.controls.rotation = rot;
        inner.repaint();
    }

    pub fn set_model_up(&self, up_axis: Vec3) {
        let mut inner = self.write();
        inner.controls.model_local_to_world = Affine3A::from_rotation_translation(
            Quat::from_rotation_arc(Vec3::NEG_Y, up_axis.normalize()),
            Vec3::ZERO,
        );
        inner.repaint();
    }

    pub fn connect_device(&self, device: WgpuDevice, ctx: egui::Context) {
        let mut inner = self.write();
        let ctx = DeviceContext { device, ctx };
        inner.cur_device_ctx = Some(ctx.clone());
        if let Some(process) = &mut inner.running_process
            && let Some(send) = process.send_device.take()
        {
            send.send(ctx).expect("Failed to send device");
        }
    }

    pub fn start_new_process(&self, source: DataSource, args: Receiver<TrainStreamConfig>) {
        let mut inner = self.write();
        let mut reset = UiProcessInner::new();
        reset.cur_device_ctx = inner.cur_device_ctx.clone();
        *inner = reset;

        let (sender, receiver) = sync::mpsc::channel(1);
        let (train_sender, mut train_receiver) = sync::mpsc::unbounded_channel();
        let (send_dev, rec_rev) = sync::oneshot::channel::<DeviceContext>();

        tokio_with_wasm::alias::task::spawn(async move {
            // Wait for device & gui ctx to be available.
            let Ok(device_ctx) = rec_rev.await else {
                // Closed before we could start the process
                return;
            };

            let stream = create_process(source, args, device_ctx.device);
            let mut stream = std::pin::pin!(stream);

            while let Some(msg) = stream.next().await {
                // Mark egui as needing a repaint.
                device_ctx.ctx.request_repaint();

                // // Stop the process if noone is listening anymore.
                if sender.send(msg).await.is_err() {
                    break;
                }

                // Check if training is paused. Don't care about other messages as pausing loading
                // doesn't make much sense.
                if matches!(train_receiver.try_recv(), Ok(ControlMessage::Paused(true))) {
                    // Pause if needed.
                    while !matches!(
                        train_receiver.recv().await,
                        Some(ControlMessage::Paused(false))
                    ) {}
                }
                // Give back control to the runtime.
                // This only really matters in the browser:
                // on native, receiving also yields. In the browser that doesn't yield
                // back control fully though whereas yield_now() does.
                tokio_wasm::task::yield_now().await;
            }
        });

        if let Some(ctx) = &inner.cur_device_ctx {
            send_dev
                .send(ctx.clone())
                .expect("Failed to send device context");
            inner.running_process = Some(RunningProcess {
                messages: receiver,
                control: train_sender,
                send_device: None,
            });
        } else {
            inner.running_process = Some(RunningProcess {
                messages: receiver,
                control: train_sender,
                send_device: Some(send_dev),
            });
        }
    }

    pub fn message_queue(&self) -> Vec<Result<ProcessMessage>> {
        let mut ret = vec![];
        let mut inner = self.write();
        if let Some(process) = inner.running_process.as_mut() {
            while let Ok(msg) = process.messages.try_recv() {
                ret.push(msg);
            }
        }

        for msg in &ret {
            // Keep track of things the ui process needs.
            match msg {
                Ok(ProcessMessage::StartLoading { training }) => {
                    inner.is_training = *training;
                    inner.is_loading = true;
                }
                Ok(ProcessMessage::DoneLoading) => {
                    inner.is_loading = false;
                }
                Err(_) => {
                    inner.is_loading = false;
                    inner.is_training = false;
                }
                _ => (),
            }
        }
        drop(inner);
        ret
    }

    pub fn ui_mode(&self) -> UiMode {
        self.read().ui_mode
    }

    pub fn set_ui_mode(&self, mode: UiMode) {
        self.write().ui_mode = mode;
    }

    pub fn request_reset_layout(&self) {
        self.write().reset_layout_requested = true;
    }

    pub fn take_reset_layout_request(&self) -> bool {
        let mut inner = self.write();
        let requested = inner.reset_layout_requested;
        inner.reset_layout_requested = false;
        requested
    }

    pub fn reset_session(&self) {
        let mut inner = self.write();
        let device_ctx = inner.cur_device_ctx.clone();
        *inner = UiProcessInner::new();
        inner.cur_device_ctx = device_ctx;
        inner.session_reset_requested = true;
    }

    pub fn take_session_reset_request(&self) -> bool {
        let mut inner = self.write();
        let requested = inner.session_reset_requested;
        inner.session_reset_requested = false;
        requested
    }
}

struct UiProcessInner {
    is_loading: bool,
    is_training: bool,
    camera: Camera,
    splat_scale: Option<f32>,
    controls: CameraController,
    running_process: Option<RunningProcess>,
    cur_device_ctx: Option<DeviceContext>,
    ui_mode: UiMode,
    background_style: BackgroundStyle,
    train_paused: bool,
    reset_layout_requested: bool,
    session_reset_requested: bool,
}

impl UiProcessInner {
    pub fn new() -> Self {
        let position = -Vec3::Z * 2.5;
        let rotation = Quat::IDENTITY;

        let controls = CameraController::new(position, rotation, CameraSettings::default());
        let camera = Camera::new(Vec3::ZERO, Quat::IDENTITY, 0.8, 0.8, glam::vec2(0.5, 0.5));

        Self {
            camera,
            controls,
            splat_scale: None,
            is_loading: false,
            is_training: false,
            running_process: None,
            cur_device_ctx: None,
            ui_mode: UiMode::Default,
            background_style: BackgroundStyle::Black,
            train_paused: false,
            reset_layout_requested: false,
            session_reset_requested: false,
        }
    }

    fn repaint(&self) {
        if let Some(ctx) = &self.cur_device_ctx {
            ctx.ctx.request_repaint();
        }
    }

    fn set_camera_transform(&mut self, position: Vec3, rotation: Quat) {
        self.controls.position = position;
        self.controls.rotation = rotation;
        self.camera.position = position;
        self.camera.rotation = rotation;
    }

    fn set_focal_point(&mut self, focal_point: Vec3, focus_distance: f32, rotation: Quat) {
        let position = focal_point - rotation * Vec3::Z * focus_distance;
        self.set_camera_transform(position, rotation);
        self.controls.focus_distance = focus_distance;
    }
}
