use anyhow::Result;
use brush_dataset::{Dataset, scene::SceneView};
use brush_process::{config::ProcessArgs, message::ProcessMessage, process::process_stream};
use brush_render::camera::Camera;
use brush_ui::{BrushUiProcess, app::CameraSettings, camera_controls::CameraController};
use brush_vfs::DataSource;
use burn_wgpu::WgpuDevice;
use egui::Response;
use glam::{Affine3A, Quat, Vec3};
use parking_lot::RwLock;
use tokio::sync;
use tokio_stream::StreamExt;
use tokio_with_wasm::alias as tokio_wasm;

#[derive(Debug, Clone)]
enum ControlMessage {
    Paused(bool),
}

#[derive(Debug, Clone)]
struct DeviceContext {
    device: WgpuDevice,
    ctx: egui::Context,
}

struct RunningProcess {
    messages: sync::mpsc::Receiver<Result<ProcessMessage, anyhow::Error>>,
    control: sync::mpsc::UnboundedSender<ControlMessage>,
    send_device: Option<sync::oneshot::Sender<DeviceContext>>,
}

/// A thread-safe wrapper around the UI process.
/// This allows the UI process to be accessed from multiple threads.
pub struct UiProcess {
    inner: RwLock<UiProcessInner>,
}

impl UiProcess {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(UiProcessInner::new()),
        }
    }
}

impl BrushUiProcess for UiProcess {
    fn is_loading(&self) -> bool {
        self.inner.read().is_loading
    }

    fn is_training(&self) -> bool {
        self.inner.read().is_training
    }

    fn tick_controls(&self, response: &Response, ui: &egui::Ui) {
        self.inner.write().controls.tick(response, ui);
    }

    fn model_local_to_world(&self) -> glam::Affine3A {
        self.inner.read().model_local_to_world
    }

    fn current_camera(&self) -> Camera {
        self.inner.read().camera.clone()
    }

    fn selected_view(&self) -> Option<SceneView> {
        self.inner.read().selected_view.clone()
    }

    fn set_train_paused(&self, paused: bool) {
        self.inner.write().set_train_paused(paused);
    }

    fn set_camera(&self, cam: Camera) {
        self.inner.write().set_camera(cam);
    }

    fn set_cam_settings(&self, settings: CameraSettings) {
        self.inner.write().set_cam_settings(settings);
    }

    fn focus_view(&self, view: &SceneView) {
        self.inner.write().focus_view(view);
    }

    fn set_model_up(&self, up_axis: Vec3) {
        self.inner.write().set_model_up(up_axis);
    }

    fn connect_device(&self, device: WgpuDevice, ctx: egui::Context) {
        self.inner.write().connect_device(device, ctx);
    }

    fn start_new_process(&self, source: DataSource, args: ProcessArgs) {
        self.inner.write().start_new_process(source, args);
    }

    fn try_recv_message(&self) -> Option<Result<ProcessMessage>> {
        self.inner.write().try_recv_message()
    }
}

struct UiProcessInner {
    dataset: Dataset,
    is_loading: bool,
    is_training: bool,
    camera: Camera,
    view_aspect: Option<f32>,
    controls: CameraController,
    model_local_to_world: Affine3A,
    running_process: Option<RunningProcess>,
    cam_settings: CameraSettings,
    selected_view: Option<SceneView>,
    cur_device_ctx: Option<DeviceContext>,
}

impl UiProcessInner {
    pub fn new() -> Self {
        let model_transform = Affine3A::IDENTITY;
        let cam_settings = CameraSettings::default();
        let controls = CameraController::new(
            cam_settings.position,
            cam_settings.rotation,
            cam_settings.focus_distance,
            cam_settings.speed_scale,
            cam_settings.clamping.clone(),
        );

        // Camera position will be controlled by the orbit controls.
        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            cam_settings.focal,
            cam_settings.focal,
            glam::vec2(0.5, 0.5),
        );

        Self {
            camera,
            controls,
            model_local_to_world: model_transform,
            view_aspect: None,
            dataset: Dataset::empty(),
            is_loading: false,
            is_training: false,
            selected_view: None,
            running_process: None,
            cam_settings,
            cur_device_ctx: None,
        }
    }

    fn match_controls_to(&mut self, cam: &Camera) {
        // We want model * controls.transform() == view_cam.transform() ->
        //  controls.transform = model.inverse() * view_cam.transform.
        let transform = self.model_local_to_world.inverse() * cam.local_to_world();
        self.controls.position = transform.translation.into();
        self.controls.rotation = Quat::from_mat3a(&transform.matrix3);
    }

    fn set_train_paused(&self, paused: bool) {
        if let Some(process) = self.running_process.as_ref() {
            let _ = process.control.send(ControlMessage::Paused(paused));
        }
    }

    fn set_camera(&mut self, cam: Camera) {
        self.match_controls_to(&cam);
        self.camera = cam;
        self.controls.stop_movement();
    }

    fn set_cam_settings(&mut self, settings: CameraSettings) {
        self.controls = CameraController::new(
            settings.position,
            settings.rotation,
            settings.focus_distance,
            settings.speed_scale,
            settings.clamping.clone(),
        );
        self.cam_settings = settings;
        let cam = self.camera.clone();
        self.match_controls_to(&cam);
    }

    fn focus_view(&mut self, view: &SceneView) {
        self.set_camera(view.camera.clone());
        self.view_aspect = Some(view.image.width() as f32 / view.image.height() as f32);
        if let Some(extent) = self.dataset.train.estimate_extent() {
            self.controls.focus_distance = extent / 3.0;
        } else {
            self.controls.focus_distance = self.cam_settings.focus_distance;
        }
        self.controls.focus_distance = self.cam_settings.focus_distance;
    }

    fn set_model_up(&mut self, up_axis: Vec3) {
        self.model_local_to_world = Affine3A::from_rotation_translation(
            Quat::from_rotation_arc(up_axis.normalize(), Vec3::NEG_Y),
            Vec3::ZERO,
        );

        let cam = self.camera.clone();
        self.match_controls_to(&cam);
    }

    fn connect_device(&mut self, device: WgpuDevice, ctx: egui::Context) {
        let ctx = DeviceContext { device, ctx };
        self.cur_device_ctx = Some(ctx.clone());

        log::info!("Connecting to device ctx");

        #[cfg(feature = "tracing")]
        {
            // TODO: In debug only?
            #[cfg(target_family = "wasm")]
            {
                use tracing_subscriber::layer::SubscriberExt;
                tracing::subscriber::set_global_default(
                    tracing_subscriber::registry()
                        .with(tracing_wasm::WASMLayer::new(Default::default())),
                )
                .expect("Failed to set tracing subscriber");
            }

            #[cfg(all(feature = "tracy", not(target_family = "wasm")))]
            {
                use tracing_subscriber::layer::SubscriberExt;

                tracing::subscriber::set_global_default(
                    tracing_subscriber::registry()
                        .with(tracing_tracy::TracyLayer::default())
                        .with(sync_span::SyncLayer::<
                            burn_cubecl::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>,
                        >::new(device.clone())),
                )
                .expect("Failed to set tracing subscriber");
            }
        }

        if let Some(process) = &mut self.running_process {
            if let Some(send) = process.send_device.take() {
                send.send(ctx).expect("Failed to send device");
            }
        }
    }

    fn start_new_process(&mut self, source: DataSource, args: ProcessArgs) {
        let mut reset = Self::new();
        reset.cur_device_ctx = self.cur_device_ctx.clone();
        *self = reset;

        let (sender, receiver) = sync::mpsc::channel(1);
        let (train_sender, mut train_receiver) = sync::mpsc::unbounded_channel();
        let (send_dev, rec_rev) = sync::oneshot::channel::<DeviceContext>();

        tokio_with_wasm::alias::task::spawn(async move {
            // Wait for device & gui ctx to be available.
            let Ok(device_ctx) = rec_rev.await else {
                // Closed before we could start the process
                return;
            };

            let stream = process_stream(source, args, device_ctx.device);
            let mut stream = std::pin::pin!(stream);

            while let Some(msg) = stream.next().await {
                // Mark egui as needing a repaint.
                device_ctx.ctx.request_repaint();

                let is_train_step = matches!(msg, Ok(ProcessMessage::TrainStep { .. }));

                // Stop the process if noone is listening anymore.
                if sender.send(msg).await.is_err() {
                    break;
                }

                // Check if training is paused. Don't care about other messages as pausing loading
                // doesn't make much sense.
                if is_train_step
                    && matches!(train_receiver.try_recv(), Ok(ControlMessage::Paused(true)))
                {
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
                if cfg!(target_family = "wasm") {
                    tokio_wasm::task::yield_now().await;
                }
            }
        });

        if let Some(ctx) = &self.cur_device_ctx {
            send_dev
                .send(ctx.clone())
                .expect("Failed to send device context");
            self.running_process = Some(RunningProcess {
                messages: receiver,
                control: train_sender,
                send_device: None,
            });
        } else {
            self.running_process = Some(RunningProcess {
                messages: receiver,
                control: train_sender,
                send_device: Some(send_dev),
            });
        }
    }

    fn try_recv_message(&mut self) -> Option<Result<ProcessMessage>> {
        if let Some(process) = self.running_process.as_mut() {
            // If none, just return none.
            let msg = process.messages.try_recv().ok()?;

            // Keep track of things the ui process needs.
            match msg.as_ref() {
                Ok(ProcessMessage::Dataset { dataset }) => {
                    self.selected_view = dataset.train.views.last().cloned();
                }
                Ok(ProcessMessage::StartLoading { training }) => {
                    self.is_training = *training;
                    self.is_loading = true;
                }
                Ok(ProcessMessage::DoneLoading) => {
                    self.is_loading = false;
                }
                _ => (),
            }
            // Forward msg.
            Some(msg)
        } else {
            None
        }
    }
}
