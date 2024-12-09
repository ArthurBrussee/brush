use core::f32;
use std::ops::Range;
use std::path::Path;
use std::{pin::Pin, sync::Arc};

use async_fn_stream::try_fn_stream;

use brush_dataset::brush_vfs::{BrushVfs, PathReader};
use brush_dataset::{self, splat_import, Dataset, LoadDatasetArgs, LoadInitArgs};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_train::train::{RefineStats, TrainStepStats};
use brush_train::{eval::EvalStats, train::TrainConfig};
use burn::backend::Autodiff;
use burn_wgpu::{Wgpu, WgpuDevice};
use eframe::egui;
use egui_tiles::{Container, Tile, TileId, Tiles};
use glam::{Affine3A, Quat, Vec3, Vec3A};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::bytes::Bytes;
use tokio_util::io::StreamReader;
use tokio_with_wasm::alias as tokio;

use ::tokio::io::AsyncReadExt;
use ::tokio::sync::mpsc::error::TrySendError;
use ::tokio::sync::mpsc::{Receiver, Sender, UnboundedReceiver};
use ::tokio::{io::AsyncRead, io::BufReader, sync::mpsc::channel};
use std::collections::HashMap;
use tokio::task;
use wgpu::Features;

use tokio_stream::{Stream, StreamExt};
use web_time::Instant;

type Backend = Wgpu;

use crate::{
    orbit_controls::OrbitControls,
    panels::{DatasetPanel, LoadDataPanel, PresetsPanel, ScenePanel, StatsPanel, TracingPanel},
    train_loop::{self, TrainMessage},
    PaneType, ViewerTree,
};

struct TrainStats {
    loss: f32,
    train_image_index: usize,
}

fn parse_search(search: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    let search = search.trim_start_matches('?');
    for pair in search.split('&') {
        // Split each pair on '=' to separate key and value
        if let Some((key, value)) = pair.split_once('=') {
            // URL decode the key and value and insert into HashMap
            params.insert(
                urlencoding::decode(key).unwrap_or_default().into_owned(),
                urlencoding::decode(value).unwrap_or_default().into_owned(),
            );
        }
    }
    params
}

pub enum UiControlMessage {
    LoadData(String),
}

#[derive(Clone)]
pub(crate) enum ProcessMessage {
    NewSource,
    StartLoading {
        training: bool,
    },
    /// Some process errored out, and want to display this error
    /// to the user.
    Error(Arc<anyhow::Error>),
    /// Loaded a splat from a ply file.
    ///
    /// Nb: This includes all the intermediately loaded splats.
    /// Nb: Animated splats will have the 'frame' number set.
    ViewSplats {
        up_axis: Vec3,
        splats: Box<Splats<Backend>>,
        frame: usize,
        total_frames: usize,
    },
    /// Loaded a bunch of viewpoints to train on.
    Dataset {
        data: Dataset,
    },
    /// Splat, or dataset and initial splat, are done loading.
    DoneLoading {
        training: bool,
    },
    /// Some number of training steps are done.
    TrainStep {
        splats: Box<Splats<Backend>>,
        stats: Box<TrainStepStats<Autodiff<Backend>>>,
        iter: u32,
        timestamp: Instant,
    },
    /// Some number of training steps are done.
    RefineStep {
        stats: Box<RefineStats>,
        iter: u32,
    },
    /// Eval was run successfully with these results.
    EvalResult {
        iter: u32,
        eval: EvalStats<Backend>,
    },
}

pub struct Viewer {
    tree: egui_tiles::Tree<PaneType>,
    datasets: Option<TileId>,
    tree_ctx: ViewerTree,
}

// TODO: Bit too much random shared state here.
pub(crate) struct ViewerContext {
    pub dataset: Dataset,
    pub camera: Camera,
    pub controls: OrbitControls,

    pub model_transform: Affine3A,

    device: WgpuDevice,
    ctx: egui::Context,

    rec_ui_control_msg: UnboundedReceiver<UiControlMessage>,

    send_train_msg: Option<Sender<TrainMessage>>,
    rec_process_msg: Option<Receiver<ProcessMessage>>,
}

fn process_loop(
    source: DataSource,
    device: WgpuDevice,
    train_receiver: Receiver<TrainMessage>,
    load_data_args: LoadDatasetArgs,
    load_init_args: LoadInitArgs,
    train_config: TrainConfig,
) -> Pin<Box<impl Stream<Item = anyhow::Result<ProcessMessage>>>> {
    let stream = try_fn_stream(|emitter| async move {
        let _ = emitter.emit(ProcessMessage::NewSource).await;

        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let data = source.into_reader()?;
        let mut data = BufReader::new(data);
        let mut peek = [0; 128];
        data.read_exact(&mut peek).await?;
        let reader = std::io::Cursor::new(peek).chain(data);

        let mut vfs = if peek.starts_with("ply".as_bytes()) {
            let mut path_reader = PathReader::default();
            path_reader.add(Path::new("input.ply"), reader);
            BrushVfs::from_paths(path_reader)
        } else if peek.starts_with("PK".as_bytes()) {
            BrushVfs::from_zip_reader(reader).await?
        } else if peek.starts_with("<!DOCTYPE html>".as_bytes()) {
            anyhow::bail!("Failed to download data (are you trying to download from Google Drive? You might have to use the proxy.")
        } else {
            anyhow::bail!("only zip and ply files are supported.");
        };

        let names: Vec<_> = vfs.file_names().map(|x| x.to_path_buf()).collect();
        log::info!("Mounted VFS with {} files", names.len());

        if names.len() == 1 && names[0].extension().is_some_and(|p| p == "ply") {
            log::info!("Loading single ply file");

            let _ = emitter
                .emit(ProcessMessage::StartLoading { training: false })
                .await;

            let sub_sample = None; // Subsampling a trained ply doesn't really make sense.
            let splat_stream = splat_import::load_splat_from_ply(
                vfs.open_path(&names[0]).await?,
                sub_sample,
                device.clone(),
            );

            let mut splat_stream = std::pin::pin!(splat_stream);

            while let Some(message) = splat_stream.next().await {
                let message = message?;
                emitter
                    .emit(ProcessMessage::ViewSplats {
                        up_axis: message.meta.up_axis,
                        splats: Box::new(message.splats),
                        frame: message.meta.current_frame,
                        total_frames: message.meta.frame_count,
                    })
                    .await;
            }

            emitter
                .emit(ProcessMessage::DoneLoading { training: true })
                .await;
        } else {
            let _ = emitter
                .emit(ProcessMessage::StartLoading { training: true })
                .await;

            let stream = train_loop::train_loop(
                vfs,
                device,
                train_receiver,
                load_data_args,
                load_init_args,
                train_config,
            );
            let mut stream = std::pin::pin!(stream);
            while let Some(message) = stream.next().await {
                emitter.emit(message?).await;
            }
        }

        Ok(())
    });

    Box::pin(stream)
}

#[derive(Debug)]
pub enum DataSource {
    PickFile,
    Url(String),
}

type DataRead = Pin<Box<dyn AsyncRead + Send>>;

impl DataSource {
    fn into_reader(self) -> anyhow::Result<impl AsyncRead + Send> {
        let (send, rec) = ::tokio::sync::mpsc::channel(16);

        // Spawn the data reading.
        tokio::spawn(async move {
            let stream = try_fn_stream(|emitter| async move {
                match self {
                    DataSource::PickFile => {
                        let picked = rrfd::pick_file()
                            .await
                            .map_err(|_| std::io::ErrorKind::NotFound)?;
                        let data = picked.read().await;
                        emitter.emit(Bytes::from_owner(data)).await;
                    }
                    DataSource::Url(url) => {
                        let mut url = url.to_owned();
                        if !url.starts_with("http://") && !url.starts_with("https://") {
                            url = format!("https://{}", url);
                        }
                        let mut response = reqwest::get(url)
                            .await
                            .map_err(|_| std::io::ErrorKind::InvalidInput)?
                            .bytes_stream();

                        while let Some(bytes) = response.next().await {
                            let bytes = bytes.map_err(|_| std::io::ErrorKind::ConnectionAborted)?;
                            emitter.emit(bytes).await;
                        }
                    }
                };
                anyhow::Result::<(), std::io::Error>::Ok(())
            });

            let mut stream = std::pin::pin!(stream);

            while let Some(data) = stream.next().await {
                if send.send(data).await.is_err() {
                    break;
                }
            }
        });

        let reader = StreamReader::new(ReceiverStream::new(rec));
        Ok(reader)
    }
}

struct CameraSettings {
    focal: f64,
    radius: f32,

    yaw_range: Range<f32>,
    pitch_range: Range<f32>,
    radius_range: Range<f32>,
}

impl ViewerContext {
    fn new(
        device: WgpuDevice,
        ctx: egui::Context,
        cam_settings: CameraSettings,
        controller: UnboundedReceiver<UiControlMessage>,
    ) -> Self {
        let model_transform = Affine3A::IDENTITY;

        let controls = OrbitControls::new(
            cam_settings.radius,
            cam_settings.radius_range,
            cam_settings.yaw_range,
            cam_settings.pitch_range,
        );

        // Camera position will be controller by controls.
        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            cam_settings.focal,
            cam_settings.focal,
            glam::vec2(0.5, 0.5),
        );

        // TODO: Generalize this "inner control" logic.
        let (inner_send, inner_control) = ::tokio::sync::mpsc::unbounded_channel();
        let ctx_spawn = ctx.clone();
        let mut controller = controller;
        task::spawn(async move {
            // Loop until there are no more messages, processing is done.
            while let Some(m) = controller.recv().await {
                ctx_spawn.request_repaint();

                // Give back to the runtime for a second.
                // This only really matters in the browser.
                tokio::task::yield_now().await;

                // If channel is closed, bail.
                if inner_send.send(m).is_err() {
                    break;
                }
            }
        });

        Self {
            camera,
            controls,
            model_transform,
            device,
            ctx,
            dataset: Dataset::empty(),
            rec_process_msg: None,
            send_train_msg: None,
            rec_ui_control_msg: inner_control,
        }
    }

    pub fn set_up_axis(&mut self, up_axis: Vec3) {
        let rotation = Quat::from_rotation_arc(Vec3::Y, up_axis);
        let model_transform = Affine3A::from_rotation_translation(rotation, Vec3::ZERO).inverse();
        self.model_transform = model_transform;
    }

    pub fn focus_view(&mut self, cam: &Camera) {
        // set the controls transform.
        let cam_transform = Affine3A::from_rotation_translation(cam.rotation, cam.position);
        let transform = self.model_transform.inverse() * cam_transform;
        let (_, rotation, position) = transform.to_scale_rotation_translation();
        self.controls.position = position.into();
        self.controls.rotation = rotation;

        // Copy the camera, mostly to copy over the intrinsics and such.
        self.controls.focus = transform.translation
            + transform.matrix3 * Vec3A::Z * self.dataset.train.bounds().extent.length() * 0.5;
        self.controls.dirty = true;
        self.camera = cam.clone();
    }

    pub(crate) fn start_data_load(
        &mut self,
        source: DataSource,
        load_data_args: LoadDatasetArgs,
        load_init_args: LoadInitArgs,
        train_config: TrainConfig,
    ) {
        let device = self.device.clone();
        log::info!("Start data load {source:?}");

        // create a channel for the train loop.
        let (train_sender, train_receiver) = channel(32);

        // Create a small channel. We don't want 10 updated splats to be stuck in the queue eating up memory!
        // Bigger channels could mean the train loop spends less time waiting for the UI though.
        let (sender, receiver) = channel(1);

        self.rec_process_msg = Some(receiver);
        self.send_train_msg = Some(train_sender);

        self.dataset = Dataset::empty();
        let ctx = self.ctx.clone();
        ctx.request_repaint();

        let fut = async move {
            ctx.request_repaint();

            // Map errors to a viewer message containing thee error.
            let mut stream = process_loop(
                source,
                device,
                train_receiver,
                load_data_args,
                load_init_args,
                train_config,
            )
            .map(|m| m.unwrap_or_else(|e| ProcessMessage::Error(Arc::new(e))));

            // Loop until there are no more messages, processing is done.
            while let Some(m) = stream.next().await {
                ctx.request_repaint();

                // Give back to the runtime for a second.
                // This only really matters in the browser.
                tokio::task::yield_now().await;

                // If channel is closed, bail.
                if sender.send(m).await.is_err() {
                    break;
                }
            }
        };

        task::spawn(fut);
    }

    pub fn send_train_message(&self, message: TrainMessage) {
        if let Some(sender) = self.send_train_msg.as_ref() {
            match sender.try_send(message) {
                Ok(_) => {}
                Err(TrySendError::Closed(_)) => {}
                Err(TrySendError::Full(_)) => {
                    unreachable!("Should use an unbounded channel for train messages.")
                }
            }
        }
    }

    fn receive_controls(&mut self) {
        while let Ok(m) = self.rec_ui_control_msg.try_recv() {
            match m {
                UiControlMessage::LoadData(url) => {
                    self.start_data_load(
                        DataSource::Url(url.to_owned()),
                        LoadDatasetArgs::default(),
                        LoadInitArgs::default(),
                        TrainConfig::default(),
                    );
                }
            }
        }
    }
}

impl Viewer {
    pub fn new(
        cc: &eframe::CreationContext,
        start_uri: Option<String>,
        controller: UnboundedReceiver<UiControlMessage>,
    ) -> Self {
        // For now just assume we're running on the default
        let state = cc.wgpu_render_state.as_ref().unwrap();
        let device = brush_ui::create_wgpu_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        brush_render::render::set_hard_floats_available(
            state
                .adapter
                .features()
                .contains(Features::SHADER_FLT32_ATOMIC),
        );

        if cfg!(feature = "tracing") {
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
                            burn_jit::JitBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>,
                        >::new(device.clone())),
                )
                .expect("Failed to set tracing subscriber");
            }
        }

        #[cfg(target_family = "wasm")]
        let start_uri = start_uri.or(web_sys::window().and_then(|w| w.location().search().ok()));

        let search_params = parse_search(&start_uri.unwrap_or("".to_owned()));

        let mut zen = false;
        if let Some(z) = search_params.get("zen") {
            zen = z.parse::<bool>().unwrap_or(false);
        }

        let focal = search_params
            .get("focal")
            .and_then(|f| f.parse().ok())
            .unwrap_or(0.5);
        let radius = search_params
            .get("radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(4.0);
        let min_radius = search_params
            .get("min_radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(1.0);
        let max_radius = search_params
            .get("max_radius")
            .and_then(|f| f.parse().ok())
            .unwrap_or(10.0);

        let min_yaw = search_params
            .get("min_yaw")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MIN);
        let max_yaw = search_params
            .get("max_yaw")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MAX);

        let min_pitch = search_params
            .get("min_pitch")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MIN);
        let max_pitch = search_params
            .get("max_pitch")
            .and_then(|f| f.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(f32::MAX);

        let settings = CameraSettings {
            focal,
            radius,
            radius_range: min_radius..max_radius,
            yaw_range: min_yaw..max_yaw,
            pitch_range: min_pitch..max_pitch,
        };

        let context = ViewerContext::new(device.clone(), cc.egui_ctx.clone(), settings, controller);

        let mut tiles: Tiles<PaneType> = Tiles::default();
        let scene_pane = ScenePanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
            zen,
        );

        let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        let root_container = if !zen {
            let loading_subs = vec![
                tiles.insert_pane(Box::new(LoadDataPanel::new())),
                tiles.insert_pane(Box::new(PresetsPanel::new())),
            ];
            let loading_pane = tiles.insert_tab_tile(loading_subs);

            #[allow(unused_mut)]
            let mut sides = vec![
                loading_pane,
                tiles.insert_pane(Box::new(StatsPanel::new(
                    device.clone(),
                    state.adapter.clone(),
                ))),
            ];

            #[cfg(not(target_family = "wasm"))]
            {
                sides.push(
                    tiles.insert_pane(Box::new(crate::panels::RerunPanel::new(device.clone()))),
                );
            }

            if cfg!(feature = "tracing") {
                sides.push(tiles.insert_pane(Box::new(TracingPanel::default())));
            }

            let side_panel = tiles.insert_vertical_tile(sides);

            let mut lin = egui_tiles::Linear::new(
                egui_tiles::LinearDir::Horizontal,
                vec![side_panel, scene_pane_id],
            );
            lin.shares.set_share(side_panel, 0.4);
            tiles.insert_container(lin)
        } else {
            scene_pane_id
        };

        let tree = egui_tiles::Tree::new("viewer_tree", root_container, tiles);

        let mut tree_ctx = ViewerTree { zen, context };

        let url = search_params.get("url");
        if let Some(url) = url {
            tree_ctx.context.start_data_load(
                DataSource::Url(url.to_owned()),
                LoadDatasetArgs::default(),
                LoadInitArgs::default(),
                TrainConfig::default(),
            );
        }

        Viewer {
            tree,
            tree_ctx,
            datasets: None,
        }
    }
}

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.tree_ctx.context.receive_controls();

        if let Some(rec) = self.tree_ctx.context.rec_process_msg.as_mut() {
            let mut messages = vec![];

            while let Ok(message) = rec.try_recv() {
                messages.push(message);
            }

            for message in messages {
                if let ProcessMessage::Dataset { data: _ } = message {
                    // Show the dataset panel if we've loaded one.
                    if self.datasets.is_none() {
                        let pane_id = self.tree.tiles.insert_pane(Box::new(DatasetPanel::new()));
                        self.datasets = Some(pane_id);
                        if let Some(Tile::Container(Container::Linear(lin))) =
                            self.tree.tiles.get_mut(self.tree.root().unwrap())
                        {
                            lin.add_child(pane_id);
                        }
                    }
                }

                for (_, pane) in self.tree.tiles.iter_mut() {
                    match pane {
                        Tile::Pane(pane) => {
                            pane.on_message(&message, &mut self.tree_ctx.context);
                        }
                        Tile::Container(_) => {}
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Close when pressing escape (in a native viewer anyway).
            if ui.input(|r| r.key_pressed(egui::Key::Escape)) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            self.tree.ui(&mut self.tree_ctx, ui);
        });
    }
}
