use std::{pin::Pin, sync::Arc};

use async_fn_stream::try_fn_stream;

use brush_dataset::{self, splat_import, Dataset, LoadDatasetArgs, LoadInitArgs};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_train::train::TrainStepStats;
use brush_train::{eval::EvalStats, train::TrainConfig};
use burn::backend::Autodiff;
use burn_wgpu::{Wgpu, WgpuDevice};
use eframe::egui;
use egui_tiles::{Container, Tile, TileId, Tiles};
use glam::{Quat, Vec3};
use tokio_with_wasm::alias as tokio;

use ::tokio::io::AsyncReadExt;
use ::tokio::sync::mpsc::error::TrySendError;
use ::tokio::sync::mpsc::{Receiver, Sender};
use ::tokio::{io::AsyncRead, io::BufReader, sync::mpsc::channel};
use tokio::task;

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

#[derive(Clone)]
pub(crate) enum ViewerMessage {
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
    Splats {
        iter: u32,
        splats: Box<Splats<Backend>>,
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
        stats: Box<TrainStepStats<Autodiff<Backend>>>,
        iter: u32,
        timestamp: Instant,
    },
    /// Eval was run sucesfully with these results.
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
    device: WgpuDevice,
    ctx: egui::Context,

    sender: Option<Sender<TrainMessage>>,
    receiver: Option<Receiver<ViewerMessage>>,
}

fn process_loop(
    source: DataSource,
    device: WgpuDevice,
    train_receiver: Receiver<TrainMessage>,
    load_data_args: LoadDatasetArgs,
    load_init_args: LoadInitArgs,
    train_config: TrainConfig,
) -> Pin<Box<impl Stream<Item = anyhow::Result<ViewerMessage>>>> {
    let stream = try_fn_stream(|emitter| async move {
        let _ = emitter.emit(ViewerMessage::NewSource).await;

        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let data = source.read().await?;
        let mut data = BufReader::new(data);
        let mut peek = [0; 128];
        data.read_exact(&mut peek).await?;
        let data = std::io::Cursor::new(peek).chain(data);

        log::info!("{:?}", String::from_utf8(peek.to_vec()));

        if peek.starts_with("ply".as_bytes()) {
            log::info!("Attempting to load data as .ply data");

            let _ = emitter
                .emit(ViewerMessage::StartLoading { training: false })
                .await;

            let subsample = None; // Subsampling a trained ply doesn't really make sense.
            let splat_stream = splat_import::load_splat_from_ply(data, subsample, device.clone());

            let mut splat_stream = std::pin::pin!(splat_stream);
            while let Some(splats) = splat_stream.next().await {
                emitter
                    .emit(ViewerMessage::Splats {
                        iter: 0, // For viewing just use "training step 0", bit weird.
                        splats: Box::new(splats?),
                    })
                    .await;
            }
        } else if peek.starts_with("PK".as_bytes()) {
            log::info!("Attempting to load data as .zip data");

            let _ = emitter
                .emit(ViewerMessage::StartLoading { training: true })
                .await;

            let stream = train_loop::train_loop(
                data,
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
        } else if peek.starts_with("<!DOCTYPE html>".as_bytes()) {
            anyhow::bail!("Failed to download data (are you trying to download from Google Drive? You might have to use the proxy.")
        } else {
            anyhow::bail!("only zip and ply files are supported.");
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

#[cfg(target_family = "wasm")]
type DataRead = Pin<Box<dyn AsyncRead>>;

#[cfg(not(target_family = "wasm"))]
type DataRead = Pin<Box<dyn AsyncRead + Send>>;

impl DataSource {
    async fn read(&self) -> anyhow::Result<DataRead> {
        match self {
            DataSource::PickFile => {
                let picked = rrfd::pick_file().await?;
                let data = picked.read().await;
                Ok(Box::pin(std::io::Cursor::new(data)))
            }
            DataSource::Url(url) => {
                let mut url = url.to_owned();
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    url = format!("https://{}", url);
                }
                let response = reqwest::get(url).await?.bytes_stream();
                let mapped = response
                    .map(|e| e.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
                Ok(Box::pin(tokio_util::io::StreamReader::new(mapped)))
            }
        }
    }
}

impl ViewerContext {
    fn new(device: WgpuDevice, ctx: egui::Context) -> Self {
        Self {
            camera: Camera::new(
                -Vec3::Z * 5.0,
                Quat::IDENTITY,
                0.5,
                0.5,
                glam::vec2(0.5, 0.5),
            ),
            controls: OrbitControls::new(),
            device,
            ctx,
            dataset: Dataset::empty(),
            receiver: None,
            sender: None,
        }
    }

    pub fn focus_view(&mut self, cam: &Camera) {
        self.camera = cam.clone();
        self.controls.focus = self.camera.position
            + self.camera.rotation
                * glam::Vec3::Z
                * self.dataset.train.bounds(0.0, 0.0).extent.length()
                * 0.5;
        self.controls.dirty = true;
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

        self.receiver = Some(receiver);
        self.sender = Some(train_sender);

        self.dataset = Dataset::empty();
        let ctx = self.ctx.clone();

        let fut = async move {
            // Map errors to a viewer message containing thee error.
            let mut stream = process_loop(
                source,
                device,
                train_receiver,
                load_data_args,
                load_init_args,
                train_config,
            )
            .map(|m| match m {
                Ok(m) => m,
                Err(e) => ViewerMessage::Error(Arc::new(e)),
            });

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
        if let Some(sender) = self.sender.as_ref() {
            match sender.try_send(message) {
                Ok(_) => {}
                Err(TrySendError::Closed(_)) => {}
                Err(TrySendError::Full(_)) => {
                    unreachable!("Should use an unbounded channel for train messages.")
                }
            }
        }
    }
}

impl Viewer {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        // For now just assume we're running on the default
        let state = cc.wgpu_render_state.as_ref().unwrap();
        let device = brush_ui::create_wgpu_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        cfg_if::cfg_if! {
            if #[cfg(target_family = "wasm")] {
                use tracing_subscriber::layer::SubscriberExt;

                let subscriber = tracing_subscriber::registry().with(tracing_wasm::WASMLayer::new(Default::default()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            } else if #[cfg(feature = "tracy")] {
                use tracing_subscriber::layer::SubscriberExt;
                let subscriber = tracing_subscriber::registry()
                    .with(tracing_tracy::TracyLayer::default())
                    .with(sync_span::SyncLayer::new(device.clone()));
                tracing::subscriber::set_global_default(subscriber)
                    .expect("Failed to set tracing subscriber");
            }
        }

        let mut start_url = None;
        if cfg!(target_family = "wasm") {
            if let Some(window) = web_sys::window() {
                if let Ok(search) = window.location().search() {
                    if let Ok(search_params) = web_sys::UrlSearchParams::new_with_str(&search) {
                        let url = search_params.get("url");
                        start_url = url;
                    }
                }
            }
        }

        let mut tiles: Tiles<PaneType> = egui_tiles::Tiles::default();

        let context = ViewerContext::new(device.clone(), cc.egui_ctx.clone());

        let scene_pane = ScenePanel::new(
            state.queue.clone(),
            state.device.clone(),
            state.renderer.clone(),
        );

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
            sides.push(tiles.insert_pane(Box::new(crate::panels::RerunPanel::new(device.clone()))));
        }

        if cfg!(feature = "tracing") {
            sides.push(tiles.insert_pane(Box::new(TracingPanel::default())));
        }

        let side_panel = tiles.insert_vertical_tile(sides);
        let scene_pane_id = tiles.insert_pane(Box::new(scene_pane));

        let mut lin = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![side_panel, scene_pane_id],
        );
        lin.shares.set_share(side_panel, 0.4);

        let root_container = tiles.insert_container(lin);
        let tree = egui_tiles::Tree::new("viewer_tree", root_container, tiles);

        let mut tree_ctx = ViewerTree { context };

        if let Some(start_url) = start_url {
            tree_ctx.context.start_data_load(
                DataSource::Url(start_url.to_owned()),
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
        if let Some(rec) = self.tree_ctx.context.receiver.as_mut() {
            let mut messages = vec![];

            while let Ok(message) = rec.try_recv() {
                messages.push(message);
            }

            for message in messages {
                if let ViewerMessage::Dataset { data: _ } = message {
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

                ctx.request_repaint();
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
