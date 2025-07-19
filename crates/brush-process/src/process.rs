use std::sync::Arc;

use brush_vfs::DataSource;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use tokio::sync::mpsc::{Receiver, Sender};

#[allow(unused)]
use brush_dataset::splat_export;

use crate::{
    config::ProcessArgs, message::ProcessMessage, train_stream::train_stream,
    view_stream::view_stream,
};

pub struct Process {
    pub stream: Receiver<ProcessMessage>,
    // last_splat: watch::Receiver<Splats<MainBackend>>,
}

#[derive(Clone)]
pub struct ProcessSend {
    send: Arc<Sender<ProcessMessage>>,
}

impl ProcessSend {
    pub fn new(send: Sender<ProcessMessage>) -> Self {
        Self {
            send: Arc::new(send),
        }
    }

    pub async fn send(&self, message: ProcessMessage) -> anyhow::Result<()> {
        self.send.send(message).await?;
        Ok(())
    }
}

fn new_process_channel() -> (ProcessSend, Process) {
    let (send, stream) = tokio::sync::mpsc::channel(16);
    (ProcessSend::new(send), Process { stream })
}

pub fn start_process_stream(
    source: DataSource,
    process_args: tokio::sync::oneshot::Receiver<ProcessArgs>,
    device: WgpuDevice,
) -> Process {
    let (send, process) = new_process_channel();
    tokio_with_wasm::alias::task::spawn(async move {
        let _ = send.send(ProcessMessage::NewProcess).await;
        let result = process_stream(source, process_args, device, send.clone()).await;
        let _ = send.send(ProcessMessage::Complete { result }).await;
    });
    process
}

async fn process_stream(
    source: DataSource,
    process_args: tokio::sync::oneshot::Receiver<ProcessArgs>,
    device: WgpuDevice,
    stream: ProcessSend,
) -> anyhow::Result<()> {
    log::info!("Starting process with source {source:?}");
    let vfs = Arc::new(source.into_vfs().await?);

    let client = WgpuRuntime::client(&device);
    // Start with memory cleared out.
    client.memory_cleanup();

    let vfs_counts = vfs.file_count();
    let ply_count = vfs.files_with_extension("ply").count();

    log::info!(
        "Mounted VFS with {} files. (plys: {})",
        vfs.file_count(),
        ply_count
    );

    log::info!("Start of view stream");

    if vfs_counts == ply_count {
        drop(process_args);
        view_stream(vfs, device, stream).await?;
    } else {
        // Receive the processing args.
        train_stream(vfs, process_args, device, stream).await?;
    };

    Ok(())
}
