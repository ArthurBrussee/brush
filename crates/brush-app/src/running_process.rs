use brush_process::{
    data_source::DataSource,
    process_loop::{ControlMessage, ProcessArgs, ProcessMessage, process_stream},
};
use burn_wgpu::WgpuDevice;
use tokio::sync::mpsc::{Receiver, UnboundedSender};
use tokio_stream::StreamExt;
use tokio_with_wasm::alias as tokio_wasm;

pub struct RunningProcess {
    pub start_args: ProcessArgs,
    pub messages: Receiver<Result<ProcessMessage, anyhow::Error>>,
    pub control: UnboundedSender<ControlMessage>,
}

pub fn start_process(
    source: DataSource,
    args: ProcessArgs,
    device: WgpuDevice,
    ctx: egui::Context,
) -> RunningProcess {
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let (train_sender, mut train_receiver) = tokio::sync::mpsc::unbounded_channel();

    let args_loop = args.clone();

    tokio_with_wasm::alias::task::spawn(async move {
        let stream = process_stream(source, args_loop, device);
        let mut stream = std::pin::pin!(stream);

        while let Some(msg) = stream.next().await {
            // Mark egui as needing a repaint.
            ctx.request_repaint();

            sender.send(msg).await.ok();

            // Pause if needed.
            if matches!(train_receiver.try_recv(), Ok(ControlMessage::Paused(true))) {
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

    RunningProcess {
        start_args: args,
        messages: receiver,
        control: train_sender,
    }
}
