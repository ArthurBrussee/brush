use ::tokio::sync::mpsc::{channel, Receiver};
use tokio_with_wasm::alias as tokio;

pub fn reactive_receiver<T: Send + 'static>(
    receiver: Receiver<T>,
    ctx: egui::Context,
) -> Receiver<T> {
    let mut receiver = receiver;
    let (send_inner, receive) = channel(1);
    tokio::spawn(async move {
        // Listen for inconimg messages.
        while let Some(m) = receiver.recv().await {
            // Mark egui as needing a repaint.
            ctx.request_repaint();
            // Wait for message to send (aka previous message is done).
            if send_inner.send(m).await.is_err() {
                break;
            }
            // Give back control to the runtime.
            // This only really matters in the browser:
            // on native, receiving also yields. In the browser that doesn't yield
            // back control fully though whereas yield_now() does.
            tokio::task::yield_now().await;
        }
    });
    receive
}