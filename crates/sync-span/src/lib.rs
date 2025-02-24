use std::sync::atomic::{AtomicBool, Ordering};

use burn::prelude::Backend;
use tracing::{Subscriber, info_span};
use tracing_subscriber::{
    layer::{Context, Layer},
    registry::LookupSpan,
};

// Global flag to enable/disable sync
static SYNC_ENABLED: AtomicBool = AtomicBool::new(false);

// Tracing layer for sync events
pub struct SyncLayer<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SyncLayer<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend, S> Layer<S> for SyncLayer<B>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        if SYNC_ENABLED.load(Ordering::Relaxed) {
            let metadata = ctx.metadata(&id).expect("Span ID invalid");

            if metadata.is_span() && metadata.fields().field("sync_burn").is_some() {
                let _span = info_span!("GPU Wait", name = metadata.name()).entered();
                // TODO: Need something that works on wasm.
                B::sync(&self.device);
            }
        }
    }
}

pub fn is_enabled() -> bool {
    SYNC_ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(enabled: bool) {
    SYNC_ENABLED.store(enabled, Ordering::Relaxed);
}
