use std::sync::Mutex;
use std::sync::OnceLock;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use tracing::{info_span, Subscriber};
use tracing_subscriber::{
    layer::{Context, Layer},
    registry::LookupSpan,
};

// Global device storage
static SYNC_DEVICE: OnceLock<Mutex<Option<WgpuDevice>>> = OnceLock::new();

// Tracing layer for sync events
#[derive(Default)]
pub struct SyncLayer {}

impl<S> Layer<S> for SyncLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let metadata = ctx.metadata(&id).expect("Span ID invalid");

        if metadata.is_span() && metadata.fields().field("sync_burn").is_some() {
            if let Some(device) = SYNC_DEVICE
                .get()
                .and_then(|lock| lock.lock().ok())
                .as_deref()
                .and_then(|opt| opt.as_ref())
            {
                let _span = info_span!("GPU Wait", name = metadata.name()).entered();
                <Wgpu as burn::prelude::Backend>::sync(device);
            }
        }
    }
}

pub fn is_enabled() -> bool {
    SYNC_DEVICE
        .get()
        .and_then(|lock| lock.lock().ok())
        .is_some_and(|guard| guard.is_some())
}

pub fn set_enabled(device: Option<WgpuDevice>) {
    *SYNC_DEVICE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("Failed to lock sync device") = device;
}
