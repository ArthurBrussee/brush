use burn_cubecl::cubecl::config::{GlobalConfig, streaming::StreamingConfig};
use std::sync::Once;

static STARTUP: Once = Once::new();

#[allow(unused)]
pub(crate) fn startup() {
    STARTUP.call_once(|| {
        // Set the global config once on startup.
        GlobalConfig::set(GlobalConfig {
            streaming: StreamingConfig {
                max_streams: 1,
                ..Default::default()
            },
            ..Default::default()
        });
    });
}
