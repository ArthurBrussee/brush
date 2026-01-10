use burn_cubecl::cubecl::config::{GlobalConfig, streaming::StreamingConfig};

#[allow(unused)]
pub(crate) fn startup() {
    // Set the global config once on startup.
    GlobalConfig::set(GlobalConfig {
        streaming: StreamingConfig {
            max_streams: 1,
            ..Default::default()
        },
        ..Default::default()
    });
}
