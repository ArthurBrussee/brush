use burn_cubecl::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, streaming::StreamingConfig};

#[allow(unused)]
pub(crate) fn startup() {
    // Set the global config once on startup.
    CubeClRuntimeConfig::set(CubeClRuntimeConfig {
        streaming: StreamingConfig {
            max_streams: 1,
            ..Default::default()
        },
        ..Default::default()
    });
}
