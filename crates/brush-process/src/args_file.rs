use std::path::Path;

use brush_vfs::BrushVfs;
use clap::Parser;
use tokio::io::AsyncReadExt;

use crate::config::TrainStreamConfig;

pub fn split_args_str(content: &str) -> Vec<String> {
    content.split_whitespace().map(|s| s.to_owned()).collect()
}

/// Load `TrainStreamConfig` from args.txt via VFS.
pub async fn load_config_from_vfs(vfs: &BrushVfs) -> Option<TrainStreamConfig> {
    let args_path = Path::new("args.txt");

    // Check if args.txt exists in the VFS
    if !vfs
        .file_paths()
        .any(|p| p.file_name().and_then(|n| n.to_str()) == Some("args.txt"))
    {
        return None;
    }

    let mut reader = match vfs.reader_at_path(args_path).await {
        Ok(r) => r,
        Err(e) => {
            log::warn!("Failed to open args: {e}");
            return None;
        }
    };

    let mut content = String::new();
    if let Err(e) = reader.read_to_string(&mut content).await {
        log::warn!("Failed to read args: {e}");
        return None;
    }

    let file_args = split_args_str(&content);
    if file_args.is_empty() {
        return None;
    }

    log::info!("Loaded settings from args");
    let mut all_args = vec!["brush".to_owned()];
    all_args.extend(file_args);

    TrainStreamConfig::try_parse_from(&all_args).ok()
}

/// Convert a `TrainStreamConfig` back to command-line argument format for saving.
/// Only includes values that differ from the defaults.
pub fn config_to_args(config: &TrainStreamConfig) -> Vec<String> {
    use serde_json::Value;

    let config_json = serde_json::to_value(config).unwrap_or(Value::Null);
    let default_json = serde_json::to_value(TrainStreamConfig::default()).unwrap_or(Value::Null);

    let mut args = Vec::new();

    if let (Value::Object(config_map), Value::Object(default_map)) = (config_json, default_json) {
        for (key, value) in config_map {
            // Skip if value equals default
            if default_map.get(&key) == Some(&value) {
                continue;
            }

            // Skip null values (None options that are also None by default would be caught above)
            if value.is_null() {
                continue;
            }

            // Format the argument
            let arg_name = format!("--{key}");
            match value {
                Value::Bool(b) => {
                    // For booleans, only output flag if true (and different from default)
                    if b {
                        args.push(arg_name);
                    }
                }
                Value::String(s) => {
                    args.push(format!("{arg_name} {s}"));
                }
                Value::Number(n) => {
                    args.push(format!("{arg_name} {n}"));
                }
                _ => {
                    // For other types, use the JSON representation
                    args.push(format!("{arg_name} {value}"));
                }
            }
        }
    }

    args
}

/// Merge an initial config (e.g., from args.txt) with CLI arguments.
/// CLI arguments take precedence over the initial config values.
pub fn merge_configs(
    initial_config: &TrainStreamConfig,
    cli_config: &TrainStreamConfig,
) -> TrainStreamConfig {
    let initial_args = config_to_args(initial_config);
    let cli_args = config_to_args(cli_config);

    // Combine: initial first, then CLI
    let mut all_args = vec!["brush".to_owned()];
    for arg in &initial_args {
        all_args.extend(arg.split_whitespace().map(|s| s.to_owned()));
    }
    for arg in &cli_args {
        all_args.extend(arg.split_whitespace().map(|s| s.to_owned()));
    }

    // Parse the combined arguments
    match TrainStreamConfig::try_parse_from(&all_args) {
        Ok(config) => config,
        Err(e) => {
            log::warn!("Failed to parse merged config: {e}");
            cli_config.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_to_args_only_includes_changes() {
        let mut config = TrainStreamConfig::default();
        config.train_config.total_steps = 5000;
        config.model_config.sh_degree = 2;
        config.load_config.max_frames = Some(10);
        let args = config_to_args(&config);

        // Should have exactly 3 args for the 3 changes
        assert_eq!(args.len(), 3, "Should have 3 args, got: {args:?}");
        // Verify they contain the right values
        let args_str = args.join(" ");
        assert!(
            args_str.contains("--total-steps 5000"),
            "Missing total-steps"
        );
        assert!(args_str.contains("--sh-degree 2"), "Missing sh-degree");
        assert!(args_str.contains("--max-frames 10"), "Missing max-frames");
    }

    #[test]
    fn test_config_round_trip() {
        let mut original = TrainStreamConfig::default();
        original.train_config.total_steps = 5000;
        original.model_config.sh_degree = 2;
        original.load_config.max_frames = Some(10);
        original.process_config.seed = 123;

        // Convert to args
        let args = config_to_args(&original);

        // Parse args back
        let mut cli_args = vec!["brush".to_owned()];
        for arg in &args {
            cli_args.extend(arg.split_whitespace().map(|s| s.to_owned()));
        }

        let parsed = TrainStreamConfig::try_parse_from(&cli_args).expect("Should parse");
        assert_eq!(parsed.train_config.total_steps, 5000);
        assert_eq!(parsed.model_config.sh_degree, 2);
        assert_eq!(parsed.load_config.max_frames, Some(10));
        assert_eq!(parsed.process_config.seed, 123);
    }
}
