use brush_render::AlphaMode;
use clap::Args;

fn parse_max_resolution(s: &str) -> Result<u32, String> {
    if s.eq_ignore_ascii_case("max") {
        Ok(u32::MAX)
    } else {
        s.parse::<u32>()
            .map_err(|e| format!("Invalid max resolution '{s}': {e}"))
    }
}

#[derive(Clone, Debug, Args)]
pub struct ModelConfig {
    /// SH degree of splats.
    #[arg(long, help_heading = "Model Options", default_value = "3")]
    pub sh_degree: u32,
}

#[derive(Clone, Debug, Args)]
pub struct LoadDataseConfig {
    /// Max nr. of frames of dataset to load
    #[arg(long, help_heading = "Dataset Options")]
    pub max_frames: Option<usize>,
    /// Max resolution of images to load.
    ///
    /// Pass `max` to use the maximum source image resolution detected in the dataset.
    #[arg(
        long,
        help_heading = "Dataset Options",
        default_value = "1920",
        value_parser = parse_max_resolution
    )]
    pub max_resolution: u32,
    /// Create an eval dataset by selecting every nth image
    #[arg(long, help_heading = "Dataset Options")]
    pub eval_split_every: Option<usize>,
    /// Load only every nth frame
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_frames: Option<u32>,
    /// Load only every nth point from the initial sfm data
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_points: Option<u32>,
    /// Whether to interpret an alpha channel (or masks) as transparency or masking.
    #[arg(long, help_heading = "Dataset Options")]
    pub alpha_mode: Option<AlphaMode>,
}

#[cfg(test)]
mod tests {
    use super::parse_max_resolution;

    #[test]
    fn parse_max_resolution_allows_max() {
        assert_eq!(parse_max_resolution("max").unwrap(), u32::MAX);
        assert_eq!(parse_max_resolution("MAX").unwrap(), u32::MAX);
    }

    #[test]
    fn parse_max_resolution_allows_numbers() {
        assert_eq!(parse_max_resolution("1920").unwrap(), 1920);
    }

    #[test]
    fn parse_max_resolution_rejects_invalid() {
        assert!(parse_max_resolution("nope").is_err());
    }
}
