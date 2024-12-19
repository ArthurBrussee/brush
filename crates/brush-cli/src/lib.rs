mod ui;

use brush_process::{data_source::DataSource, process_loop::ProcessArgs};
use clap::Parser;

#[derive(Parser)]
#[command(
    author,
    version,
    arg_required_else_help = false,
    about = "Brush - universal splats"
)]
pub struct Cli {
    /// Source to load from (path or URL).
    #[arg(value_name = "PATH_OR_URL")]
    pub source: Option<DataSource>,
    #[clap(flatten)]
    pub process: ProcessArgs,
}
