pub mod ui;

use brush_process::{data_source::DataSource, process_loop::ProcessArgs};
use clap::{builder::ArgPredicate, error::ErrorKind, Error, Parser};

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

    #[arg(
        long,
        default_value = "true",
        default_value_if("source", ArgPredicate::IsPresent, "false")
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub process: ProcessArgs,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}
