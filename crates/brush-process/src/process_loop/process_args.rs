use brush_dataset::{LoadDatasetArgs, ModelOptions};
use brush_train::train::TrainConfig;
use clap::Args;

#[derive(Clone, Default, Args)]
pub struct ProcessArgs {
    #[clap(flatten)]
    pub train_config: TrainConfig,
    #[clap(flatten)]
    pub init_args: ModelOptions,
    #[clap(flatten)]
    pub load_args: LoadDatasetArgs,
}
