use crate::data_source::DataSource;
use brush_dataset::{LoadDatasetArgs, LoadInitArgs};
use brush_train::train::TrainConfig;

#[derive(Clone)]
pub struct ProcessArgs {
    pub source: DataSource,
    pub load_args: LoadDatasetArgs,
    pub init_args: LoadInitArgs,
    pub train_config: TrainConfig,
}
