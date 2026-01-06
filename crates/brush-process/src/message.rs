use brush_vfs::DataSource;
use glam::Vec3;

#[cfg(feature = "training")]
use crate::config::TrainStreamConfig;

#[cfg(feature = "training")]
pub enum TrainMessage {
    /// Training configuration - sent at the start of training.
    TrainConfig {
        config: Box<TrainStreamConfig>,
    },
    /// Loaded a dataset to train on.
    Dataset {
        dataset: brush_dataset::Dataset,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    TrainStep {
        iter: u32,
        total_steps: u32,
        total_elapsed: web_time::Duration,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    RefineStep {
        cur_splat_count: u32,
        iter: u32,
    },
    /// Eval was run successfully with these results.
    #[allow(unused)]
    EvalResult {
        iter: u32,
        avg_psnr: f32,
        avg_ssim: f32,
    },
    DoneTraining,
}

pub enum ProcessMessage {
    /// A new process is starting (before we know what type)
    NewProcess,
    /// Source has been loaded, contains the display name and type
    StartLoading {
        name: String,
        source: DataSource,
        training: bool,
    },
    /// Notification that splats have been updated.
    SplatsUpdated {
        up_axis: Option<Vec3>,
        frame: u32,
        total_frames: u32,
    },
    #[cfg(feature = "training")]
    TrainMessage(TrainMessage),
    /// Some warning occurred during the process, but the process can continue.
    Warning { error: anyhow::Error },
    /// Splat, or dataset and initial splat, are done loading.
    #[allow(unused)]
    DoneLoading,
}
