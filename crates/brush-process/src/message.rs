use brush_render::MainBackend;
use brush_render::gaussian_splats::Splats;
use glam::Vec3;

#[cfg(feature = "training")]
pub enum TrainMessage {
    /// Loaded a dataset to train on.
    Dataset {
        dataset: brush_dataset::Dataset,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    TrainStep {
        splats: Box<Splats<MainBackend>>,
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
    /// Source has been loaded, contains the display name
    NewSource {
        name: String,
    },
    StartLoading {
        training: bool,
    },
    /// Loaded a splat from a ply file.
    ///
    /// Nb: This includes all the intermediately loaded splats.
    /// Nb: Animated splats will have the 'frame' number set.
    ViewSplats {
        up_axis: Option<Vec3>,
        splats: Box<Splats<MainBackend>>,
        frame: u32,
        total_frames: u32,
        progress: f32,
    },

    #[cfg(feature = "training")]
    TrainMessage(TrainMessage),

    /// Some warning occurred during the process, but the process can continue.
    Warning {
        error: anyhow::Error,
    },
    /// Splat, or dataset and initial splat, are done loading.
    #[allow(unused)]
    DoneLoading,
}
