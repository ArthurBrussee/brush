use brush_dataset::Dataset;
use brush_render::MainBackend;
use brush_render::gaussian_splats::Splats;
use brush_train::msg::{RefineStats, TrainStepStats};
use glam::Vec3;
use web_time::Duration;

pub enum ProcessMessage {
    NewSource,
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
    },
    /// Loaded a bunch of viewpoints to train on.
    Dataset {
        dataset: Dataset,
    },
    /// Splat, or dataset and initial splat, are done loading.
    #[allow(unused)]
    DoneLoading,
    /// Some number of training steps are done.
    #[allow(unused)]
    TrainStep {
        splats: Box<Splats<MainBackend>>,
        stats: Box<TrainStepStats<MainBackend>>,
        iter: u32,
        total_elapsed: Duration,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    RefineStep {
        stats: Box<RefineStats>,
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
}
