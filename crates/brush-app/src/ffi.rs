use brush_process::config::TrainStreamConfig;
use brush_process::message::TrainMessage;
use brush_process::{message::ProcessMessage, process::create_process};
use brush_vfs::DataSource;
use burn_wgpu::WgpuDevice;
use std::convert::TryFrom;
use std::ffi::{CStr, c_char, c_void};
use tokio::sync::oneshot;
use tokio_stream::StreamExt;

#[repr(C)]
pub enum TrainExitCode {
    Success = 0,
    Error = 1,
}

#[repr(C)]
pub enum ProgressMessage {
    NewProcess,
    Training { iter: u32 },
    DoneTraining,
}

impl TryFrom<ProcessMessage> for ProgressMessage {
    type Error = ();

    fn try_from(value: ProcessMessage) -> Result<Self, Self::Error> {
        match value {
            ProcessMessage::NewProcess => Ok(Self::NewProcess),
            ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter, .. }) => {
                Ok(Self::Training { iter })
            }
            ProcessMessage::TrainMessage(TrainMessage::DoneTraining) => Ok(Self::DoneTraining),
            _ => Err(()),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrainOptions {
    pub total_steps: u32,
    pub refine_every: u32,
    pub max_resolution: u32,
    pub export_every: u32,
    pub output_path: *const c_char,
}

impl TrainOptions {
    /// # Safety
    ///
    /// If `output_path` is not null, it must be a valid pointer to a null-terminated C string.
    unsafe fn into_train_stream_config(self) -> TrainStreamConfig {
        let process_args = TrainStreamConfig::default();
        let mut process_args = process_args;
        if !self.output_path.is_null() {
            // SAFETY: Path is not null, caller guarantees the string is a valid C-string.
            process_args.process_config.export_path = unsafe {
                CStr::from_ptr(self.output_path)
                    .to_string_lossy()
                    .into_owned()
            };
        }
        process_args.train_config.total_steps = self.total_steps;
        process_args.train_config.refine_every = self.refine_every;
        process_args.load_config.max_resolution = self.max_resolution;
        process_args.process_config.export_every = self.export_every;
        process_args.process_config.eval_save_to_disk = true;
        process_args
    }
}

pub type ProgressCallback =
    extern "C" fn(progress_message: ProgressMessage, user_data: *mut c_void);

/// Trains a model from a dataset and saves the result.
///
/// This function is designed to be called from other languages via FFI. It will
/// block the current thread until training is complete.
///
/// # Arguments
///
/// * `dataset_path` - A pointer to a null-terminated C string representing the path to the dataset.
/// * `options` - A pointer to a `TrainOptions` struct.
/// * `progress_callback` - A callback function that will be invoked with progress updates.
/// * `user_data` - An opaque pointer passed to the `progress_callback`.
///
/// # Safety
///
/// The caller must uphold several invariants. Passing `null` for `dataset_path` or `options`
/// is safe and will result in an error code, but if they are non-null, they must be valid.
///
/// - If `dataset_path` is not null, it must point to a valid, null-terminated C string. The
///   memory it points to must be valid for reading for the duration of this call.
///
/// - If `options` is not null, it must point to a valid `TrainOptions` struct. The memory it
///   points to must be valid for reading for the duration of this call. It's `output_path` must
///   be a valid, null-terminated C string if not null.
///
/// - The `user_data` pointer is passed to `progress_callback` but is not dereferenced by this
///   function. If it is not null, the caller must ensure it points to memory that remains
///   valid for the entire duration of this function call, as the callback may dereference it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_and_save(
    dataset_path: *const c_char,
    options: *const TrainOptions,
    progress_callback: ProgressCallback,
    user_data: *mut c_void,
) -> TrainExitCode {
    if dataset_path.is_null() || options.is_null() {
        return TrainExitCode::Error;
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    rt.block_on(async {
        let dataset_path_str =
            // SAFETY: Checked if dataset_path is not null, caller guarantees the string is a valid C-string.
            unsafe { CStr::from_ptr(dataset_path).to_string_lossy().into_owned() };

        let source = DataSource::Path(dataset_path_str);

        let device = WgpuDevice::default();

        // SAFETY: Option is checked to not be null before the future.
        let train_options = unsafe { *options };
        // SAFETY: Caller guarantees the output_path is a valid C-string if not null.
        let process_args = unsafe { train_options.into_train_stream_config() };
        let (tx, rx) = oneshot::channel::<TrainStreamConfig>();
        let _ = tx.send(process_args.clone());
        let mut stream = std::pin::pin!(create_process(source, rx, device));

        while let Some(message_result) = stream.next().await {
            match message_result {
                Ok(message) => {
                    if let Ok(progress_message) = message.try_into() {
                        progress_callback(progress_message, user_data);
                    }
                }
                Err(_) => {
                    return TrainExitCode::Error;
                }
            }
        }

        TrainExitCode::Success
    })
}
