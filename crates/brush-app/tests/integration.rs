#![cfg(feature = "training")]

use std::ffi::{CString, c_void};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use brush_app::ffi::{ProgressMessage, TrainExitCode, TrainOptions, train_and_save};

#[repr(C)]
struct CallbackState {
    call_count: AtomicUsize,
    finished_called: std::sync::atomic::AtomicBool,
}

extern "C" fn test_progress_callback(process_message: ProgressMessage, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: user_data is a pointer to a CallbackState struct
    let state = unsafe { (user_data as *const CallbackState).as_ref().unwrap() };
    state.call_count.fetch_add(1, Ordering::SeqCst);

    match process_message {
        ProgressMessage::NewProcess => {
            println!("FFI Test: Training starting...");
        }
        ProgressMessage::Training { iter } => {
            println!("FFI Test: Training iteration: {iter:.2}%");
        }
        ProgressMessage::DoneTraining => {
            println!("FFI Test: Training finished!");
            state
                .finished_called
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[test]
fn test_train_and_save_ffi_short() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dataset_path = Path::new(manifest_dir)
        .join("tests")
        .join("data")
        .join("test_dataset");

    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();

    let dataset_path_cstr = CString::new(dataset_path.to_str().unwrap()).unwrap();

    let mut callback_state = CallbackState {
        call_count: AtomicUsize::new(0),
        finished_called: std::sync::atomic::AtomicBool::new(false),
    };

    let options = TrainOptions {
        total_steps: 10,
        refine_every: 5,
        export_every: 10,
        max_resolution: 50,
        output_path: output_path_cstr.as_ptr(),
    };

    // SAFETY: paths are valid, user_data is valid for lifetime of callback_state
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Success));
    assert!(callback_state.call_count.load(Ordering::SeqCst) > 2);

    let output_files: Vec<_> = fs::read_dir(output_path)
        .unwrap()
        .filter_map(Result::ok)
        .collect();
    assert!(!output_files.is_empty(), "No output file was created");
}

#[test]
fn test_train_and_save_ffi_invalid_path() {
    let invalid_dataset_path = "/path/that/does/not/exist/and/should/fail";
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_invalid_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();

    let dataset_path_cstr = CString::new(invalid_dataset_path).unwrap();
    let mut callback_state = CallbackState {
        call_count: AtomicUsize::new(0),
        finished_called: std::sync::atomic::AtomicBool::new(false),
    };

    let options = TrainOptions {
        total_steps: 10,
        refine_every: 5,
        export_every: 10,
        max_resolution: 50,
        output_path: output_path_cstr.as_ptr(),
    };

    // SAFETY: The paths are valid, and the callback state is alive for the duration of the call.
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            &options,
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Error));
}

#[test]
fn test_train_and_save_ffi_null_options() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dataset_path = Path::new(manifest_dir)
        .join("tests")
        .join("data")
        .join("test_dataset");

    let dataset_path_cstr = CString::new(dataset_path.to_str().unwrap()).unwrap();

    let mut callback_state = CallbackState {
        call_count: AtomicUsize::new(0),
        finished_called: std::sync::atomic::AtomicBool::new(false),
    };

    // SAFETY: The paths are valid, and the callback state is alive for the duration of the call.
    let status = unsafe {
        train_and_save(
            dataset_path_cstr.as_ptr(),
            std::ptr::null(),
            test_progress_callback,
            std::ptr::from_mut(&mut callback_state).cast::<c_void>(),
        )
    };

    assert!(matches!(status, TrainExitCode::Error));
}

#[test]
fn test_train_and_save_ffi_null_dataset() {
    let temp_dir = tempfile::Builder::new()
        .prefix("ffi_test_invalid_")
        .tempdir()
        .unwrap();
    let output_path = temp_dir.path().to_str().unwrap();
    let output_path_cstr = CString::new(output_path).unwrap();

    let options = TrainOptions {
        total_steps: 10,
        refine_every: 5,
        export_every: 10,
        max_resolution: 50,
        output_path: output_path_cstr.as_ptr(),
    };

    // SAFETY: The paths are valid, and the callback state is null.
    let status_null_dataset = unsafe {
        train_and_save(
            std::ptr::null(),
            &options,
            test_progress_callback,
            std::ptr::null_mut(),
        )
    };

    assert!(matches!(status_null_dataset, TrainExitCode::Error));
}
