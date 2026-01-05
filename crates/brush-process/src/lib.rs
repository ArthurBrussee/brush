#![recursion_limit = "256"]

pub mod config;
pub mod message;
pub mod process;

#[cfg(feature = "training")]
pub mod train_stream;

pub mod slot;
