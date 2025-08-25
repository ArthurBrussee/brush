#![recursion_limit = "256"]

pub mod export;
pub mod import;
pub mod ply_gaussian;
pub mod quant;

// Re-export main functionality
pub use export::splat_to_ply;
pub use import::{ParseMetadata, SplatMessage, load_splat_from_ply, stream_splat_from_ply};
pub use ply_gaussian::{PlyGaussian, QuantSh, QuantSplat};

// Re-export serde-ply types for compatibility
pub use serde_ply::{DeserializeError, SerializeError};
