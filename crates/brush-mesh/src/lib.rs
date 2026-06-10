//! GOF-style triangle-mesh extraction from a trained 3DGS splat.
//!
//! Implements the pipeline from "Gaussian Opacity Fields: Efficient Adaptive
//! Surface Reconstruction in Unbounded Scenes" (Yu et al., SIGGRAPH Asia '24)
//! against an already-trained splat: sample tetrahedral seed points from each
//! Gaussian, build a 3D Delaunay triangulation on CPU, integrate per-point
//! opacity along training-camera rays on GPU, then run marching tets +
//! binary-search refinement against the α = 0.5 level set.
//!
//! Reference: <https://github.com/autonomousvision/gaussian-opacity-fields>.

pub mod delaunay;
pub mod extract;
pub mod filter;
pub mod marching_tet;
pub mod ply;
pub mod refine;
pub mod simplify;
pub mod smooth;
pub mod tetra_points;
pub mod texture;

pub use extract::{ExtractConfig, ExtractOutput, extract_mesh};

/// Triangle mesh produced by the extractor. Vertices are world-space.
/// Per-vertex `colors` (u8 RGB) are the visibility-weighted blend of the
/// brush-rendered RGB across the views that saw the vertex.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<glam::Vec3>,
    pub vertex_colors: Vec<[u8; 3]>,
    pub faces: Vec<[u32; 3]>,
}
