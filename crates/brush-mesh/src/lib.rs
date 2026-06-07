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
pub mod tetra_points;
pub mod view_select;

pub use extract::{ExtractConfig, extract_mesh};

/// Triangle mesh produced by the extractor. Vertices are world-space.
/// Per-vertex `scale` is the seed-point's owning Gaussian's max axis length
/// (used by the optional [`filter::filter_mesh`] step). Per-vertex `colors`
/// (u8 RGB) come from the brush-rendered RGB image of the view that had
/// the best line of sight to the vertex.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<glam::Vec3>,
    pub vertex_scales: Vec<f32>,
    pub vertex_colors: Vec<[u8; 3]>,
    pub faces: Vec<[u32; 3]>,
}
