# Chunked Rendering Implementation Plan

## Overview

This plan details how to implement chunked rendering (max 1024x1024 per chunk) for the Gaussian splatting renderer. This is necessary to support arbitrarily large render resolutions while keeping GPU memory bounded.

## Current Architecture Summary

**Forward Pass Flow:**
1. `ProjectSplats` - Projects all gaussians, culls invisible ones, computes depths
2. `DepthSort` - Sorts visible gaussians by depth (back-to-front)
3. `ProjectVisible` - Computes 2D xy, conic, SH colors for visible gaussians → `projected_splats`
4. `MapGaussiansToIntersect` (prepass) - Counts tile intersections per gaussian
5. `PrefixSum` - Cumulative intersection counts
6. `MapGaussiansToIntersect` (main) - Writes tile→gaussian mapping
7. `TileSort` - Sorts intersections by tile ID
8. `GetTileOffsets` - Creates per-tile start/end offsets into intersection buffer
9. `Rasterize` - Renders each tile (16x16 pixels), one workgroup per tile

**Backward Pass Flow:**
1. `RasterizeBackwards` - Computes gradients w.r.t. projected_splats (xy, conic, rgb, alpha)
2. `ProjectBackwards` - Computes gradients w.r.t. original parameters (means, scales, quats, sh_coeffs, opacities)

**Key Tensors Stored for Backward:**
- `projected_splats` - [num_visible, 9] - 2D positions, conics, colors
- `compact_gid_from_isect` - [num_intersections] - Per-intersection gaussian IDs
- `global_from_compact_gid` - [num_visible] - Maps compact→global gaussian IDs
- `tile_offsets` - [tile_y, tile_x, 2] - Per-tile intersection ranges
- `out_img` - [height, width, 4] - Forward pass output
- `uniforms_buffer` - Render uniforms

## Design Decisions

### Chunk Size
- Maximum chunk size: 1024x1024 pixels
- Tiles are 16x16 pixels, so max 64x64 = 4096 tiles per chunk
- Chunks will be aligned to tile boundaries

### What Changes Per Chunk
1. **Intersection buffers** - Must be computed per-chunk (tile→gaussian mapping is chunk-specific)
2. **Tile offsets** - Per-chunk (different tiles per chunk)
3. **Rasterization** - Per-chunk (writes to different parts of output image)

### What Stays Global (Computed Once)
1. **ProjectSplats** - Visibility culling is based on full image frustum
2. **DepthSort** - Global depth ordering (back-to-front)
3. **ProjectVisible** - Computes projected_splats once for all chunks

### Backward Pass Changes
The key insight: **`RasterizeBackwards` needs to be run per-chunk**, but it writes gradients atomically to global per-gaussian buffers (`v_grads`, `v_opacs`, `v_refines`). This is already safe for chunked execution.

**`ProjectBackwards`** runs once globally after all chunks, using the accumulated `v_grads`.

### Memory Savings
By chunking, we avoid storing:
- Large per-chunk `tile_offsets` (now computed on-demand per chunk)
- Large per-chunk `compact_gid_from_isect` (now computed on-demand per chunk)

For backward pass with recomputation:
- We no longer store `tile_offsets`, `compact_gid_from_isect` for the entire image
- Instead, backward pass recomputes these per-chunk

## Implementation Plan

### Phase 1: Refactor Forward Pass for Chunking

#### Step 1.1: Add chunk iteration infrastructure
**Files:** `crates/brush-render/src/render.rs`

- Add `ChunkConfig` struct with chunk dimensions and iteration logic
- Add helper function `iter_chunks(img_size: UVec2, max_chunk_size: u32) -> impl Iterator<Item=ChunkInfo>`
- `ChunkInfo` contains: `offset: UVec2`, `size: UVec2`, `tile_bounds: UVec2`

#### Step 1.2: Split render_splats into phases
**Files:** `crates/brush-render/src/render.rs`

Refactor `render_splats` into:
1. `project_and_sort_splats()` - Steps 1-3 (global, runs once)
   - Returns: `projected_splats`, `global_from_compact_gid`, `num_visible`
2. `render_chunk()` - Steps 4-9 (per-chunk)
   - Takes: chunk info, projected_splats, global_from_compact_gid
   - Returns: writes directly to output image region

#### Step 1.3: Update uniforms for per-chunk rendering
**Files:** `crates/brush-render/src/shaders/helpers.wgsl`, `crates/brush-render/src/render.rs`

Add to `RenderUniforms`:
```wgsl
chunk_offset: vec2u,    // Pixel offset of this chunk in full image
chunk_size: vec2u,      // Size of this chunk in pixels
```

#### Step 1.4: Update intersection mapping shader
**Files:** `crates/brush-render/src/shaders/map_gaussian_to_intersects.wgsl`

Modify to use chunk-relative tile coordinates:
- Compute tile bounding box relative to chunk offset
- Clamp to chunk tile bounds instead of full image tile bounds

#### Step 1.5: Update rasterize shader
**Files:** `crates/brush-render/src/shaders/rasterize.wgsl`

Modify pixel coordinate calculation:
- Add chunk offset when writing to output image
- Use chunk-relative tile coordinates for tile offset lookups

#### Step 1.6: Update forward pass orchestration
**Files:** `crates/brush-render/src/render.rs`

Main loop structure:
```rust
fn render_splats(...) {
    // Global projection (once)
    let (projected_splats, global_from_compact_gid, num_visible) = 
        project_and_sort_splats(...);
    
    // Allocate full output image
    let out_img = create_tensor([img_size.y, img_size.x, out_dim], ...);
    
    // Per-chunk rendering
    for chunk in iter_chunks(img_size, 1024) {
        // Per-chunk intersection mapping
        let (chunk_tile_offsets, chunk_compact_gid_from_isect, chunk_num_intersections) = 
            compute_chunk_intersections(&chunk, &projected_splats, &global_from_compact_gid);
        
        // Per-chunk rasterization (writes to out_img at chunk offset)
        rasterize_chunk(&chunk, &out_img, ...);
    }
    
    return (out_img, RenderAux { ... });
}
```

### Phase 2: Refactor Backward Pass for Chunking with Recomputation

#### Step 2.1: Update GaussianBackwardState
**Files:** `crates/brush-render-bwd/src/burn_glue.rs`

Remove per-chunk data that will be recomputed:
```rust
pub struct GaussianBackwardState<B: Backend> {
    // Keep these (global, computed once)
    pub means: FloatTensor<B>,
    pub quats: FloatTensor<B>,
    pub log_scales: FloatTensor<B>,
    pub raw_opac: FloatTensor<B>,
    pub out_img: FloatTensor<B>,
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    
    // REMOVE these (will be recomputed per-chunk):
    // pub compact_gid_from_isect: IntTensor<B>,  
    // pub tile_offsets: IntTensor<B>,
    
    pub render_mode: SplatRenderMode,
    pub sh_degree: u32,
    pub img_size: glam::UVec2,  // Add this to know full image size
}
```

#### Step 2.2: Update RenderAux
**Files:** `crates/brush-render/src/render_aux.rs`

Similarly remove per-chunk data:
```rust
pub struct RenderAux<B: Backend> {
    pub projected_splats: FloatTensor<B>,
    pub uniforms_buffer: IntTensor<B>,
    pub global_from_compact_gid: IntTensor<B>,
    pub visible: FloatTensor<B>,
    pub img_size: glam::UVec2,
    // REMOVE: num_intersections, tile_offsets, compact_gid_from_isect
}
```

#### Step 2.3: Extract reusable intersection computation
**Files:** `crates/brush-render/src/render.rs` (or new file `crates/brush-render/src/intersect.rs`)

Create a shared function that can be called from both forward and backward:
```rust
pub fn compute_chunk_intersections(
    chunk: &ChunkInfo,
    projected_splats: &FloatTensor<B>,
    global_from_compact_gid: &IntTensor<B>,
    uniforms_buffer: &IntTensor<B>,
    num_visible: &IntTensor<B>,
) -> (IntTensor<B>, IntTensor<B>, IntTensor<B>) // (tile_offsets, compact_gid_from_isect, num_intersections)
```

#### Step 2.4: Update backward pass to iterate chunks
**Files:** `crates/brush-render-bwd/src/render_bwd.rs`

```rust
fn render_splats_bwd(state: GaussianBackwardState<Self>, v_output: FloatTensor<Self>) -> SplatGrads<Self> {
    // Allocate global gradient accumulators (once)
    let v_grads = zeros([num_points, 8], ...);
    let v_opacs = zeros([num_points], ...);
    let v_refines = zeros([num_points], ...);
    
    // Per-chunk backward rasterization
    for chunk in iter_chunks(state.img_size, 1024) {
        // RECOMPUTE intersection data for this chunk
        let (chunk_tile_offsets, chunk_compact_gid_from_isect, _) = 
            compute_chunk_intersections(&chunk, &state.projected_splats, ...);
        
        // Run backward rasterization for this chunk
        // Atomically accumulates into v_grads, v_opacs, v_refines
        rasterize_backwards_chunk(&chunk, &chunk_tile_offsets, ...);
    }
    
    // Global projection backward (once, after all chunks)
    project_backwards(..., &v_grads, ...);
    
    return SplatGrads { ... };
}
```

#### Step 2.5: Update rasterize_backwards shader
**Files:** `crates/brush-render-bwd/src/shaders/rasterize_backwards.wgsl`

Similar changes to forward rasterize shader:
- Add chunk offset when reading from output/v_output buffers
- Use chunk-relative tile coordinates

### Phase 3: Testing and Validation

#### Step 3.1: Update existing tests
**Files:** `crates/brush-render/src/tests/mod.rs`

- Update `renders_at_all` test to work with new structure
- Add test with 32x32 image (single chunk, no chunking needed)

#### Step 3.2: Add chunking-specific tests
**Files:** `crates/brush-render/src/tests/mod.rs`

Add tests:
1. `test_single_chunk` - Image fits in one chunk (e.g., 512x512)
2. `test_multiple_chunks_horizontal` - Wide image (e.g., 2048x512)
3. `test_multiple_chunks_vertical` - Tall image (e.g., 512x2048)
4. `test_multiple_chunks_both` - Large image (e.g., 2048x2048)
5. `test_chunk_boundary_gaussian` - Gaussian spanning chunk boundaries renders correctly
6. `test_backward_consistency` - Gradients match between chunked and non-chunked (for small images)

#### Step 3.3: Add gradient verification tests
**Files:** `crates/brush-render-bwd/src/tests/` (may need to create)

- Numerical gradient checking for chunked backward pass
- Verify gradients are identical for same image rendered chunked vs non-chunked

### Phase 4: Optimization (Optional, After Correctness)

#### Step 4.1: Parallel chunk processing
For inference (non-training), chunks could potentially be processed in parallel if there are no data dependencies. However, for training, sequential is fine since we're accumulating gradients.

#### Step 4.2: Memory pooling
Reuse intersection buffers across chunks to avoid repeated allocations.

## File Change Summary

### Modified Files:
1. `crates/brush-render/src/render.rs` - Main refactoring
2. `crates/brush-render/src/render_aux.rs` - Remove per-chunk fields
3. `crates/brush-render/src/shaders/helpers.wgsl` - Add chunk uniforms
4. `crates/brush-render/src/shaders/map_gaussian_to_intersects.wgsl` - Chunk-relative tiles
5. `crates/brush-render/src/shaders/rasterize.wgsl` - Chunk offset handling
6. `crates/brush-render-bwd/src/burn_glue.rs` - Update GaussianBackwardState
7. `crates/brush-render-bwd/src/render_bwd.rs` - Add chunk iteration with recomputation
8. `crates/brush-render-bwd/src/shaders/rasterize_backwards.wgsl` - Chunk offset handling
9. `crates/brush-render/src/tests/mod.rs` - Update and add tests

### Potentially New Files:
1. `crates/brush-render/src/chunking.rs` - Chunk iteration utilities (optional, could be in render.rs)

## Risk Assessment

### High Risk Areas:
1. **Chunk boundary handling** - Gaussians spanning chunks must contribute to all relevant chunks
2. **Gradient accumulation** - Atomic operations must correctly accumulate across chunks
3. **Uniform buffer updates** - Per-chunk uniforms must be correctly set

### Mitigation:
- Extensive testing with gaussians at chunk boundaries
- Comparison tests: chunked vs non-chunked rendering for small images
- Numerical gradient verification

## Implementation Order Recommendation

1. **Phase 1.1-1.3** - Infrastructure and uniforms (low risk, enables testing)
2. **Phase 1.4-1.6** - Forward pass chunking (medium risk)
3. **Phase 3.1-3.2** - Forward pass tests (validates Phase 1)
4. **Phase 2.1-2.5** - Backward pass chunking with recomputation (highest risk)
5. **Phase 3.3** - Backward pass tests (validates Phase 2)
6. **Phase 4** - Optimizations (only after everything works)

## Questions to Clarify

1. Should we support configurable chunk sizes, or always use 1024x1024?
2. For very small images (<1024x1024), should we skip chunking entirely for performance?
3. Should the backward pass store the `num_visible` tensor, or recompute it from `global_from_compact_gid.len()`?
