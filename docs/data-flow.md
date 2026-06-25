# Data Flow

How raw input becomes trained splats, and where memory is spent along the way.

## End-to-end

```mermaid
flowchart LR
  SRC["DataSource<br/>path / URL / zip / .ply"] --> VFS{brush-vfs}
  VFS -->|directory| DIR["tokio::fs<br/>5MB BufReader"]
  VFS -->|zip| ZIP["InMemory:<br/>whole archive in RAM"]
  VFS -->|single ply| STREAM["streaming reader"]
  DIR --> FMT{format}
  ZIP --> FMT
  FMT -->|cameras.bin/.txt| COLMAP[colmap-reader]
  FMT -->|transforms.json| NERF[Nerfstudio]
  COLMAP --> SCENE
  NERF --> SCENE
  SCENE["Scene = Arc&lt;Vec&lt;SceneView&gt;&gt;<br/>each view = lazy LoadImage + Camera"]
  SCENE --> LOADER
  subgraph LOADER["SceneLoader (N actors × 2 tasks)"]
    DEC["decode (JPEG IDCT 1/2·1/4·1/8) → mask → resize"]
    PACK["pack → [H,W] u32 RGBA premultiplied"]
    CACHE["BatchCache (6 GiB native / 2 GiB wasm)"]
    DEC --> PACK --> CACHE
  end
  CACHE -->|mpsc(4)| TRAIN["train_stream loop"]
  TRAIN --> SPLATS["Splats on GPU"]
  SPLATS --> EXPORT["export_{iter}.ply<br/>(tokio::fs::write)"]
  SPLATS --> VIEW["Slot → live viewer"]
```

## Stages

1. **DataSource → VFS.** `brush-vfs` abstracts three backends: on-disk directory (lazy reads,
   5 MB buffered), in-memory ZIP (entire archive decompressed into
   `HashMap<Path, Arc<Vec<u8>>>` — `crates/brush-vfs/src/lib.rs:185-208`), and a one-shot
   streaming reader for a single `.ply`.
2. **Format parse.** COLMAP (binary or text) via `colmap-reader`, or Nerfstudio
   `transforms.json`. Produces a `Scene` of `SceneView`s; each holds a **lazy** `LoadImage`
   descriptor (`crates/brush-dataset/src/load_image.rs:11-19`) — pixels are not decoded yet.
3. **Load on demand.** `LoadImage::load()` reads bytes, decodes, optionally applies a mask, and
   resizes to `--max-resolution` (`load_image.rs:56-109`). JPEGs use `jpeg-decoder`'s
   IDCT scale-on-decode to land at 1/2, 1/4, or 1/8 size during decode (`:208-226`), saving
   4–16× on oversized images; other formats decode full-res then resize.
4. **Pack & cache.** The decoded image is packed to a GPU-ready `[H,W]` u32 RGBA buffer
   (premultiplied) and cached. A cache **hit** is a single buffer copy; a **miss** re-runs
   decode→pack (`crates/brush-dataset/src/scene_loader.rs:16-27`).
5. **Prefetch to trainer.** Batches flow through an `mpsc::channel(4)` — at most 4 batches
   ahead — so the loader naturally back-pressures (`scene_loader.rs:64-68`).
6. **Train → publish → export.** The trainer consumes a batch per step, updates splats,
   publishes a snapshot to the `Slot` for viewers, evaluates on a held-out split every
   `--eval-every`, and writes `.ply` checkpoints every `--export-every`
   (`crates/brush-process/src/train_stream.rs`).

## Memory characteristics (see [performance.md](./performance.md) for the full model)
- Images are **lazily decoded**, not bulk-loaded — good.
- The **packed-batch cache is bounded at 6 GiB** but has **no LRU eviction**: once full, new
  views bypass the cache and are re-decoded on every visit (`scene_loader.rs:42-53`).
- **Decode concurrency = CPU cores × 2** (`scene_loader.rs:73-78`); each in-flight task holds a
  full decoded image briefly, so peak decode memory scales with core count.
- ZIP inputs are held **entirely in RAM**.
