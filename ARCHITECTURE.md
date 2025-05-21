# Brush: System Architecture

Brush is a cutting-edge 3D reconstruction engine that leverages the power of Gaussian splatting techniques to create high-fidelity 3D models from input imagery.

## Project Goals and Strengths

Brush aims to provide a powerful, versatile, and accessible solution for 3D reconstruction. Its development is guided by several core goals, which translate into key strengths of the engine:

*   **Real-Time Interactivity**: A primary objective is to achieve real-time rendering and interaction with the reconstructed 3D scenes. This allows for immediate feedback and a fluid user experience when navigating and inspecting complex models. Gaussian splatting, combined with efficient GPU utilization via `wgpu`, is crucial for this.

*   **Cross-Platform Compatibility**: Brush is designed from the ground up to operate across a wide array of platforms. By leveraging Rust, `wgpu` (for graphics), and `burn` (for ML computations with a `wgpu` backend), the engine targets:
    *   **Desktop**: Windows, macOS, and Linux.
    *   **Web**: Via WebAssembly, enabling in-browser 3D reconstruction and viewing.
    *   **Mobile**: Specifically Android, with potential for other mobile platforms.

*   **Dependency-Free Binaries**: Thanks to Rust's compilation model, Brush aims to produce single, self-contained executable files where possible, especially for desktop applications. This simplifies distribution and reduces the likelihood of runtime issues caused by missing external dependencies on user systems.

*   **Live Training Visualization**: To provide insight into the reconstruction process and facilitate debugging and parameter tuning, Brush incorporates live visualization of the training progress. Users can observe how the Gaussian splat model evolves and improves over iterations, which is often handled by `brush-app` in conjunction with `brush-ui` and `egui`.

These features make Brush a forward-looking engine suitable for a broad range of applications, from academic research to interactive content creation.

## Core Technology: Gaussian Splatting

Gaussian splatting is a novel rasterization technique that allows for the rendering of photorealistic 3D scenes. Instead of traditional polygon meshes, scenes are represented by a collection of 3D Gaussians. This approach enables fast rendering, high-quality results, and efficient representation of complex geometries and view-dependent effects.

## Key Technologies

Brush is built upon a foundation of modern, high-performance technologies, carefully chosen to enable its advanced capabilities and cross-platform reach.

*   **Rust**: The core of Brush is implemented in Rust, a systems programming language renowned for its performance, memory safety, and concurrency features. Rust's strong type system and ownership model eliminate many common bugs at compile time, making it ideal for developing robust and reliable software. Its excellent tooling and growing ecosystem further support the development of complex applications like Brush.

*   **`wgpu`**: For graphics rendering and computation, Brush utilizes `wgpu`, a modern, cross-platform graphics API written in Rust. `wgpu` provides an abstraction over native graphics libraries such as Vulkan, Metal, DirectX 12, OpenGL ES, and WebGL. This allows Brush to run seamlessly on various desktop operating systems (Windows, macOS, Linux), in web browsers (via WebAssembly compilation), and on mobile platforms like Android. The `brush-wgsl` crate specifically houses the shader programs written in WGSL that `wgpu` executes on the GPU.

*   **WGSL (WebGPU Shading Language)**: WGSL is the shading language for WebGPU. It's designed to be a modern, safe, and performant language for programming GPUs. All shaders used by `wgpu` in Brush (for rendering, splatting logic, and potentially other GPGPU tasks) are written in WGSL. This ensures compatibility and optimal performance across all platforms supported by `wgpu`.

*   **`burn`**: Brush leverages the `burn` deep learning framework, also written in Rust. `burn` is used for the machine learning aspects of the Gaussian splatting process, particularly for training and optimizing the parameters of the 3D Gaussians. `burn` is designed for flexibility and performance, supporting various backends (CPU, GPU via CUDA, and importantly, `wgpu` for cross-platform GPU acceleration). This integration allows Brush to perform computationally intensive training tasks efficiently across different hardware.

*   **`egui`**: For creating graphical user interfaces (GUIs) in tools like `brush-app`, Brush employs `egui`. `egui` is an easy-to-use, immediate mode GUI library in Rust. It's highly portable and can be integrated with `wgpu` for rendering, making it a natural fit for Brush's cross-platform ambitions. This allows for the development of interactive tools for visualizing datasets, monitoring training progress, and inspecting reconstructed 3D models.

The strategic combination of these technologies enables Brush to deliver high-performance 3D reconstruction, while maintaining portability across desktops, web browsers, and mobile devices.

## Core Components (Crates)

Brush is architected as a collection of specialized Rust crates, each responsible for a distinct aspect of the 3D reconstruction pipeline. This modular design promotes code organization, reusability, and maintainability.

*   **`brush-app`**: Likely the main application crate that orchestrates the overall workflow, integrating functionalities from other crates to provide a cohesive user experience, possibly for a GUI or interactive application.
*   **`brush-cli`**: Provides the command-line interface for Brush. This allows users to interact with the engine, run reconstruction tasks, and manage datasets through terminal commands.
*   **`brush-dataset`**: Manages data loading, processing, and augmentation. This crate would be responsible for handling input images, camera parameters, and other data required for the reconstruction process.
*   **`brush-kernel`**: Contains core computational routines and algorithms, possibly low-level operations or shared utilities used by other Brush crates. This could include mathematical functions or data structures fundamental to Gaussian splatting.
*   **`brush-prefix-sum`**: Implements prefix sum (scan) operations. This is a common parallel primitive often used in graphics and data processing for tasks like stream compaction or building data structures on the GPU.
*   **`brush-sort`**: Provides sorting algorithms, likely optimized for the types of data encountered in the rendering or training pipeline (e.g., sorting Gaussians by depth).
*   **`brush-wgsl`**: Contains WGSL (WebGPU Shading Language) shaders. `wgpu` uses WGSL for defining GPU programs, so this crate would house the shader code for rendering, and potentially for compute tasks.
*   **`brush-render`**: Implements the forward rendering pipeline for the Gaussian splatting. This crate takes the 3D Gaussian representation and generates 2D images.
*   **`brush-render-bwd`**: Likely implements the backward pass (gradient computation) for the rendering process. This is crucial for training and optimizing the parameters of the 3D Gaussians.
*   **`brush-train`**: Manages the training and optimization loop. This crate would use the rendering outputs and ground truth data to refine the Gaussian splat parameters, likely employing techniques from `brush-render-bwd` and `burn`.
*   **`brush-ui`**: Provides user interface components, possibly using a Rust UI framework (like Egui, Iced) or bindings to web technologies, to interact with the `brush-app`.
*   **`brush-vfs`**: Implements a virtual file system. This can be useful for abstracting file access across different platforms (e.g., native file system, web storage) or for managing assets packed within the application.

### External Helper Crates

Brush also leverages several external or more specialized crates:

*   **`colmap-reader`**: A utility to read and parse data from COLMAP, a popular open-source Structure-from-Motion (SfM) and Multi-View Stereo (MVS) software. This is likely used to ingest camera poses and sparse point clouds.
*   **`lpips`**: Implements the Learned Perceptual Image Patch Similarity (LPIPS) metric. This is a common metric used in image synthesis and computational photography to evaluate the perceptual similarity between two images, often used as a loss function during training.
*   **`rrfd`**: Likely stands for "Randomized Resampling for Fast Denoising" or a similar technique. This could be a component used for improving the quality of the input images or the rendered output by reducing noise.
*   **`sync-span`**: Provides utilities for synchronization and tracing across asynchronous operations or spans of code, which is helpful for performance analysis and debugging in complex systems.

## Data Flow

The Brush engine processes data in a pipeline, transforming input imagery and camera information into a fully reconstructed 3D scene represented by Gaussian splats. This process involves data ingestion, training/optimization, and rendering, coordinated by various specialized crates.

1.  **Input Data Ingestion and Preparation**:
    *   The process typically begins with a dataset consisting of multiple images of a scene and corresponding camera parameters (poses, intrinsics).
    *   The `brush-dataset` crate is responsible for loading and managing this input data. It may perform initial processing steps like image decoding or normalization.
    *   Camera poses and sparse point clouds, potentially derived from Structure-from-Motion (SfM) software like COLMAP, are read using the `colmap-reader` crate. This provides the initial geometric context for the scene.
    *   The `brush-vfs` crate might be used at this stage to abstract access to dataset files, especially if they are located in different storage environments (e.g., local disk, cloud storage for web versions).

2.  **Initialization of Gaussian Splats**:
    *   Initially, 3D Gaussian splats are often initialized from the sparse point cloud obtained from SfM (e.g., via `colmap-reader`) or other methods. Each Gaussian has parameters like position, covariance (shape/size), color, and opacity.
    *   Core mathematical operations for manipulating these structures might reside in `brush-kernel`.

3.  **Training and Optimization Loop**:
    *   The `brush-train` crate orchestrates the core optimization loop. The goal is to adjust the parameters of each Gaussian splat so that when rendered, they accurately reproduce the input images from their respective camera viewpoints.
    *   **Forward Pass**: For each training iteration and for each input view:
        *   The current set of Gaussian splats is rendered by `brush-render`. This involves:
            *   Optionally, `brush-sort` may be used to sort Gaussians (e.g., by depth) for correct alpha blending and efficient rendering.
            *   `brush-prefix-sum` might be employed for parallelizing parts of the splatting rasterization process on the GPU.
            *   The actual rendering logic, converting 3D Gaussians to a 2D image, is executed on the GPU using shaders defined in `brush-wgsl` and managed by `wgpu`.
    *   **Loss Calculation**: The rendered image is compared against the ground truth input image. A loss function (or multiple) quantifies the difference. The `lpips` crate might be used here to provide a perceptually accurate loss metric.
    *   **Backward Pass**:
        *   `brush-render-bwd` calculates the gradients of the loss function with respect to the parameters of the Gaussian splats. This indicates how each parameter should change to reduce the loss.
    *   **Optimization**:
        *   The `burn` deep learning framework, utilizing its `wgpu` backend for GPU acceleration, takes these gradients and applies an optimization algorithm (e.g., Adam) to update the Gaussian splat parameters.
    *   This loop (forward pass, loss calculation, backward pass, optimization) is repeated for many iterations until the quality of the rendered images converges. `brush-kernel` might provide fundamental numerical operations used throughout this process.

4.  **Final Rendering and Visualization**:
    *   Once the Gaussian splats are optimized, they represent the final 3D scene.
    *   The `brush-render` crate can then be used to render this scene from novel camera viewpoints (not seen during training) to create new images or interactive visualizations.
    *   This rendering also uses `brush-wgsl` shaders and `wgpu` for GPU execution.
    *   The `brush-app` (potentially using `brush-ui` and `egui`) provides an interface for users to load models, trigger rendering, and view the 3D scene interactively. The `brush-cli` offers similar capabilities through a command-line interface.

Throughout this entire process, `sync-span` can be used to trace and profile operations, helping to identify performance bottlenecks in either the training or rendering stages.
