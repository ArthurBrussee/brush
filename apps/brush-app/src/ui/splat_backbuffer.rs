use brush_async::{Actor, AsyncMap};
use brush_process::slot::Slot;
use brush_render::{TextureMode, camera::Camera, gaussian_splats::Splats, render_splats};
use egui::Rect;
use glam::{UVec2, Vec3};
use std::sync::Arc;

use eframe::egui_wgpu::{self, CallbackTrait, wgpu};

#[derive(Clone)]
struct RenderRequest {
    splats: Slot<Splats>,
    ctx: egui::Context,
    state: LastRenderState,
}

#[derive(Clone, PartialEq)]
struct LastRenderState {
    frame: usize,
    camera: Camera,
    background: Vec3,
    splat_scale: Option<f32>,
    img_size: UVec2,
}

/// A rendered frame on its way to the screen.
///
/// Training runs on its own device, so there is no way to bind its buffer into
/// the viewer's bind group. The pixels come back through the host instead:
/// packed RGBA8, four bytes per pixel, so there is nothing to convert on the
/// way. The readback is awaited off the render thread, so no frame is dropped
/// for it.
#[derive(Clone)]
struct Frame {
    width: u32,
    height: u32,
    pixels: Arc<Vec<u8>>,
}

pub struct SplatBackbuffer {
    pipe: AsyncMap<RenderRequest, Frame>,
}

impl SplatBackbuffer {
    pub fn new(state: &eframe::egui_wgpu::RenderState) -> Self {
        // The viewer gets its own thread, deliberately. Reading a frame back
        // from cubecl-metal blocks the calling thread until the GPU drains
        // (cubecl-metal's `read` flushes and waits synchronously, then
        // memcpys), and cubecl assigns one stream per thread by default. On
        // the process actor that would mean every displayed frame blocks the
        // thread driving training, on training's own stream.
        let actor = Actor::new("splat-view");
        // Register splat backbuffer resources
        state
            .renderer
            .write()
            .callback_resources
            .insert(SplatBackbufferResources::new(
                &state.device,
                state.target_format,
            ));

        let pipe = AsyncMap::new(
            actor,
            async move |req: &RenderRequest| {
                let (image, _) = render_splats(
                    req.splats.get(req.state.frame).unwrap(),
                    &req.state.camera,
                    req.state.img_size,
                    req.state.background,
                    req.state.splat_scale,
                    TextureMode::Packed,
                )
                .await;

                let shape = image.shape();
                let (height, width) = (shape[0] as u32, shape[1] as u32);

                let data = image
                    .into_data_async()
                    .await
                    .expect("Failed to read back frame");

                Frame {
                    width,
                    height,
                    pixels: Arc::new(data.into_bytes().to_vec()),
                }
            },
            |req: &RenderRequest| req.ctx.request_repaint(),
        );

        Self { pipe }
    }

    pub fn paint(
        &self,
        rect: Rect,
        ui: &egui::Ui,
        splats: &Slot<Splats>,
        camera: &Camera,
        frame: usize,
        background: Vec3,
        splat_scale: Option<f32>,
        splats_dirty: bool,
    ) {
        // Calculate pixel size for rendering
        let ppp = ui.ctx().pixels_per_point();
        let img_size = UVec2::new(
            (rect.width() * ppp).round() as u32,
            (rect.height() * ppp).round() as u32,
        );

        // Check if we need to re-render
        let current_state = LastRenderState {
            frame,
            camera: *camera,
            background,
            splat_scale,
            img_size,
        };

        let dirty = splats_dirty
            || self.pipe.last_request().map(|r| r.state) != Some(current_state.clone());

        if dirty && !splats.is_empty() {
            self.pipe.request(RenderRequest {
                splats: splats.clone(),
                ctx: ui.ctx().clone(),
                state: current_state,
            });
        }

        if let Some(frame) = self.pipe.latest() {
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    SplatBackbufferPainter { frame },
                ));
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    img_width: u32,
    img_height: u32,
}

pub struct SplatBackbufferResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    // Per-frame bind group - created in prepare() with the current tensor buffer
    bind_group: Option<wgpu::BindGroup>,
    // Destination for copied frames. Kept around between frames and only
    // reallocated when the window resizes.
    upload_buffer: Option<wgpu::Buffer>,
}

impl SplatBackbufferResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Splat Backbuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/splat_backbuffer.wgsl").into()),
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat Backbuffer Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Splat Backbuffer Bind Group Layout"),
            entries: &[
                // Uniform buffer for image dimensions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Storage buffer for image data (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Splat Backbuffer Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Splat Backbuffer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers - using fullscreen triangle trick
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group: None,
            upload_buffer: None,
        }
    }

    /// Make sure the upload buffer holds at least `size` bytes, allocating a
    /// new one if the current one is too small.
    fn reserve_upload_buffer(&mut self, device: &wgpu::Device, size: u64) {
        let fits = self
            .upload_buffer
            .as_ref()
            .is_some_and(|b| b.size() >= size);
        if !fits {
            self.upload_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Splat Backbuffer Upload Buffer"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }
}

struct SplatBackbufferPainter {
    frame: Frame,
}

impl CallbackTrait for SplatBackbufferPainter {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = resources.get_mut::<SplatBackbufferResources>() else {
            return Vec::new();
        };

        // Update uniform buffer with image dimensions
        queue.write_buffer(
            &res.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                img_width: self.frame.width,
                img_height: self.frame.height,
            }]),
        );

        res.reserve_upload_buffer(device, self.frame.pixels.len() as u64);
        let img_buffer = res.upload_buffer.as_ref().expect("just reserved");
        queue.write_buffer(img_buffer, 0, &self.frame.pixels);

        // Create a new bind group with the current image buffer
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Splat Backbuffer Bind Group"),
            layout: &res.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: img_buffer.as_entire_binding(),
                },
            ],
        });

        res.bind_group = Some(bind_group);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = callback_resources.get::<SplatBackbufferResources>() else {
            return;
        };

        let Some(bind_group) = res.bind_group.as_ref() else {
            return;
        };

        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
