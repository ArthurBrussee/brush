use brush_process::slot::Slot;
use brush_render::{
    MainBackend, MainBackendBase, TextureMode, camera::Camera, gaussian_splats::Splats,
    render_splats,
};
use burn::tensor::Tensor;
use egui::Rect;
use glam::{UVec2, Vec3};
use tokio::sync::mpsc;

use eframe::egui_wgpu::{self, CallbackTrait, wgpu};

/// Internal request sent to the async render worker.
#[derive(Clone)]
struct RenderRequest {
    slot: Slot<Splats<MainBackend>>,
    ctx: egui::Context,
    state: LastRenderState,
}

/// State used to track if we need to re-render.
#[derive(Clone, PartialEq)]
struct LastRenderState {
    frame: usize,
    camera: Camera,
    background: Vec3,
    splat_scale: Option<f32>,
    img_size: UVec2,
}

pub struct SplatBackbuffer {
    req_send: mpsc::UnboundedSender<RenderRequest>,
    img_rec: mpsc::Receiver<Tensor<MainBackend, 3>>,
    last_image: Option<Tensor<MainBackend, 3>>,
    last_state: Option<LastRenderState>,
}

impl SplatBackbuffer {
    pub fn new(state: &eframe::egui_wgpu::RenderState) -> Self {
        // Create channel for render requests
        let (req_send, req_rec) = mpsc::unbounded_channel();
        let (img_send, img_rec) = mpsc::channel(1);

        // Register splat backbuffer resources
        state
            .renderer
            .write()
            .callback_resources
            .insert(SplatBackbufferResources::new(
                &state.device,
                state.target_format,
            ));

        tokio::task::spawn(render_worker(req_rec, img_send));
        Self {
            req_send,
            img_rec,
            last_image: None,
            last_state: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        rect: Rect,
        ui: &egui::Ui,
        slot: &Slot<Splats<MainBackend>>,
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
            camera: camera.clone(),
            background,
            splat_scale,
            img_size,
        };

        let dirty = splats_dirty || self.last_state.as_ref() != Some(&current_state);

        if dirty {
            self.last_state = Some(current_state.clone());
            // Send request to worker (ignore send errors if channel closed)
            let _ = self.req_send.send(RenderRequest {
                slot: slot.clone(),
                state: current_state,
                ctx: ui.ctx().clone(),
            });
        }

        while let Ok(img) = self.img_rec.try_recv() {
            self.last_image = Some(img);
        }

        if let Some(image) = &self.last_image {
            let shape = image.shape();
            let img_height = shape.dims[0] as u32;
            let img_width = shape.dims[1] as u32;

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    SplatBackbufferPainter {
                        last_img: image.clone(),
                        img_width,
                        img_height,
                    },
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
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group: None,
        }
    }
}

struct SplatBackbufferPainter {
    last_img: Tensor<MainBackend, 3>,
    img_width: u32,
    img_height: u32,
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
                img_width: self.img_width,
                img_height: self.img_height,
            }]),
        );

        // Extract the wgpu buffer from the Burn tensor
        let last_img = self.last_img.clone().into_primitive().tensor();
        let prim_tensor = last_img
            .client
            .clone()
            .resolve_tensor_int::<MainBackendBase>(last_img);
        let img_res_handle = prim_tensor
            .client
            .get_resource(prim_tensor.handle.binding());

        // Create a new bind group with the current tensor buffer
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
                    resource: img_res_handle.resource().buffer.as_entire_binding(),
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
        render_pass.draw(0..3, 0..1); // 3 vertices for fullscreen triangle
    }
}

/// Async render worker that processes render requests.
async fn render_worker(
    mut receiver: mpsc::UnboundedReceiver<RenderRequest>,
    img_sender: mpsc::Sender<Tensor<MainBackend, 3>>,
) {
    loop {
        // Wait for at least one request and get latest.
        let Some(mut request) = receiver.recv().await else {
            break;
        };
        while let Ok(newer) = receiver.try_recv() {
            request = newer;
        }

        let image = request
            .slot
            .act(request.state.frame, async |splats| {
                let (image, _) = render_splats(
                    splats.clone(),
                    &request.state.camera,
                    request.state.img_size,
                    request.state.background,
                    request.state.splat_scale,
                    TextureMode::Packed,
                )
                .await;
                (splats, image)
            })
            .await;

        if let Some(image) = image {
            let _ = img_sender.send(image).await;
        }

        // Trigger egui repaint so the new texture gets picked up.
        request.ctx.request_repaint();
    }
}
