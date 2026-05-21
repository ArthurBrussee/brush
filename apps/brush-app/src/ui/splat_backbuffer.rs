use brush_async::{Actor, AsyncMap};
use brush_process::slot::Slot;
use brush_render::{
    TextureMode, burn_glue::resolve_to_cube_float, camera::Camera, gaussian_splats::Splats,
    render_splats,
};
use burn::tensor::Tensor;
use egui::Rect;
use glam::{UVec2, Vec3};

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

pub struct SplatBackbuffer {
    pipe: AsyncMap<RenderRequest, Tensor<3>>,
}

impl SplatBackbuffer {
    pub fn new(state: &eframe::egui_wgpu::RenderState) -> Self {
        // Register splat backbuffer resources
        state
            .renderer
            .write()
            .callback_resources
            .insert(SplatBackbufferResources::new(
                &state.device,
                state.target_format,
            ));

        // The viewer renders on its own OS thread, separate from the
        // trainer's `ui-process` actor, so training and preview render
        // genuinely run in parallel. Sharing tensors across the two
        // threads (the trainer's splats are rendered here) is safe since
        // burn's shared-tensor handling (tracel-ai/burn#4962). The actor
        // is owned by `pipe`; dropping the backbuffer shuts the thread down.
        let actor = Actor::new("viewer");

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
                image
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

        if let Some(image) = self.pipe.latest() {
            let shape = image.shape();
            let img_height = shape[0] as u32;
            let img_width = shape[1] as u32;

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    SplatBackbufferPainter {
                        last_img: image,
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
        }
    }
}

struct SplatBackbufferPainter {
    last_img: Tensor<3>,
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
        let prim_tensor = resolve_to_cube_float(self.last_img.clone());
        let img_res_handle = prim_tensor
            .client
            .get_resource(prim_tensor.handle)
            .expect("Failed to get img resource");

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
        render_pass.draw(0..3, 0..1);
    }
}
