use brush_render::camera::Camera;
use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    grid_opacity: f32,
    _padding: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Resources for Widget3D stored in callback_resources.
pub struct Widget3DResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
    up_axis_vertex_buffer: wgpu::Buffer,
    up_axis_vertex_count: u32,
}

impl Widget3DResources {
    /// Create Widget3D resources. Call this once during initialization.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Widget 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/widget_3d.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Widget 3D Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Widget 3D Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Widget 3D Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Widget 3D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Pipeline without depth stencil - draws on top of egui content
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Widget 3D Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
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
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None, // No depth buffer - draw on top
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (grid_vertices, grid_vertex_count) = Self::create_grid_geometry();
        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let (up_axis_vertices, up_axis_vertex_count) = Self::create_up_axis_geometry();
        let up_axis_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Up Axis Vertex Buffer"),
            contents: bytemuck::cast_slice(&up_axis_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            grid_vertex_buffer,
            grid_vertex_count,
            up_axis_vertex_buffer,
            up_axis_vertex_count,
        }
    }

    fn create_grid_geometry() -> (Vec<Vertex>, u32) {
        let mut vertices = Vec::new();
        let size = 10.0;
        let step = 1.0;
        let color = [0.3, 0.3, 0.3, 0.8];

        let mut i = -size;
        while i <= size {
            vertices.push(Vertex {
                position: [-size, 0.0, i],
                color,
            });
            vertices.push(Vertex {
                position: [size, 0.0, i],
                color,
            });
            vertices.push(Vertex {
                position: [i, 0.0, -size],
                color,
            });
            vertices.push(Vertex {
                position: [i, 0.0, size],
                color,
            });
            i += step;
        }

        (vertices.clone(), vertices.len() as u32)
    }

    fn create_up_axis_geometry() -> (Vec<Vertex>, u32) {
        let vertices = vec![
            Vertex {
                position: [0.0, 0.0, 0.0],
                color: [0.0, 0.5, 1.0, 1.0],
            },
            Vertex {
                position: [0.0, -1.5, 0.0],
                color: [0.0, 0.5, 1.0, 1.0],
            },
        ];
        (vertices, 2)
    }

    /// Update uniforms for rendering.
    pub fn prepare(&self, queue: &wgpu::Queue, view_proj: Mat4, grid_opacity: f32) {
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            grid_opacity,
            _padding: [0.0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Issue draw commands to the provided render pass.
    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

        render_pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
        render_pass.draw(0..self.grid_vertex_count, 0..1);

        render_pass.set_vertex_buffer(0, self.up_axis_vertex_buffer.slice(..));
        render_pass.draw(0..self.up_axis_vertex_count, 0..1);
    }
}

/// Callback for rendering the 3D widget overlay via egui's paint system.
pub struct Widget3DCallback {
    pub camera: Camera,
    pub model_transform: glam::Affine3A,
    pub grid_opacity: f32,
}

impl egui_wgpu::CallbackTrait for Widget3DCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(widget_3d) = resources.get::<Widget3DResources>() else {
            return Vec::new();
        };

        let aspect = screen_descriptor.size_in_pixels[0] as f32
            / screen_descriptor.size_in_pixels[1] as f32;
        let proj_matrix = Mat4::perspective_lh(self.camera.fov_y as f32, aspect, 0.1, 1000.0);
        let y_flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        let view_matrix = self.camera.world_to_local();
        let world_view = Mat4::from(view_matrix) * Mat4::from(self.model_transform.inverse());
        let view_proj = proj_matrix * y_flip * world_view;

        widget_3d.prepare(queue, view_proj, self.grid_opacity);

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(widget_3d) = resources.get::<Widget3DResources>() else {
            return;
        };
        widget_3d.paint(render_pass);
    }
}
