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
    _padding: [f32; 3], // Padding for alignment
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

pub struct Widget3D {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
    up_axis_vertex_buffer: wgpu::Buffer,
    up_axis_vertex_count: u32,
}

impl Widget3D {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Widget 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/widget_3d.wgsl").into()),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Widget 3D Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout and bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Widget 3D Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, // Fragment needs access for grid_opacity
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

        // Create render pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Widget 3D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

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
                    format: wgpu::TextureFormat::Rgba8Unorm,
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create geometry
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
            device,
            queue,
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
        let color = [0.3, 0.3, 0.3, 0.8]; // Semi-transparent gray

        // Create grid lines in XZ plane (Y=0) for OpenCV coordinates
        // This creates a ground plane since Y is down in OpenCV
        let mut i = -size;
        while i <= size {
            // Lines parallel to X axis
            vertices.push(Vertex {
                position: [-size, 0.0, i],
                color,
            });
            vertices.push(Vertex {
                position: [size, 0.0, i],
                color,
            });

            // Lines parallel to Z axis
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
        let mut vertices = Vec::new();
        let length = 1.5;

        // Single blue line pointing up (negative Y in OpenCV coordinates)
        vertices.push(Vertex {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 0.5, 1.0, 1.0], // Light blue
        });
        vertices.push(Vertex {
            position: [0.0, -length, 0.0], // Negative Y is up
            color: [0.0, 0.5, 1.0, 1.0],   // Light blue
        });

        (vertices, 2)
    }

    pub fn render_to_texture(
        &self,
        camera: &brush_render::camera::Camera,
        model_transform: glam::Affine3A,
        size: glam::UVec2,
        target_texture: &wgpu::Texture,
        grid_opacity: f32,
    ) {
        let output_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Widget 3D Depth Texture"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Use perspective_lh since camera uses +Z as forward
        // But flip Y since camera uses Y-down while perspective_lh uses Y-up
        let aspect = size.x as f32 / size.y as f32;
        let proj_matrix = Mat4::perspective_lh(camera.fov_y as f32, aspect, 0.1, 1000.0);

        // Y-flip to convert from Y-up to Y-down
        let y_flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));

        // The camera already has model transform baked in
        // To get world-space view, we need to undo the model transform by applying its inverse
        let view_matrix = camera.world_to_local();
        let world_view = Mat4::from(view_matrix) * Mat4::from(model_transform.inverse());

        // Apply Y flip and combine with projection
        let view_proj = proj_matrix * y_flip * world_view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            grid_opacity,
            _padding: [0.0; 3],
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Render
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Widget 3D Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Widget 3D Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Load existing content instead of clearing
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            // Draw grid
            render_pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
            render_pass.draw(0..self.grid_vertex_count, 0..1);

            // Draw up axis
            render_pass.set_vertex_buffer(0, self.up_axis_vertex_buffer.slice(..));
            render_pass.draw(0..self.up_axis_vertex_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
