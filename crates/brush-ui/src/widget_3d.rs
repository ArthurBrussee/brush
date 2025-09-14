use glam::Mat4;
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
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
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
    axes_vertex_buffer: wgpu::Buffer,
    axes_vertex_count: u32,
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
                visibility: wgpu::ShaderStages::VERTEX,
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
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

        let (axes_vertices, axes_vertex_count) = Self::create_axes_geometry();
        let axes_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Axes Vertex Buffer"),
            contents: bytemuck::cast_slice(&axes_vertices),
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
            axes_vertex_buffer,
            axes_vertex_count,
            output_texture: None,
            output_view: None,
            texture_id: None,
            renderer,
        }
    }

    fn create_grid_geometry() -> (Vec<Vertex>, u32) {
        let mut vertices = Vec::new();
        let size = 10.0;
        let step = 1.0;
        let color = [0.3, 0.3, 0.3, 0.8]; // Semi-transparent gray

        // Create grid lines in XY plane (Z=0) for OpenCV coordinates (Y-down, Z-forward)
        let mut i = -size;
        while i <= size {
            // Lines parallel to X axis (horizontal lines)
            vertices.push(Vertex {
                position: [-size, i, 0.0],
                color,
            });
            vertices.push(Vertex {
                position: [size, i, 0.0],
                color,
            });

            // Lines parallel to Y axis (vertical lines)
            vertices.push(Vertex {
                position: [i, -size, 0.0],
                color,
            });
            vertices.push(Vertex {
                position: [i, size, 0.0],
                color,
            });

            i += step;
        }

        (vertices.clone(), vertices.len() as u32)
    }

    fn create_axes_geometry() -> (Vec<Vertex>, u32) {
        let mut vertices = Vec::new();
        let length = 2.0;

        // X axis - Red (right)
        vertices.push(Vertex {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
        });
        vertices.push(Vertex {
            position: [length, 0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
        });

        // Y axis - Green (down for OpenCV)
        vertices.push(Vertex {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],
        });
        vertices.push(Vertex {
            position: [0.0, length, 0.0], // Positive Y is down in OpenCV
            color: [0.0, 1.0, 0.0, 1.0],
        });

        // Z axis - Blue (forward into scene)
        vertices.push(Vertex {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 0.0, 1.0, 1.0],
        });
        vertices.push(Vertex {
            position: [0.0, 0.0, length], // Positive Z is forward in OpenCV
            color: [0.0, 0.0, 1.0, 1.0],
        });

        (vertices, 6)
    }

    pub fn render_to_texture(
        &mut self,
        camera: &brush_render::camera::Camera,
        model_transform: glam::Affine3A,
        size: glam::UVec2,
        target_texture: &wgpu::Texture,
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

        // Update uniforms - use same approach as splat rendering
        let view_matrix = camera.world_to_local(); // Same as splats: camera.world_to_local()
        let aspect = size.x as f32 / size.y as f32;
        let proj_matrix = Mat4::perspective_rh(camera.fov_y as f32, aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * Mat4::from(view_matrix);
        let model_view_proj = view_proj * Mat4::from(model_transform);

        let uniforms = Uniforms {
            view_proj: model_view_proj.to_cols_array_2d(),
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

            // Draw axes
            render_pass.set_vertex_buffer(0, self.axes_vertex_buffer.slice(..));
            render_pass.draw(0..self.axes_vertex_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
