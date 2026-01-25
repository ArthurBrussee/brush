use crate::widget_3d::Widget3D;
use brush_process::slot::Slot;
use brush_render::{
    MainBackend, MainBackendBase, TextureMode, camera::Camera, gaussian_splats::Splats,
    render_splats,
};
use burn::tensor::{Tensor, TensorPrimitive};
use eframe::egui_wgpu::Renderer;
use egui::TextureId;
use egui::epaint::mutex::RwLock as EguiRwLock;
use glam::Vec3;
use std::num::NonZeroU64;
use std::sync::Arc;
use tokio::task;
use wgpu::{CommandEncoderDescriptor, TexelCopyBufferLayout, TextureViewDescriptor};

use eframe::{
    egui_wgpu::wgpu::util::DeviceExt as _,
    egui_wgpu::{self, wgpu},
};

pub struct Custom3d {
    angle: f32,
}

impl Custom3d {
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
        // Get the WGPU render state from the eframe creation context. This can also be retrieved
        // from `eframe::Frame` when you don't have a `CreationContext` available.
        let wgpu_render_state = cc.wgpu_render_state.as_ref()?;

        let device = &wgpu_render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("custom3d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./custom3d_wgpu_shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("custom3d"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("custom3d"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("custom3d"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: None,
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu_render_state.target_format.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("custom3d"),
            contents: bytemuck::cast_slice(&[0.0_f32; 4]), // 16 bytes aligned!
            // Mapping at creation (as done by the create_buffer_init utility) doesn't require us to to add the MAP_WRITE usage
            // (this *happens* to workaround this bug )
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom3d"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Because the graphics pipeline must have the same lifetime as the egui render pass,
        // instead of storing the pipeline in our `Custom3D` struct, we insert it into the
        // `paint_callback_resources` type map, which is stored alongside the render pass.
        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(TriangleRenderResources {
                pipeline,
                bind_group,
                uniform_buffer,
            });

        Some(Self { angle: 0.0 })
    }
}

impl crate::DemoApp for Custom3d {
    fn demo_ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // TODO(emilk): Use `ScrollArea::inner_margin`
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("The triangle is being painted using ");
                    ui.hyperlink_to("WGPU", "https://wgpu.rs");
                    ui.label(" (Portable Rust graphics API awesomeness)");
                });
                ui.label(
                    "It's not a very impressive demo, but it shows you can embed 3D inside of egui.",
                );

                egui::Frame::canvas(ui.style()).show(ui, |ui| {
                    self.custom_painting(ui);
                });
                ui.label("Drag to rotate!");
                ui.add(egui_demo_lib::egui_github_link_file!());
            });
        });
    }
}

// Callbacks in egui_wgpu have 3 stages:
// * prepare (per callback impl)
// * finish_prepare (once)
// * paint (per callback impl)
//
// The prepare callback is called every frame before paint and is given access to the wgpu
// Device and Queue, which can be used, for instance, to update buffers and uniforms before
// rendering.
// If [`egui_wgpu::Renderer`] has [`egui_wgpu::FinishPrepareCallback`] registered,
// it will be called after all `prepare` callbacks have been called.
// You can use this to update any shared resources that need to be updated once per frame
// after all callbacks have been processed.
//
// On both prepare methods you can use the main `CommandEncoder` that is passed-in,
// return an arbitrary number of user-defined `CommandBuffer`s, or both.
// The main command buffer, as well as all user-defined ones, will be submitted together
// to the GPU in a single call.
//
// The paint callback is called after finish prepare and is given access to egui's main render pass,
// which can be used to issue draw commands.
struct CustomTriangleCallback {
    angle: f32,
}

impl egui_wgpu::CallbackTrait for CustomTriangleCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &TriangleRenderResources = resources.get().unwrap();
        resources.prepare(device, queue, self.angle);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &TriangleRenderResources = resources.get().unwrap();
        resources.paint(render_pass);
    }
}

impl Custom3d {
    fn custom_painting(&mut self, ui: &mut egui::Ui) {
        let (rect, response) =
            ui.allocate_exact_size(egui::Vec2::splat(300.0), egui::Sense::drag());

        self.angle += response.drag_motion().x * 0.01;
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            CustomTriangleCallback { angle: self.angle },
        ));
    }
}

struct TriangleRenderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

impl TriangleRenderResources {
    fn prepare(&self, _device: &wgpu::Device, queue: &wgpu::Queue, angle: f32) {
        // Update our uniform buffer with the angle from the UI
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[angle, 0.0, 0.0, 0.0]),
        );
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Draw our triangle!
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

// #[derive(Clone)]
// pub struct RenderRequest {
//     pub slot: Slot<Splats<MainBackend>>,
//     pub frame: usize,
//     pub camera: Camera,
//     pub img_size: glam::UVec2,
//     pub background: Vec3,
//     pub splat_scale: Option<f32>,
//     pub ctx: egui::Context,
//     /// Model transform for the 3D overlay (grid, axes).
//     pub model_transform: glam::Affine3A,
//     /// Opacity of the grid overlay (0.0 = hidden, 1.0 = fully visible).
//     pub grid_opacity: f32,
// }

// pub struct SplatBackbuffer {
//     texture: wgpu::Texture,
//     texture_id: TextureId,
//     device: wgpu::Device,
//     queue: wgpu::Queue,
//     widget_3d: Arc<Widget3D>,
// }

// impl SplatBackbuffer {
//     pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
//         // Start with a dummy texture
//         let texture = create_texture(glam::uvec2(64, 64), &device);
//         let id = renderer.write().register_native_texture(
//             &device,
//             &texture.create_view(&TextureViewDescriptor::default()),
//             wgpu::FilterMode::Linear,
//         );
//         let widget_3d = Arc::new(Widget3D::new(device.clone(), queue.clone()));

//         Some(Self {
//             texture,
//             texture_id: id,
//             device,
//             queue,
//             widget_3d,
//         })
//     }

//     /// Submit a render request. Spawns an async task to do the rendering.
//     pub fn submit(&mut self, req: RenderRequest) {
//         if req.img_size.x <= 8 || req.img_size.y <= 8 {
//             return;
//         }

//         // Check resizing. This is done sync, as it requires a renderer lock...
//         let needs_resize =
//             self.texture.width() != req.img_size.x || self.texture.height() != req.img_size.y;
//         if needs_resize {
//             // TODO: Restore this.
//             // let client = WgpuRuntime::client(&req.);
//             // client.memory_cleanup();
//             self.texture = create_texture(req.img_size, &self.device);
//             self.renderer.write().update_egui_texture_from_wgpu_texture(
//                 &self.device,
//                 &self.texture.create_view(&TextureViewDescriptor::default()),
//                 wgpu::FilterMode::Linear,
//                 self.texture_id,
//             );
//         }

//         let texture = self.texture.clone();
//         let device = self.device.clone();
//         let queue = self.queue.clone();

//         let camera = req.camera.clone();
//         let img_size = req.img_size;
//         let background = req.background;
//         let splat_scale = req.splat_scale;

//         let widget = self.widget_3d.clone();

//         task::spawn(async move {
//             let splats = req.slot.clone_main().await;

//             if let Some(splats) = splats {
//                 let (image, _) = render_splats(
//                     splats.clone(),
//                     &camera,
//                     img_size,
//                     background,
//                     splat_scale,
//                     TextureMode::Packed,
//                 )
//                 .await;

//                 copy_to_texture(image, &texture, &device, &queue);

//                 if req.grid_opacity > 0.0 {
//                     widget.render_to_texture(
//                         &req.camera,
//                         req.model_transform,
//                         req.img_size,
//                         &texture,
//                         req.grid_opacity,
//                     );
//                 }

//                 req.ctx.request_repaint();
//             }
//         });
//     }

//     pub fn id(&self) -> TextureId {
//         self.texture_id
//     }
// }

// impl egui_wgpu::CallbackTrait for SplatBackbuffer {}

// impl SplatBackbuffer {
//     fn custom_painting(&mut self, ui: &mut egui::Ui) {
//         let (rect, response) =
//             ui.allocate_exact_size(egui::Vec2::splat(300.0), egui::Sense::drag());

//         self.angle += response.drag_motion().x * 0.01;
//         ui.painter().add(egui_wgpu::Callback::new_paint_callback(
//             rect,
//             CustomTriangleCallback { angle: self.angle },
//         ));
//     }
// }

// fn create_texture(size: glam::UVec2, device: &wgpu::Device) -> wgpu::Texture {
//     device.create_texture(&wgpu::TextureDescriptor {
//         label: Some("Splat backbuffer"),
//         size: wgpu::Extent3d {
//             width: size.x,
//             height: size.y,
//             depth_or_array_layers: 1,
//         },
//         mip_level_count: 1,
//         sample_count: 1,
//         dimension: wgpu::TextureDimension::D2,
//         format: wgpu::TextureFormat::Rgba8Unorm,
//         usage: wgpu::TextureUsages::TEXTURE_BINDING
//             | wgpu::TextureUsages::COPY_DST
//             | wgpu::TextureUsages::RENDER_ATTACHMENT,
//         view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
//     })
// }

// fn copy_to_texture(
//     img: Tensor<MainBackend, 3>,
//     texture: &wgpu::Texture,
//     device: &wgpu::Device,
//     queue: &wgpu::Queue,
// ) {
//     let [height, width, c] = img.dims();
//     let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
//         label: Some("splat backbuffer encoder"),
//     });

//     let padded_shape = vec![height, width.div_ceil(64) * 64, c];

//     let img_prim = img.into_primitive().tensor();
//     let fusion_client = img_prim.client.clone();
//     let img = fusion_client.resolve_tensor_float::<MainBackendBase>(img_prim);
//     let img: Tensor<MainBackendBase, 3> = Tensor::from_primitive(TensorPrimitive::Float(img));

//     // Pad if needed (WebGPU requires bytes_per_row divisible by 256)
//     let img = if width % 64 != 0 {
//         let padded: Tensor<MainBackendBase, 3> = Tensor::zeros(&padded_shape, &img.device());
//         padded.slice_assign([0..height, 0..width], img)
//     } else {
//         img
//     };

//     let img = img.into_primitive().tensor();
//     let client = &img.client;
//     let img_res_handle = client.get_resource(img.handle.clone().binding());
//     client.flush();

//     let bytes_per_row = Some(4 * padded_shape[1] as u32);

//     encoder.copy_buffer_to_texture(
//         wgpu::TexelCopyBufferInfo {
//             buffer: &img_res_handle.resource().buffer,
//             layout: TexelCopyBufferLayout {
//                 offset: img_res_handle.resource().offset,
//                 bytes_per_row,
//                 rows_per_image: None,
//             },
//         },
//         wgpu::TexelCopyTextureInfo {
//             texture,
//             mip_level: 0,
//             origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
//             aspect: wgpu::TextureAspect::All,
//         },
//         wgpu::Extent3d {
//             width: width as u32,
//             height: height as u32,
//             depth_or_array_layers: 1,
//         },
//     );

//     queue.submit([encoder.finish()]);
// }
