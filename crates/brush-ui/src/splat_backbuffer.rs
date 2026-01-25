use crate::widget_3d::Widget3D;
use brush_process::slot::Slot;
use brush_render::{
    MainBackend, MainBackendBase, TextureMode, camera::Camera, gaussian_splats::Splats,
    render_splats,
};
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::WgpuRuntime;
use eframe::egui_wgpu::Renderer;
use egui::TextureId;
use egui::epaint::mutex::RwLock as EguiRwLock;
use glam::Vec3;
use pollster::block_on;
use std::sync::Arc;
use wgpu::{CommandEncoderDescriptor, TexelCopyBufferLayout, TextureViewDescriptor};

#[derive(Clone)]
pub struct RenderRequest {
    pub slot: Slot<Splats<MainBackend>>,
    pub frame: usize,
    pub camera: Camera,
    pub img_size: glam::UVec2,
    pub background: Vec3,
    pub splat_scale: Option<f32>,
    pub ctx: egui::Context,
    /// Model transform for the 3D overlay (grid, axes).
    pub model_transform: glam::Affine3A,
    /// Opacity of the grid overlay (0.0 = hidden, 1.0 = fully visible).
    pub grid_opacity: f32,
}

pub struct SplatBackbuffer {
    texture: wgpu::Texture,
    texture_id: TextureId,
    renderer: Arc<EguiRwLock<Renderer>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    widget_3d: Widget3D,
}

impl SplatBackbuffer {
    pub fn new(
        renderer: Arc<EguiRwLock<Renderer>>,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        // Start with a dummy texture
        let texture = create_texture(glam::uvec2(64, 64), &device);
        let id = renderer.write().register_native_texture(
            &device,
            &texture.create_view(&TextureViewDescriptor::default()),
            wgpu::FilterMode::Linear,
        );
        let widget_3d = Widget3D::new(device.clone(), queue.clone());

        Self {
            texture,
            texture_id: id,
            renderer,
            device,
            queue,
            widget_3d,
        }
    }

    /// Submit a render request. Spawns an async task to do the rendering.
    pub fn submit(&self, req: RenderRequest) {
        let needs_resize =
            self.texture.width() != req.img_size.x || self.texture.height() != req.img_size.y;
        if needs_resize {
            // TODO: Restore this.
            // let client = WgpuRuntime::client(&req.);
            // client.memory_cleanup();

            self.texture = create_texture(req.img_size, &self.device);
            self.renderer.write().update_egui_texture_from_wgpu_texture(
                &self.device,
                &self.texture.create_view(&TextureViewDescriptor::default()),
                wgpu::FilterMode::Linear,
                self.texture_id,
            );
        }

        block_on(async move {
            let camera = req.camera.clone();
            let img_size = req.img_size;
            let background = req.background;
            let splat_scale = req.splat_scale;

            let splats = req.slot.clone_main().await;

            if let Some(splats) = splats {
                let (image, _) = render_splats(
                    splats.clone(),
                    &camera,
                    img_size,
                    background,
                    splat_scale,
                    TextureMode::Packed,
                )
                .await;

                copy_to_texture(
                    image,
                    &self.texture,
                    self.texture_id,
                    &self.renderer,
                    &self.device,
                    &self.queue,
                );

                if req.grid_opacity > 0.0 {
                    self.widget_3d.render_to_texture(
                        &req.camera,
                        req.model_transform,
                        req.img_size,
                        &texture,
                        req.grid_opacity,
                    );
                }
            }
            req.ctx.request_repaint();
        });
    }

    pub fn id(&self) -> Option<TextureId> {
        self.texture_id
    }
}

fn create_texture(size: glam::UVec2, device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Splat backbuffer"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    })
}

fn copy_to_texture(
    img: Tensor<MainBackend, 3>,
    texture: &wgpu::Texture,
    texture_id: TextureId,
    renderer: &Arc<EguiRwLock<Renderer>>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let [h, w, c] = img.shape().dims();
    assert!(c == 1, "texture should be u8 packed RGBA");
    let size = glam::uvec2(w as u32, h as u32);

    let needs_resize = texture.width() != size.x || texture.height() != size.y;
    if needs_resize {
        let client = WgpuRuntime::client(&img.device());
        client.memory_cleanup();
        texture = create_texture(size, device);
        renderer.write().update_egui_texture_from_wgpu_texture(
            &device,
            &texture.create_view(&TextureViewDescriptor::default()),
            wgpu::FilterMode::Linear,
            texture_id,
        );
    }

    let [height, width, c] = img.dims();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("splat backbuffer encoder"),
    });

    let padded_shape = vec![height, width.div_ceil(64) * 64, c];

    let img_prim = img.into_primitive().tensor();
    let fusion_client = img_prim.client.clone();
    let img = fusion_client.resolve_tensor_float::<MainBackendBase>(img_prim);
    let img: Tensor<MainBackendBase, 3> = Tensor::from_primitive(TensorPrimitive::Float(img));

    // Pad if needed (WebGPU requires bytes_per_row divisible by 256)
    let img = if width % 64 != 0 {
        let padded: Tensor<MainBackendBase, 3> = Tensor::zeros(&padded_shape, &img.device());
        padded.slice_assign([0..height, 0..width], img)
    } else {
        img
    };

    let img = img.into_primitive().tensor();
    let client = &img.client;
    let img_res_handle = client.get_resource(img.handle.clone().binding());
    client.flush();

    let bytes_per_row = Some(4 * padded_shape[1] as u32);

    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &img_res_handle.resource().buffer,
            layout: TexelCopyBufferLayout {
                offset: img_res_handle.resource().offset,
                bytes_per_row,
                rows_per_image: None,
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
    );

    queue.submit([encoder.finish()]);
}
