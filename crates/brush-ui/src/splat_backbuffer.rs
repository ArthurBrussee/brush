use brush_process::slot::Slot;
use brush_render::{
    MainBackend, MainBackendBase, TextureMode, camera::Camera, gaussian_splats::Splats,
    render_splats,
};
use burn::tensor::{Tensor, TensorPrimitive};
use egui::TextureId;
use glam::Vec3;
use std::sync::Arc;
use tokio::sync::mpsc;
use wgpu::{CommandEncoderDescriptor, TexelCopyBufferLayout, TextureViewDescriptor};

use eframe::egui_wgpu::{self, wgpu};

/// Request sent to the async render worker.
#[derive(Clone)]
pub struct RenderRequest {
    pub slot: Slot<Splats<MainBackend>>,
    pub frame: usize,
    pub camera: Camera,
    pub img_size: glam::UVec2,
    pub background: Vec3,
    pub splat_scale: Option<f32>,
    pub ctx: egui::Context,
}

pub struct SplatRenderResources {
    pub texture: wgpu::Texture,
    pub size: glam::UVec2,
}

pub struct SplatBackbuffer {
    /// Channel to send requests to the async worker.
    request_sender: mpsc::UnboundedSender<RenderRequest>,
    /// Egui texture ID for displaying results.
    texture_id: TextureId,
    /// Device reference for texture operations.
    device: wgpu::Device,
    /// Renderer reference for texture updates.
    renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
}

impl SplatBackbuffer {
    pub fn new(
        renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
        device: wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        // Create initial texture
        let initial_size = glam::uvec2(64, 64);
        let texture = create_texture(initial_size, &device);
        let texture_id = renderer.write().register_native_texture(
            &device,
            &texture.create_view(&TextureViewDescriptor::default()),
            wgpu::FilterMode::Linear,
        );

        // Insert resources into callback_resources
        renderer
            .write()
            .callback_resources
            .insert(SplatRenderResources {
                texture,
                size: initial_size,
            });

        // Create channel for render requests
        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn the async render worker
        let worker_device = device.clone();
        let worker_queue = queue.clone();
        let worker_renderer = renderer.clone();
        tokio::task::spawn(render_worker(
            rx,
            worker_device,
            worker_queue,
            worker_renderer,
        ));

        Self {
            request_sender: tx,
            texture_id,
            device,
            renderer,
        }
    }

    /// Submit a render request. The async worker will process it.
    pub fn submit(&mut self, req: RenderRequest) {
        if req.img_size.x <= 8 || req.img_size.y <= 8 {
            return;
        }

        // Handle resize synchronously since it requires renderer lock
        let needs_resize = {
            let renderer = self.renderer.read();
            let resources: Option<&SplatRenderResources> = renderer.callback_resources.get();
            resources.is_none_or(|r| r.size != req.img_size)
        };

        if needs_resize {
            let new_texture = create_texture(req.img_size, &self.device);
            self.renderer.write().update_egui_texture_from_wgpu_texture(
                &self.device,
                &new_texture.create_view(&TextureViewDescriptor::default()),
                wgpu::FilterMode::Linear,
                self.texture_id,
            );

            // Update the texture in callback_resources
            let mut renderer = self.renderer.write();
            if let Some(resources) = renderer
                .callback_resources
                .get_mut::<SplatRenderResources>()
            {
                resources.texture = new_texture;
                resources.size = req.img_size;
            }
        }
        // Send request to worker (ignore send errors if channel closed)
        let _ = self.request_sender.send(req);
    }

    /// Get the texture ID for displaying the rendered result.
    pub fn id(&self) -> TextureId {
        self.texture_id
    }
}

/// Async render worker that processes render requests.
async fn render_worker(
    mut receiver: mpsc::UnboundedReceiver<RenderRequest>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
) {
    loop {
        // Wait for at least one request
        let Some(mut request) = receiver.recv().await else {
            break; // Channel closed
        };

        // Coalesce: drain channel, keep only the last request
        while let Ok(newer) = receiver.try_recv() {
            request = newer;
        }

        // Skip tiny sizes
        if request.img_size.x <= 8 || request.img_size.y <= 8 {
            continue;
        }

        // Clone splats (async)
        let Some(splats) = request.slot.clone_main().await else {
            continue;
        };

        // Render (async)
        let (image, _) = render_splats(
            splats,
            &request.camera,
            request.img_size,
            request.background,
            request.splat_scale,
            TextureMode::Packed,
        )
        .await;

        // Get the texture from callback_resources and copy to it
        {
            let renderer_guard = renderer.read();
            if let Some(resources) = renderer_guard
                .callback_resources
                .get::<SplatRenderResources>()
            {
                // Only copy if size matches (resize handled in submit)
                if resources.size == request.img_size {
                    copy_to_texture(image, &resources.texture, &device, &queue);
                }
            }
        }

        // Trigger egui repaint
        request.ctx.request_repaint();
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
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
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
            texture,
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
