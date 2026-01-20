//! Async splat rendering backbuffer.
//!
//! Background task renders splats directly to a wgpu texture.
//! wgpu handles GPU synchronization - no locks needed on results.

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
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_with_wasm::alias::task;
use wgpu::{CommandEncoderDescriptor, TexelCopyBufferLayout, TextureViewDescriptor};

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

/// Shared texture state that the render loop writes to directly.
struct TextureState {
    texture: Option<wgpu::Texture>,
    texture_id: Option<TextureId>,
}

/// Async splat rendering backbuffer.
///
/// Background task renders splats and writes directly to a wgpu texture.
/// No synchronization needed on results - wgpu handles that.
pub struct SplatBackbuffer {
    texture_state: Arc<Mutex<TextureState>>,
    request_tx: mpsc::UnboundedSender<RenderRequest>,
}

impl SplatBackbuffer {
    pub fn new(
        renderer: Arc<EguiRwLock<Renderer>>,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let texture_state = Arc::new(Mutex::new(TextureState {
            texture: None,
            texture_id: None,
        }));
        let (request_tx, request_rx) = mpsc::unbounded_channel();

        // Spawn the background render loop
        task::spawn(render_loop(
            Arc::clone(&texture_state),
            renderer,
            device,
            queue,
            request_rx,
        ));

        Self {
            texture_state,
            request_tx,
        }
    }

    /// Submit a new render request.
    pub fn submit(&self, request: RenderRequest) {
        let _ = self.request_tx.send(request);
    }

    /// Get the texture ID for display.
    pub fn id(&self) -> Option<TextureId> {
        self.texture_state.lock().unwrap().texture_id
    }

    /// Get the underlying texture for additional rendering.
    pub fn texture(&self) -> Option<wgpu::Texture> {
        self.texture_state.lock().unwrap().texture.clone()
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
    texture_state: &Arc<Mutex<TextureState>>,
    renderer: &Arc<EguiRwLock<Renderer>>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let [h, w, c] = img.shape().dims();
    assert!(c == 1, "texture should be u8 packed RGBA");
    let size = glam::uvec2(w as u32, h as u32);

    // Check if we need to resize/create texture
    let needs_resize = {
        let state = texture_state.lock().unwrap();
        state
            .texture
            .as_ref()
            .is_none_or(|t| t.width() != size.x || t.height() != size.y)
    };

    if needs_resize {
        // Cleanup memory when resizing
        let client = WgpuRuntime::client(&img.device());
        client.memory_cleanup();

        let texture = create_texture(size, device);

        let mut state = texture_state.lock().unwrap();
        if let Some(id) = state.texture_id {
            // Update existing registration
            renderer.write().update_egui_texture_from_wgpu_texture(
                device,
                &texture.create_view(&TextureViewDescriptor::default()),
                wgpu::FilterMode::Linear,
                id,
            );
        } else {
            // New registration
            let id = renderer.write().register_native_texture(
                device,
                &texture.create_view(&TextureViewDescriptor::default()),
                wgpu::FilterMode::Linear,
            );
            state.texture_id = Some(id);
        }

        state.texture = Some(texture);
    }

    let texture = texture_state.lock().unwrap().texture.clone().unwrap();
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

async fn render_loop(
    texture_state: Arc<Mutex<TextureState>>,
    renderer: Arc<EguiRwLock<Renderer>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    mut request_rx: mpsc::UnboundedReceiver<RenderRequest>,
) {
    // Create Widget3D for rendering the grid overlay
    let widget_3d = Widget3D::new(device.clone(), queue.clone());

    while let Some(mut req) = request_rx.recv().await {
        // Drain channel to get the latest request
        while let Ok(newer) = request_rx.try_recv() {
            req = newer;
        }

        let camera = req.camera.clone();
        let img_size = req.img_size;
        let background = req.background;
        let splat_scale = req.splat_scale;

        let render_result = req
            .slot
            .act(req.frame, async move |splats| {
                let (image, _) = render_splats(
                    splats.clone(),
                    &camera,
                    img_size,
                    background,
                    splat_scale,
                    TextureMode::Packed,
                )
                .await;
                (splats, image)
            })
            .await;

        if let Some(image) = render_result {
            copy_to_texture(image, &texture_state, &renderer, &device, &queue);

            // Render 3D overlay (grid, axes) on top of the splats
            if req.grid_opacity > 0.0
                && let Some(texture) = texture_state.lock().unwrap().texture.clone()
            {
                widget_3d.render_to_texture(
                    &req.camera,
                    req.model_transform,
                    req.img_size,
                    &texture,
                    req.grid_opacity,
                );
            }
        }

        req.ctx.request_repaint();
    }
}
