use std::sync::Arc;

use burn::{
    backend::{
        wgpu::{JitBackend, WgpuRuntime},
        Wgpu,
    },
    tensor::{Tensor, TensorPrimitive},
};
use burn_fusion::client::FusionClient;
use eframe::egui_wgpu::Renderer;
use egui::epaint::mutex::RwLock as EguiRwLock;
use egui::TextureId;
use wgpu::{CommandEncoderDescriptor, ImageDataLayout, TextureViewDescriptor};

type InnerWgpu = JitBackend<WgpuRuntime, f32, i32, u32>;

struct TextureState {
    texture: wgpu::Texture,
    id: TextureId,
}

pub struct BurnTexture {
    state: Option<TextureState>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
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
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    })
}

impl BurnTexture {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            state: None,
            device,
            queue,
        }
    }

    pub fn update_texture(
        &mut self,
        img: Tensor<Wgpu, 3>,
        renderer: &EguiRwLock<Renderer>,
    ) -> TextureId {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("viewer encoder"),
            });

        let [h, w, _] = img.shape().dims();
        let size = glam::uvec2(w as u32, h as u32);

        let dirty = if let Some(s) = self.state.as_ref() {
            s.texture.width() != size.x || s.texture.height() != size.y
        } else {
            true
        };

        if dirty {
            let texture = create_texture(glam::uvec2(w as u32, h as u32), &self.device);

            if let Some(s) = self.state.as_mut() {
                s.texture = texture;

                renderer.write().update_egui_texture_from_wgpu_texture(
                    &self.device,
                    &s.texture.create_view(&TextureViewDescriptor::default()),
                    wgpu::FilterMode::Linear,
                    s.id,
                );
            } else {
                let id = renderer.write().register_native_texture(
                    &self.device,
                    &texture.create_view(&TextureViewDescriptor::default()),
                    wgpu::FilterMode::Linear,
                );

                self.state = Some(TextureState { texture, id });
            }
        }

        let Some(s) = self.state.as_ref() else {
            unreachable!("Somehow failed to initialize")
        };

        let img_prim = img.into_primitive().tensor();
        let fusion_client = img_prim.client.clone();
        let img = fusion_client.resolve_tensor_float::<InnerWgpu>(img_prim);
        let texture: &wgpu::Texture = &s.texture;
        let [height, width, c] = img.shape.dims();

        let padded_shape = vec![height, width.div_ceil(64) * 64, c];

        // Create padded tensor if needed. The bytes_per_row needs to be divisible
        // by 256 in WebGPU, so 4 bytes per pixel means width needs to be disible by 64.
        let img = if width % 64 != 0 {
            let padded: Tensor<InnerWgpu, 3> = Tensor::zeros(&padded_shape, &img.device);
            let img = Tensor::from_primitive(TensorPrimitive::Float(img));
            let padded = padded.slice_assign([0..height, 0..width], img);
            padded.into_primitive().tensor()
        } else {
            img
        };

        // Get a hold of the Burn resource.
        let client = &img.client;
        let img_res_handle = client.get_resource(img.handle.clone().binding());

        // Now flush commands to make sure the resource is fully ready.
        client.flush();

        // Put compute passes in encoder before copying the buffer.
        let bytes_per_row = Some(4 * padded_shape[1] as u32);

        // Now copy the buffer to the texture.
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: img_res_handle.resource().buffer.as_ref(),
                layout: ImageDataLayout {
                    offset: img_res_handle.resource().offset(),
                    bytes_per_row,
                    rows_per_image: None,
                },
            },
            wgpu::ImageCopyTexture {
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

        self.queue.submit([encoder.finish()]);

        s.id
    }

    pub fn id(&self) -> Option<TextureId> {
        self.state.as_ref().map(|s| s.id)
    }
}
