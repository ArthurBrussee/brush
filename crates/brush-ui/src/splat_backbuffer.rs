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

/// Request sent to the async render worker.
#[derive(Clone)]
pub struct RenderRequest {
    pub slot: Slot<Splats<MainBackend>>,
    pub frame: usize,

    pub camera: Camera,
    pub background: Vec3,
    pub splat_scale: Option<f32>,

    pub img_size: UVec2,
    pub ctx: egui::Context,
}

pub struct SplatBackbuffer {
    req_send: mpsc::UnboundedSender<RenderRequest>,
    img_rec: mpsc::Receiver<Tensor<MainBackend, 3>>,
    last_image: Option<Tensor<MainBackend, 3>>,
}

impl SplatBackbuffer {
    pub fn new() -> Self {
        // Create channel for render requests
        let (req_send, req_rec) = mpsc::unbounded_channel();
        let (img_send, img_rec) = mpsc::channel(1);

        tokio::task::spawn(render_worker(req_rec, img_send));
        Self {
            req_send,
            img_rec,
            last_image: None,
        }
    }

    /// Submit a render request. The async worker will process it.
    pub fn submit(&mut self, req: RenderRequest) {
        // Send request to worker (ignore send errors if channel closed)
        let _ = self.req_send.send(req);
    }

    pub fn paint(
        &mut self, // Not used atm, but, in the future the widget might have some state.
        rect: Rect,
        ui: &egui::Ui,
    ) {
        while let Ok(img) = self.img_rec.try_recv() {
            self.last_image = Some(img);
        }

        if let Some(image) = &self.last_image {
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    SplatBackbufferPainter {
                        rect,
                        last_img: image.clone(),
                    },
                ));
        }
    }
}

struct SplatBackbufferPainter {
    rect: Rect,
    last_img: Tensor<MainBackend, 3>,
}

impl CallbackTrait for SplatBackbufferPainter {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        _render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let last_img = self.last_img.clone().into_primitive().tensor();

        let client = last_img.client.clone();
        let prim_tensor = client.resolve_tensor_int::<MainBackendBase>(last_img);
        let prim_client = prim_tensor.client;
        let img_res_handle = prim_client.get_resource(prim_tensor.handle.binding());
        let buffer = img_res_handle.resource();
        // TODO: Draw our tensor to the screen here from straight wgpu!
        // The shader can read from the wgpu binding.
        // Nearest neighbour sampling is fine, we _usually_ have things rendered to the correct nr. of pixels,
        // only when we're midway a resize do we not, so not much point in filtering.
        // NB: The format of this tensor is NOT a float, but PACKED RGBA8, secretely.
    }
}

/// Async render worker that processes render requests.
async fn render_worker(
    mut receiver: mpsc::UnboundedReceiver<RenderRequest>,
    img_sender: mpsc::Sender<Tensor<MainBackend, 3>>,
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

        // Don't care about errors if channel is closed.
        let _ = img_sender.send(image).await;

        // TODO: Store the latest img tensor. Only continue
        // once it has been removed from the channel.

        // Trigger egui repaint
        request.ctx.request_repaint();
    }
}
