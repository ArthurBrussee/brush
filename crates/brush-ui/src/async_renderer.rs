use brush_process::slot::Slot;
use brush_render::{
    MainBackend, TextureMode, camera::Camera, gaussian_splats::Splats, render_splats,
};
use burn::tensor::Tensor;
use glam::Vec3;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::Notify;
use tokio_with_wasm::alias::task;

pub struct RenderRequest {
    pub slot: Slot<Splats<MainBackend>>,
    pub frame: usize,
    pub camera: Camera,
    pub img_size: glam::UVec2,
    pub background: Vec3,
    pub splat_scale: Option<f32>,
}

/// Result of a render operation.
pub struct RenderResult {
    pub image: Tensor<MainBackend, 3>,
    pub camera: Camera,
    pub img_size: glam::UVec2,
}

pub struct AsyncRenderer {
    /// Latest render request.
    request: Arc<Mutex<Option<RenderRequest>>>,
    request_notify: Arc<Notify>,

    result: Arc<Mutex<Option<RenderResult>>>,

    /// Flag to signal shutdown.
    shutdown: Arc<AtomicBool>,
}

impl AsyncRenderer {
    pub fn new() -> Self {
        let request = Arc::new(Mutex::new(None));
        let request_notify = Arc::new(Notify::new());
        let result = Arc::new(Mutex::new(None));
        let shutdown = Arc::new(AtomicBool::new(false));

        task::spawn(render_loop(
            Arc::clone(&request),
            Arc::clone(&request_notify),
            Arc::clone(&result),
            Arc::clone(&shutdown),
        ));

        Self {
            request,
            request_notify,
            result,
            shutdown,
        }
    }

    /// Submit a new render request. This will overwrite any pending request.
    pub fn submit(&self, new_request: RenderRequest) {
        {
            let mut req = self.request.lock().unwrap();
            *req = Some(new_request);
        }
        self.request_notify.notify_one();
    }

    /// Check if a new render result is available and return it.
    /// Returns `None` if no new result since last check.
    pub fn try_get_result(&self) -> Option<RenderResult> {
        self.result.lock().unwrap().take()
    }
}

impl Default for AsyncRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AsyncRenderer {
    fn drop(&mut self) {
        // Signal shutdown to the background task
        self.shutdown.store(true, Ordering::SeqCst);
        self.request_notify.notify_one();
    }
}

/// Background render loop that processes the latest render request.
async fn render_loop(
    request: Arc<Mutex<Option<RenderRequest>>>,
    request_notify: Arc<Notify>,
    result: Arc<Mutex<Option<RenderResult>>>,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        // Wait for a new request or shutdown
        request_notify.notified().await;

        // Check for shutdown
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        // Take the latest request (don't hold lock across await)
        let req = {
            let mut req_guard = request.lock().unwrap();
            req_guard.take()
        };

        if let Some(req) = req {
            // Use act to hold the slot lock across the render, so training will wait.
            // We clone the splats since render_splats takes ownership.
            let render_result = {
                let camera = req.camera.clone();
                let img_size = req.img_size;
                let background = req.background;
                let splat_scale = req.splat_scale;

                req.slot
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
                    .await
            };

            // Store the result with render params for widget_3d
            if let Some(image) = render_result {
                let mut res_guard = result.lock().unwrap();
                *res_guard = Some(RenderResult {
                    image,
                    camera: req.camera,
                    img_size: req.img_size,
                });
            }
        }
    }
}
