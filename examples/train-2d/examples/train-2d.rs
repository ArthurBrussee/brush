#![recursion_limit = "256"]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use brush_dataset::scene::{SceneBatch, sample_to_tensor_data};
use brush_process::slot::Slot;
use brush_render::{
    AlphaMode, MainBackend,
    bounding_box::BoundingBox,
    camera::{Camera, focal_to_fov, fov_to_focal},
    gaussian_splats::{SplatRenderMode, Splats},
};
use brush_train::{
    RandomSplatsConfig, config::TrainConfig, create_random_splats, splats_into_autodiff,
    train::SplatTrainer,
};
use brush_ui::splat_backbuffer::{RenderRequest, SplatBackbuffer};
use burn::{backend::wgpu::WgpuDevice, module::AutodiffModule, prelude::Backend};
use egui::{ImageSource, TextureHandle, TextureOptions, load::SizedTexture};
use glam::{Quat, Vec2, Vec3};
use image::DynamicImage;
use rand::SeedableRng;
use tokio::sync::mpsc::{Receiver, Sender};

struct TrainStep {
    iter: u32,
    num_splats: u32,
}

fn spawn_train_loop(
    image: DynamicImage,
    cam: Camera,
    config: TrainConfig,
    device: WgpuDevice,
    ctx: egui::Context,
    sender: Sender<TrainStep>,
    slot: Slot<Splats<MainBackend>>,
) {
    // Spawn a task that iterates over the training stream.
    tokio::spawn(async move {
        let seed = 42;

        <MainBackend as Backend>::seed(&device, seed);
        let mut rng = rand::rngs::StdRng::from_seed([seed as u8; 32]);

        let init_bounds = BoundingBox::from_min_max(-Vec3::ONE * 5.0, Vec3::ONE * 5.0);

        let mut splats = create_random_splats(
            &RandomSplatsConfig::new().with_init_count(512),
            init_bounds,
            &mut rng,
            SplatRenderMode::Default,
            &device,
        );

        let mut trainer = SplatTrainer::new(
            &config,
            &device,
            BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
        );

        // One batch of training data, it's the same every step so can just construct it once.
        let batch = SceneBatch {
            img_tensor: sample_to_tensor_data(image),
            alpha_mode: AlphaMode::Transparent,
            camera: cam,
        };

        let mut iter = 0;

        loop {
            let (new_splats, _) = trainer.step(batch.clone(), splats).await;
            let (new_splats, _) = trainer.refine(iter, new_splats.valid()).await;
            let num_splats = new_splats.num_splats();

            // Update the slot with latest splats
            slot.set(new_splats.clone()).await;

            splats = splats_into_autodiff(new_splats);
            iter += 1;
            ctx.request_repaint();

            if sender.send(TrainStep { iter, num_splats }).await.is_err() {
                break;
            }
        }
    });
}

struct App {
    image: image::DynamicImage,
    camera: Camera,
    tex_handle: TextureHandle,
    backbuffer: SplatBackbuffer,
    slot: Slot<Splats<MainBackend>>,
    receiver: Receiver<TrainStep>,
    last_step: Option<TrainStep>,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc
            .wgpu_render_state
            .as_ref()
            .expect("No wgpu renderer enabled in egui");
        let device = brush_process::burn_init_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        let image = image::open("./crab.jpg").expect("Failed to open image");

        let fov_x = 0.5 * std::f64::consts::PI;
        let fov_y = focal_to_fov(fov_to_focal(fov_x, image.width()), image.height());

        let center_uv = Vec2::ONE * 0.5;

        let camera = Camera::new(
            glam::vec3(0.0, 0.0, -5.0),
            Quat::IDENTITY,
            fov_x,
            fov_y,
            center_uv,
        );

        let (sender, receiver) = tokio::sync::mpsc::channel(32);

        let color_img = egui::ColorImage::from_rgb(
            [image.width() as usize, image.height() as usize],
            &image.to_rgb8().into_vec(),
        );
        let handle =
            cc.egui_ctx
                .load_texture("nearest_view_tex", color_img, TextureOptions::default());

        let slot = Slot::default();

        let config = TrainConfig::default();
        spawn_train_loop(
            image.clone(),
            camera.clone(),
            config,
            device,
            cc.egui_ctx.clone(),
            sender,
            slot.clone(),
        );

        let renderer = cc
            .wgpu_render_state
            .as_ref()
            .expect("No wgpu renderer enabled in egui")
            .renderer
            .clone();

        Self {
            image,
            camera,
            tex_handle: handle,
            backbuffer: SplatBackbuffer::new(renderer, state.device.clone(), state.queue.clone()),
            slot,
            receiver,
            last_step: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(step) = self.receiver.try_recv() {
            self.last_step = Some(step);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(step) = self.last_step.as_ref() else {
                ui.label("Waiting for first training step...");
                return;
            };

            // Submit a render request
            self.backbuffer.submit(RenderRequest {
                slot: self.slot.clone(),
                frame: 0,
                camera: self.camera.clone(),
                img_size: glam::uvec2(self.image.width(), self.image.height()),
                background: Vec3::ZERO,
                splat_scale: None,
            });

            let size = egui::vec2(self.image.width() as f32, self.image.height() as f32);

            ui.horizontal(|ui| {
                if let Some(texture_id) = self.backbuffer.id() {
                    ui.image(ImageSource::Texture(SizedTexture::new(texture_id, size)));
                } else {
                    ui.label("Rendering...");
                }
                ui.image(ImageSource::Texture(SizedTexture::new(
                    self.tex_handle.id(),
                    size,
                )));
            });

            ui.label(format!("Splats: {}", step.num_splats));
            ui.label(format!("Step: {}", step.iter));
        });
    }
}

#[tokio::main]
async fn main() {
    let native_options = eframe::NativeOptions {
        // Build app display.
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::Vec2::new(1100.0, 500.0))
            .with_active(true),
        wgpu_options: brush_ui::create_egui_options(),
        ..Default::default()
    };

    eframe::run_native(
        "Brush",
        native_options,
        Box::new(move |cc| Ok(Box::new(App::new(cc)))),
    )
    .expect("Failed to run egui app");
}
