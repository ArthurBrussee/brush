#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::sync::Arc;

use brush_render::{
    bounding_box::BoundingBox,
    camera::{focal_to_fov, fov_to_focal, Camera},
    gaussian_splats::{RandomSplatsConfig, Splats},
};
use brush_train::{
    image::image_to_tensor,
    scene::SceneView,
    train::{SceneBatch, SplatTrainer, TrainConfig},
};
use brush_ui::burn_texture::BurnTexture;
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    module::AutodiffModule,
};
use egui::{load::SizedTexture, ImageSource, TextureHandle, TextureOptions};
use glam::{Quat, Vec2, Vec3};
use rand::SeedableRng;
use tokio::sync::mpsc::{Receiver, Sender};

type Backend = Wgpu;

struct TrainStep {
    splats: Splats<Backend>,
    iter: u32,
}

fn spawn_train_loop(
    view: SceneView,
    config: TrainConfig,
    device: WgpuDevice,
    ctx: egui::Context,
    sender: Sender<TrainStep>,
) {
    // Spawn a task that iterates over the training stream.
    tokio::spawn(async move {
        let seed = 42;

        <Wgpu as burn::prelude::Backend>::seed(seed);
        let mut rng = rand::rngs::StdRng::from_seed([seed as u8; 32]);

        let init_bounds = BoundingBox::from_min_max(-Vec3::ONE * 5.0, Vec3::ONE * 5.0);

        let mut splats: Splats<Autodiff<Backend>> = Splats::from_random_config(
            &RandomSplatsConfig::new().with_init_count(32),
            init_bounds,
            &mut rng,
            &device,
        );

        let mut trainer = SplatTrainer::new(&splats, &config, &device);

        // One batch of training data, it's the same every step so can just cosntruct it once.
        let batch = SceneBatch {
            gt_images: image_to_tensor(&view.image, &device).unsqueeze(),
            gt_views: vec![view],
            scene_extent: 1.0,
        };

        let mut iter = 0;

        loop {
            let (new_splats, _) = trainer.step(iter, batch.clone(), splats).await;
            let (new_splats, _) = trainer
                .refine_if_needed(iter, new_splats, batch.scene_extent)
                .await;

            splats = new_splats;
            iter += 1;
            ctx.request_repaint();

            if sender
                .send(TrainStep {
                    splats: splats.valid(),
                    iter,
                })
                .await
                .is_err()
            {
                break;
            }
        }
    });
}

struct App {
    view: SceneView,
    tex_handle: TextureHandle,
    backbuffer: BurnTexture,
    receiver: Receiver<TrainStep>,
    last_step: Option<TrainStep>,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let state = cc
            .wgpu_render_state
            .as_ref()
            .expect("No wgpu renderer enabled in egui");
        let device = brush_ui::create_wgpu_device(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        let lr_max = 1.5e-4;
        let decay = 1.0;

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

        let view = SceneView {
            name: "crabby".to_owned(),
            camera,
            image: Arc::new(image),
        };
        let (sender, receiver) = tokio::sync::mpsc::channel(32);

        let color_img = egui::ColorImage::from_rgb(
            [view.image.width() as usize, view.image.height() as usize],
            &view.image.to_rgb8().into_vec(),
        );
        let handle =
            cc.egui_ctx
                .load_texture("nearest_view_tex", color_img, TextureOptions::default());

        let config = TrainConfig::new(ExponentialLrSchedulerConfig::new(lr_max, decay))
            .with_refine_start_iter(100) // Don't really need a warmup for simple 2D
            .with_refine_stop_iter(u32::MAX) // Just keep refining
            .with_reset_alpha_every_refine(u32::MAX); // Don't use alpha reset.

        spawn_train_loop(view.clone(), config, device, cc.egui_ctx.clone(), sender);

        Self {
            view,
            tex_handle: handle,
            backbuffer: BurnTexture::new(state.device.clone(), state.queue.clone()),
            receiver,
            last_step: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        while let Ok(step) = self.receiver.try_recv() {
            self.last_step = Some(step);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(msg) = self.last_step.as_ref() else {
                return;
            };

            let image = &self.view.image;

            let (img, _) = msg.splats.render(
                &self.view.camera,
                glam::uvec2(image.width(), image.height()),
                true,
            );

            let renderer = &frame
                .wgpu_render_state()
                .expect("No wgpu renderer enabled in egui")
                .renderer;
            let size = egui::vec2(image.width() as f32, image.height() as f32);

            ui.horizontal(|ui| {
                let texture_id = self.backbuffer.update_texture(img, renderer);
                ui.image(ImageSource::Texture(SizedTexture::new(texture_id, size)));
                ui.image(ImageSource::Texture(SizedTexture::new(
                    self.tex_handle.id(),
                    size,
                )));
            });

            ui.label(format!("Splats: {}", msg.splats.num_splats()));
            ui.label(format!("Step: {}", msg.iter));
        });
    }
}

#[tokio::main]
async fn main() {
    // NB: Load carrying icon. egui at head fails when no icon is included
    // as the built-in one is git-lfs which cargo doesn't clone properly.
    let icon = eframe::icon_data::from_png_bytes(
        &include_bytes!("../../brush-app/assets/icon-256.png")[..],
    )
    .expect("Failed to load icon");

    let native_options = eframe::NativeOptions {
        // Build app display.
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::Vec2::new(1100.0, 500.0))
            .with_active(true)
            .with_icon(std::sync::Arc::new(icon)),
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
