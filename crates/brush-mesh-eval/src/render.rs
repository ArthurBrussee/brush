//! Tiny wgpu mesh renderer: vertex-colored triangles, no lighting, no
//! tone-mapping — the rendered pixel is exactly the barycentric
//! interpolation of the three vertex colours, so PSNR vs the GT image
//! reflects mesh fidelity rather than any rendering bias.
//!
//! Camera convention matches brush: view-space `+X right, +Y down, +Z
//! forward`. A projection matrix is built that maps view-space directly
//! to wgpu clip space; no axis flips are needed.

use std::borrow::Cow;

use anyhow::Result;
use brush_mesh::Mesh;
use brush_render::camera::Camera;
use glam::{Mat4, UVec2};
use image::RgbImage;
use wgpu::util::DeviceExt;

// `Rgba8Unorm` (no auto sRGB encoding on write). Vertex colours are
// already sRGB bytes; the shader passes them through as-is, and we want
// the output PNG bytes to match GT sRGB directly. Using `Rgba8UnormSrgb`
// here would treat the shader output as linear and re-encode to sRGB,
// double-brightening the result.
const COLOR_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const NEAR_PLANE: f32 = 0.01;
const FAR_PLANE: f32 = 1.0e4;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}

pub struct MeshRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
}

impl MeshRenderer {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: true,
        }))?;
        let limits = adapter.limits();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("mesh-render"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                // SAFETY: we don't load arbitrary passthrough shaders; only
                // the local WGSL embedded in this file. The flag exists in
                // brush's wgpu fork; defaulting to enabled matches the
                // rest of the brush wgpu setup.
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            }))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh-render-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("mesh_render.wgsl"))),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh-render-bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-render-pl"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh-render-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FMT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Mesh extraction doesn't guarantee a consistent winding
                // (and marching-tets' triangle table flips with sign
                // pattern). Render both sides.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FMT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_layout,
        })
    }

    pub fn render_with_depth(
        &self,
        mesh: &Mesh,
        camera: &Camera,
        img_size: UVec2,
    ) -> (RgbImage, Vec<f32>) {
        let (color, depth) = self.render_inner(mesh, camera, img_size, true);
        (color, depth.expect("requested depth"))
    }

    fn render_inner(
        &self,
        mesh: &Mesh,
        camera: &Camera,
        img_size: UVec2,
        read_depth: bool,
    ) -> (RgbImage, Option<Vec<f32>>) {
        let (verts, indices) = build_verts(mesh);
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh-vb"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh-ib"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mvp = build_mvp(camera, img_size);
        let mvp_arr: [f32; 16] = mvp.to_cols_array();
        let ub = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh-ub"),
                contents: bytemuck::cast_slice(&mvp_arr),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh-bg"),
            layout: &self.bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ub.as_entire_binding(),
            }],
        });

        let w = img_size.x;
        let h = img_size.y;
        let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh-color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FMT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_usage = if read_depth {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh-depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FMT,
            usage: depth_usage,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Color readback buffer (256-byte row alignment).
        let row_align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let color_padded_row = (w * 4).div_ceil(row_align) * row_align;
        let color_read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-read-color"),
            size: (color_padded_row as u64) * (h as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth_padded_row = (w * 4).div_ceil(row_align) * row_align;
        let depth_read = if read_depth {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh-read-depth"),
                size: (depth_padded_row as u64) * (h as u64),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mesh-encoder"),
            });
        {
            let depth_store = if read_depth {
                wgpu::StoreOp::Store
            } else {
                wgpu::StoreOp::Discard
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: depth_store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..(indices.len() as u32), 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &color_read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(color_padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        if let Some(buf) = &depth_read {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &depth_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(depth_padded_row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.queue.submit([encoder.finish()]);

        let rgb = map_and_unpack_rgb(&self.device, &color_read, w, h, color_padded_row);
        let depth_lin = depth_read.map(|buf| {
            let raw = map_and_unpack_f32(&self.device, &buf, w, h, depth_padded_row);
            let z_scale = FAR_PLANE / (FAR_PLANE - NEAR_PLANE);
            let z_bias = -NEAR_PLANE * FAR_PLANE / (FAR_PLANE - NEAR_PLANE);
            raw.into_iter()
                .map(|d| {
                    if d >= 1.0 - 1e-6 {
                        f32::NAN
                    } else {
                        z_bias / (d - z_scale)
                    }
                })
                .collect::<Vec<f32>>()
        });

        let img = RgbImage::from_raw(w, h, rgb).expect("rgb buf size");
        (img, depth_lin)
    }
}

fn build_verts(mesh: &Mesh) -> (Vec<Vertex>, Vec<u32>) {
    let n_verts = mesh.vertices.len();
    let has_color = mesh.vertex_colors.len() == n_verts;
    let mut verts: Vec<Vertex> = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let p = mesh.vertices[i];
        let c = if has_color {
            let c = mesh.vertex_colors[i];
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        } else {
            [0.5, 0.5, 0.5]
        };
        verts.push(Vertex {
            pos: [p.x, p.y, p.z],
            color: c,
        });
    }
    let mut indices: Vec<u32> = Vec::with_capacity(mesh.faces.len() * 3);
    for f in &mesh.faces {
        indices.extend_from_slice(f);
    }
    (verts, indices)
}

fn map_and_unpack_rgb(
    device: &wgpu::Device,
    buf: &wgpu::Buffer,
    w: u32,
    h: u32,
    padded_row: u32,
) -> Vec<u8> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("buffer map");
    rx.recv().expect("map channel").expect("map async result");
    let raw = slice.get_mapped_range().expect("get_mapped_range");
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for row in 0..h {
        let row_start = (row as usize) * (padded_row as usize);
        for col in 0..w {
            let base = row_start + (col as usize) * 4;
            rgb.push(raw[base]);
            rgb.push(raw[base + 1]);
            rgb.push(raw[base + 2]);
        }
    }
    drop(raw);
    buf.unmap();
    rgb
}

fn map_and_unpack_f32(
    device: &wgpu::Device,
    buf: &wgpu::Buffer,
    w: u32,
    h: u32,
    padded_row: u32,
) -> Vec<f32> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("buffer map");
    rx.recv().expect("map channel").expect("map async result");
    let raw = slice.get_mapped_range().expect("get_mapped_range");
    let mut out = Vec::with_capacity((w * h) as usize);
    for row in 0..h {
        let row_start = (row as usize) * (padded_row as usize);
        for col in 0..w {
            let base = row_start + (col as usize) * 4;
            let v = f32::from_le_bytes([raw[base], raw[base + 1], raw[base + 2], raw[base + 3]]);
            out.push(v);
        }
    }
    drop(raw);
    buf.unmap();
    out
}

/// Build the MVP matrix mapping world-space vertex positions to wgpu
/// clip space. Brush's view frame is `+X right, +Y down, +Z forward`,
/// which already matches wgpu's NDC convention (`+Y down`, `+Z forward`,
/// `Z ∈ [0, 1]`), so the projection is a straight build with no axis
/// flips.
fn build_mvp(camera: &Camera, img_size: UVec2) -> Mat4 {
    let pinhole = camera.build_pinhole_params(img_size);
    let w = img_size.x as f32;
    let h = img_size.y as f32;
    let near = NEAR_PLANE;
    let far = FAR_PLANE;

    // Pinhole-projection matrix tailored to brush's intrinsics. Pixel
    // (px, py) → NDC ((px - W/2)/(W/2), (py - H/2)/(H/2)), so we
    // factor that into the projection:
    //   clip.x = (2 * fx / W) * x_view + (1 - 2 * cx / W) * z_view
    //   clip.y = (2 * fy / H) * y_view + (1 - 2 * cy / H) * z_view
    //   clip.z = (far / (far - near)) * z_view  − (near * far / (far - near))
    //   clip.w = z_view
    let fx = pinhole.fx;
    let fy = pinhole.fy;
    let cx = pinhole.cx;
    let cy = pinhole.cy;
    let z_scale = far / (far - near);
    let z_bias = -near * far / (far - near);
    // wgpu NDC has +X right, +Y *up*, +Z forward; brush view space has
    // +X right, +Y *down*, +Z forward. Negate the Y row of the
    // projection to flip the screen vertically, so a pixel at the
    // bottom of brush's image lands at the bottom of the NDC frame.
    // glam Mat4 is column-major.
    let proj = Mat4::from_cols_array(&[
        // col 0
        2.0 * fx / w,
        0.0,
        0.0,
        0.0,
        // col 1
        0.0,
        -(2.0 * fy / h),
        0.0,
        0.0,
        // col 2
        1.0 - 2.0 * cx / w,
        -(1.0 - 2.0 * cy / h),
        z_scale,
        1.0,
        // col 3
        0.0,
        0.0,
        z_bias,
        0.0,
    ]);

    let view = Mat4::from(camera.world_to_local());
    proj * view
}

/// Map a linear-z buffer (`NaN` = no mesh) to a Turbo-colormapped image
/// (near = blue, far = red). Foreground depth is normalized to its
/// 5th/95th percentile so a single outlier vertex doesn't crush dynamic
/// range; background pixels are black.
pub fn depth_to_color(depth: &[f32], w: u32, h: u32) -> RgbImage {
    let mut vals: Vec<f32> = depth.iter().copied().filter(|v| v.is_finite()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).expect("finite-only sort"));
    let (lo, hi) = if vals.is_empty() {
        (0.0, 1.0)
    } else {
        let p_lo = vals[vals.len() * 5 / 100];
        let p_hi = vals[(vals.len() * 95 / 100).min(vals.len() - 1)];
        (p_lo, if p_hi > p_lo { p_hi } else { p_lo + 1e-6 })
    };
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for v in depth {
        if v.is_finite() {
            let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            out.extend_from_slice(&turbo(t));
        } else {
            out.extend_from_slice(&[0, 0, 0]);
        }
    }
    RgbImage::from_raw(w, h, out).expect("color buf size")
}

/// Google Turbo colormap, degree-5 polynomial approximation
/// (Mikhailov). `x` in `[0, 1]` → RGB, near=blue through green/yellow to
/// far=red.
fn turbo(x: f32) -> [u8; 3] {
    let x = x.clamp(0.0, 1.0);
    let (x2, x3, x4, x5) = (x * x, x * x * x, x * x * x * x, x * x * x * x * x);
    let r = 0.13572138 + 4.615_392_7 * x - 42.660_324 * x2 + 132.131_09 * x3 - 152.942_4 * x4
        + 59.286_38 * x5;
    let g = 0.09140261 + 2.194_188_4 * x + 4.842_966_6 * x2 - 14.185_034 * x3
        + 4.277_298_5 * x4
        + 2.829_566 * x5;
    let b = 0.106_673_3 + 12.641_946 * x - 60.582_047 * x2 + 110.362_77 * x3 - 89.903_11 * x4
        + 27.348_25 * x5;
    [
        (r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (b.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}
