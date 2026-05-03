// Tiny WebGPU point-cloud renderer for live Brush splats.
//
// Reads the host-supplied storage buffers Brush exposes:
//   - transforms     [N, 10] f32: means(3) | rotation_xyzw(4) | log_scales(3)
//   - sh_coeffs      [N, n_coeffs, 3] f32 (only the L0 term is used for color)
//   - raw_opacities  [N] f32 (sigmoid -> alpha)
//
// Each splat is drawn as a soft circular sprite with the splat's color
// attenuated by its opacity. Transparent edges blend over the canvas.

const SHADER = /* wgsl */ `
struct Uniforms {
  view_proj: mat4x4<f32>,
  inv_screen: vec2<f32>,
  point_radius_px: f32,
  transforms_stride: u32,
  sh_stride: u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> transforms: array<f32>;
@group(0) @binding(2) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(3) var<storage, read> raw_opacities: array<f32>;

const SH_C0 = 0.28209479177387814;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) alpha: f32,
};

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
  // Mean (xyz) is the first 3 floats of each transforms row.
  let tbase = ii * u.transforms_stride;
  let pos_world = vec3<f32>(transforms[tbase], transforms[tbase + 1u], transforms[tbase + 2u]);

  // SH L0 (DC) term — first 3 floats of each splat's SH block.
  let sbase = ii * u.sh_stride;
  let sh0 = vec3<f32>(sh_coeffs[sbase], sh_coeffs[sbase + 1u], sh_coeffs[sbase + 2u]);
  let color = clamp(vec3<f32>(0.5) + SH_C0 * sh0, vec3<f32>(0.0), vec3<f32>(1.0));
  let alpha = sigmoid(raw_opacities[ii]);

  // Triangle-strip quad corners: (-1,-1) (1,-1) (-1,1) (1,1).
  let corner = vec2<f32>(
    f32((vi & 1u) * 2u) - 1.0,
    f32(((vi >> 1u) & 1u) * 2u) - 1.0,
  );

  let clip = u.view_proj * vec4<f32>(pos_world, 1.0);
  // Offset in clip space so each splat is point_radius_px on screen.
  // 2.0 * inv_screen.x maps 1px → clip-space units; multiply by clip.w
  // to keep the size constant in pixels regardless of perspective.
  let offset = corner * u.point_radius_px * 2.0 * u.inv_screen * clip.w;

  var out: VsOut;
  out.pos = vec4<f32>(clip.xy + offset, clip.z, clip.w);
  out.uv = corner;
  out.color = color;
  out.alpha = alpha;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  // Soft circular falloff: full alpha at center, 0 at the edge.
  let falloff = 1.0 - r2;
  let a = in.alpha * falloff;
  // Premultiplied alpha output (pairs with src-factor: 'one').
  return vec4<f32>(in.color * a, a);
}
`;

export interface Camera {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fovYRad: number;
  near: number;
  far: number;
}

interface Binding {
  transforms: GPUBuffer;
  shCoeffs: GPUBuffer;
  rawOpacities: GPUBuffer;
  count: number;
  shStride: number; // floats per splat in sh_coeffs (n_coeffs * 3)
}

const UNIFORM_SIZE = 96;

export class PointRenderer {
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private uniformBuffer: GPUBuffer;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup | null = null;
  private current: Binding | null = null;
  /** Point radius in pixels — feel free to expose this in the UI later. */
  pointRadiusPx = 4;

  constructor(
    private device: GPUDevice,
    private canvas: HTMLCanvasElement,
  ) {
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('canvas.getContext("webgpu") returned null');
    this.context = ctx;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });

    const module = device.createShaderModule({ code: SHADER });
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [
          {
            format: this.format,
            blend: {
              // Premultiplied alpha (we output color*alpha in the FS).
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-strip' },
    });

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Bind Brush's three live splat buffers. Re-creates the bind group only
   * when a buffer reference actually changes.
   */
  bindExternal(b: Binding): void {
    const same =
      this.current &&
      this.current.transforms === b.transforms &&
      this.current.shCoeffs === b.shCoeffs &&
      this.current.rawOpacities === b.rawOpacities;

    if (!same) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: b.transforms } },
          { binding: 2, resource: { buffer: b.shCoeffs } },
          { binding: 3, resource: { buffer: b.rawOpacities } },
        ],
      });
    }
    this.current = b;
  }

  render(camera: Camera): void {
    const w = Math.max(1, this.canvas.clientWidth | 0);
    const h = Math.max(1, this.canvas.clientHeight | 0);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    const view = lookAt(camera.position, camera.target, camera.up);
    const proj = perspective(camera.fovYRad, w / h, camera.near, camera.far);
    const uniforms = new ArrayBuffer(UNIFORM_SIZE);
    const f32 = new Float32Array(uniforms);
    const u32 = new Uint32Array(uniforms);
    f32.set(mul(proj, view), 0);
    f32[16] = 1 / w;
    f32[17] = 1 / h;
    f32[18] = this.pointRadiusPx;
    u32[19] = this.current ? 10 : 0; // transforms_stride
    u32[20] = this.current ? this.current.shStride : 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.06, g: 0.07, b: 0.09, a: 1.0 },
        },
      ],
    });

    if (this.current && this.bindGroup) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      // 4 vertices per quad (triangle-strip), one instance per splat.
      pass.draw(4, this.current.count, 0, 0);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

// -------------------------------------------------------------------------------------------
// Tiny matrix helpers — column-major mat4, output ready for WGSL.
// -------------------------------------------------------------------------------------------

type Vec3 = [number, number, number];

function sub3(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross3(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm3(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function lookAt(eye: Vec3, center: Vec3, up: Vec3): Float32Array {
  const f = norm3(sub3(center, eye));
  const s = norm3(cross3(f, up));
  const u = cross3(s, f);
  return new Float32Array([
    s[0], u[0], -f[0], 0,
    s[1], u[1], -f[1], 0,
    s[2], u[2], -f[2], 0,
    -dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1,
  ]);
}

function perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function mul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let v = 0;
      for (let k = 0; k < 4; k++) v += a[k * 4 + j] * b[i * 4 + k];
      out[i * 4 + j] = v;
    }
  }
  return out;
}
