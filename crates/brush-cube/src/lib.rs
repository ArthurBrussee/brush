//! Shared brush primitives. Host-side tensor / launch helpers (re-exported
//! at the crate root) plus cube-side math types (`Vec3A`, `Quat`, `Mat3`,
//! `Sym2`), tile/pixel rect aggregates, and pure helpers (`sigmoid`,
//! `is_finite_*`, `calc_sigma`, `inverse_sym2`, `det2_strict`).
//!
//! Methods like `Vec3A::add` / `Quat::scale` are deliberately inherent
//! rather than `Add`/`Mul` impls — `#[cube]` traces method calls into
//! the IR, while operator overloading bypasses it.

#![allow(clippy::should_implement_trait)]

mod host;
pub mod test_helpers;
pub use host::*;

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

/// 3-component f32 vector, 16-byte aligned, padded to 4 lanes — same
/// shape as `glam::Vec3A`. Backed by `Vector<f32, Const<4>>` so the
/// emitted code uses native `vec4<f32>` ops; the 4th lane is pinned to
/// 0 so it never contributes to `dot` / `length` / etc.
///
/// `Vector<f32, Const<3>>` would be the more natural choice but
/// cubecl-cpp's Metal dialect emits `alignas(elem_size * lanes)`
/// literally — for 3 lanes of f32 that's `alignas(12)`, which is
/// invalid C++ (alignas requires a power of 2) and Metal rejects the
/// shader. 4-lane vectors get `alignas(16)` which is valid.
#[derive(CubeType, CubeTypeMut, Copy, Clone)]
pub struct Vec3A {
    inner: Vector<f32, Const<4>>,
}

#[cube]
impl Vec3A {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3A {
        let mut v = Vector::<f32, Const<4>>::empty();
        v[0] = x;
        v[1] = y;
        v[2] = z;
        // Padding lane — must stay 0 so `dot` and `length` see only
        // the three real components.
        v[3] = 0.0f32;
        Vec3A { inner: v }
    }

    pub fn splat(s: f32) -> Vec3A {
        Vec3A::new(s, s, s)
    }

    pub fn zero() -> Vec3A {
        Vec3A::splat(0.0f32)
    }

    pub fn x(self) -> f32 {
        self.inner[0]
    }
    pub fn y(self) -> f32 {
        self.inner[1]
    }
    pub fn z(self) -> f32 {
        self.inner[2]
    }

    pub fn add(self, other: Vec3A) -> Vec3A {
        Vec3A {
            inner: self.inner + other.inner,
        }
    }

    pub fn sub(self, other: Vec3A) -> Vec3A {
        Vec3A {
            inner: self.inner - other.inner,
        }
    }

    pub fn scale(self, s: f32) -> Vec3A {
        Vec3A {
            inner: self.inner * Vector::new(s),
        }
    }

    pub fn dot(self, other: Vec3A) -> f32 {
        let p = self.inner * other.inner;
        // Lane 3 is always 0 in both operands, so adding it is a no-op
        // we don't need to special-case.
        p[0] + p[1] + p[2] + p[3]
    }

    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        f32::sqrt(self.length_sq())
    }

    /// Normalize. Caller's responsibility to ensure non-zero length.
    pub fn normalize(self) -> Vec3A {
        self.scale(1.0f32 / self.length())
    }

    pub fn is_finite(self) -> bool {
        is_finite_f32(self.x()) && is_finite_f32(self.y()) && is_finite_f32(self.z())
    }
}

/// Unit quaternion stored as `(w, x, y, z)` in a 4-lane `cubecl` vector
/// so the emit becomes a native `vec4<f32>`. `Vector<f32, Const<3>>`
/// would be the natural choice for `Vec3A` too, but cubecl-cpp emits
/// `alignas(12)` for the 3-lane case, which Metal rejects (alignas
/// requires a power of 2). 4-lane is fine — `alignas(16)`.
#[derive(CubeType, CubeTypeMut, Copy, Clone)]
pub struct Quat {
    inner: Vector<f32, Const<4>>,
}

#[cube]
#[allow(clippy::should_implement_trait)]
impl Quat {
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Quat {
        let mut v = Vector::<f32, Const<4>>::empty();
        v[0] = w;
        v[1] = x;
        v[2] = y;
        v[3] = z;
        Quat { inner: v }
    }

    pub fn w(self) -> f32 {
        self.inner[0]
    }
    pub fn x(self) -> f32 {
        self.inner[1]
    }
    pub fn y(self) -> f32 {
        self.inner[2]
    }
    pub fn z(self) -> f32 {
        self.inner[3]
    }

    pub fn dot(self, other: Quat) -> f32 {
        let p = self.inner * other.inner;
        p[0] + p[1] + p[2] + p[3]
    }

    pub fn scale(self, s: f32) -> Quat {
        Quat {
            inner: self.inner * Vector::new(s),
        }
    }

    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Normalize. Caller's responsibility to ensure non-zero length.
    pub fn normalize(self) -> Quat {
        self.scale(1.0f32 / f32::sqrt(self.length_sq()))
    }

    /// Rotation matrix for this (assumed unit) quaternion. Column-major.
    pub fn to_mat3(self) -> Mat3 {
        let w = self.w();
        let qx = self.x();
        let qy = self.y();
        let qz = self.z();
        let x2 = qx * qx;
        let y2 = qy * qy;
        let z2 = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = w * qx;
        let wy = w * qy;
        let wz = w * qz;
        Mat3 {
            c0_x: 1.0f32 - 2.0f32 * (y2 + z2),
            c0_y: 2.0f32 * (xy + wz),
            c0_z: 2.0f32 * (xz - wy),
            c1_x: 2.0f32 * (xy - wz),
            c1_y: 1.0f32 - 2.0f32 * (x2 + z2),
            c1_z: 2.0f32 * (yz + wx),
            c2_x: 2.0f32 * (xz + wy),
            c2_y: 2.0f32 * (yz - wx),
            c2_z: 1.0f32 - 2.0f32 * (x2 + y2),
        }
    }
}

/// 3x3 matrix, column-major. `c{i}_{x,y,z}` is column i, row x/y/z.
#[derive(CubeType, Copy, Clone)]
pub struct Mat3 {
    pub c0_x: f32,
    pub c0_y: f32,
    pub c0_z: f32,
    pub c1_x: f32,
    pub c1_y: f32,
    pub c1_z: f32,
    pub c2_x: f32,
    pub c2_y: f32,
    pub c2_z: f32,
}

#[cube]
impl Mat3 {
    pub fn from_cols(c0: Vec3A, c1: Vec3A, c2: Vec3A) -> Mat3 {
        Mat3 {
            c0_x: c0.x(),
            c0_y: c0.y(),
            c0_z: c0.z(),
            c1_x: c1.x(),
            c1_y: c1.y(),
            c1_z: c1.z(),
            c2_x: c2.x(),
            c2_y: c2.y(),
            c2_z: c2.z(),
        }
    }

    pub fn col0(self) -> Vec3A {
        Vec3A::new(self.c0_x, self.c0_y, self.c0_z)
    }

    pub fn col1(self) -> Vec3A {
        Vec3A::new(self.c1_x, self.c1_y, self.c1_z)
    }

    pub fn col2(self) -> Vec3A {
        Vec3A::new(self.c2_x, self.c2_y, self.c2_z)
    }

    /// `M * v`.
    pub fn mul_vec3(self, v: Vec3A) -> Vec3A {
        self.col0()
            .scale(v.x())
            .add(self.col1().scale(v.y()))
            .add(self.col2().scale(v.z()))
    }

    /// `M^T * v`. Equivalent to taking the dot of each column with `v`.
    pub fn transpose_mul_vec3(self, v: Vec3A) -> Vec3A {
        Vec3A::new(self.col0().dot(v), self.col1().dot(v), self.col2().dot(v))
    }

    /// `M * N`. Each output column is `M * N.col_i`.
    pub fn mul_mat3(self, n: Mat3) -> Mat3 {
        Mat3::from_cols(
            self.mul_vec3(n.col0()),
            self.mul_vec3(n.col1()),
            self.mul_vec3(n.col2()),
        )
    }

    /// Right-multiply by `diag(s)` — column-wise scale.
    pub fn mul_diag(self, s: Vec3A) -> Mat3 {
        Mat3::from_cols(
            self.col0().scale(s.x()),
            self.col1().scale(s.y()),
            self.col2().scale(s.z()),
        )
    }
}

/// Symmetric 2x2 matrix. Three independent entries: `c00`, `c01`, `c11`.
#[derive(CubeType, Copy, Clone)]
pub struct Sym2 {
    pub c00: f32,
    pub c01: f32,
    pub c11: f32,
}

/// 2D bbox in tile coords (inclusive min, exclusive max).
#[derive(CubeType, Copy, Clone)]
pub struct TileBbox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

/// 2D pixel bbox as a rect (min/max corners in pixel coords).
#[derive(CubeType, Copy, Clone)]
pub struct PixelRect {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

#[cube]
pub fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + f32::exp(-x))
}

/// Bit-level finite check. NaN / ±Inf have an all-ones exponent.
#[cube]
pub fn is_finite_f32(x: f32) -> bool {
    let bits = u32::reinterpret(x);
    ((bits >> 23u32) & 0xFFu32) != 0xFFu32
}

#[cube]
pub fn is_finite_sym2(c: Sym2) -> bool {
    is_finite_f32(c.c00) && is_finite_f32(c.c11) && is_finite_f32(c.c01)
}

/// `sigma = 0.5 * (cx*dx² + cz*dy²) + cy*dx*dy` for `(dx, dy) = pix - xy`.
#[cube]
pub fn calc_sigma(px: f32, py: f32, conic: Sym2, xy_x: f32, xy_y: f32) -> f32 {
    let dx = px - xy_x;
    let dy = py - xy_y;
    0.5f32 * (conic.c00 * dx * dx + conic.c11 * dy * dy) + conic.c01 * dx * dy
}

/// 2x2 inverse of a symmetric matrix, returning the inverse as a `Sym2`.
/// Returns the zero matrix when `det <= 0` — matches the WGSL non-PD guard.
#[cube]
pub fn inverse_sym2(c: Sym2) -> Sym2 {
    let det = c.c00 * c.c11 - c.c01 * c.c01;
    let invertible = det > 0.0f32;
    let inv_det = select(invertible, 1.0f32 / det, 0.0f32);
    Sym2 {
        c00: c.c11 * inv_det,
        c01: -c.c01 * inv_det,
        c11: c.c00 * inv_det,
    }
}

/// 2x2 strict determinant — `ad` and `bc` computed separately so the
/// compiler can't FMA-fuse them into a single rounding step.
#[cube]
pub fn det2_strict(c: Sym2) -> f32 {
    let ad = c.c00 * c.c11;
    let bc = c.c01 * c.c01;
    ad - bc
}
