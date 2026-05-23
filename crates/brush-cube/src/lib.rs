//! Shared brush primitives. Host-side tensor / launch helpers (re-exported
//! at the crate root) plus cube-side math types (`Vec3A`, `Quat`, `Mat3`,
//! `Sym2`), tile/pixel rect aggregates, and pure helpers (`sigmoid`,
//! `is_finite_*`, `calc_sigma`, `inverse_sym2`, `det2_strict`).
//!
//! Methods like `Vec3A::add` / `Quat::scale` are deliberately inherent
//! rather than `Add`/`Mul` impls — `#[cube]` traces method calls into
//! the IR, while operator overloading bypasses it.
//!
//! `Vec3A` and `Quat` both wrap `Vector<f32, Const<4>>` for native
//! `vec4<f32>` codegen. `Const<3>` would be more natural for `Vec3A`,
//! but cubecl-cpp's Metal dialect emits `alignas(elem_size * lanes)`
//! literally — `alignas(12)` is invalid C++ (alignas requires a power
//! of 2) and Metal rejects the shader. 4-lane gets `alignas(16)`,
//! which is fine. `Vec3A` pins lane 3 to zero so `dot`/`length`/etc.
//! see only the three real components.

#![allow(clippy::should_implement_trait)]

mod host;
pub mod test_helpers;
pub use host::*;

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

/// 3-component f32 vector, padded to 4 lanes — same shape as
/// `glam::Vec3A`. See the module-level note on the cubecl-cpp
/// alignas-12 workaround.
#[derive(CubeType, CubeTypeMut, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct Vec3A {
    inner: Vector<f32, Const<4>>,
}

#[cube]
impl Vec3A {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3A {
        let mut v = Vector::<f32, Const<4>>::empty();
        v.insert(0, x);
        v.insert(1, y);
        v.insert(2, z);
        // Padding lane — must stay 0 so `dot` / `length` see only the
        // three real components.
        v.insert(3, 0.0f32);
        Vec3A { inner: v }
    }

    pub fn x(self) -> f32 {
        self.inner.extract(0)
    }
    pub fn y(self) -> f32 {
        self.inner.extract(1)
    }
    pub fn z(self) -> f32 {
        self.inner.extract(2)
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
        // Lane 3 is always 0 in both operands → no special case.
        p.extract(0) + p.extract(1) + p.extract(2) + p.extract(3)
    }

    pub fn length(self) -> f32 {
        f32::sqrt(self.dot(self))
    }

    /// Normalize. Caller's responsibility to ensure non-zero length.
    pub fn normalize(self) -> Vec3A {
        self.scale(1.0f32 / self.length())
    }

    pub fn is_finite(self) -> bool {
        is_finite_f32(self.x()) && is_finite_f32(self.y()) && is_finite_f32(self.z())
    }
}

#[derive(CubeType, CubeTypeMut, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct Vec2 {
    inner: Vector<f32, Const<2>>,
}

#[cube]
impl Vec2 {
    pub fn new(x: f32, y: f32) -> Vec2 {
        let mut v = Vector::<f32, Const<2>>::empty();
        v.insert(0, x);
        v.insert(1, y);
        Vec2 { inner: v }
    }

    pub fn x(self) -> f32 {
        self.inner.extract(0)
    }
    pub fn y(self) -> f32 {
        self.inner.extract(1)
    }

    pub fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            inner: self.inner + other.inner,
        }
    }

    pub fn scale(self, s: f32) -> Vec2 {
        Vec2 {
            inner: self.inner * Vector::new(s),
        }
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x() * other.x() + self.y() * other.y()
    }
}

/// Unit quaternion stored as `(w, x, y, z)` in a 4-lane cubecl vector.
#[derive(CubeType, CubeTypeMut, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct Quat {
    inner: Vector<f32, Const<4>>,
}

#[cube]
impl Quat {
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Quat {
        let mut v = Vector::<f32, Const<4>>::empty();
        v.insert(0, w);
        v.insert(1, x);
        v.insert(2, y);
        v.insert(3, z);
        Quat { inner: v }
    }

    pub fn w(self) -> f32 {
        self.inner.extract(0)
    }
    pub fn x(self) -> f32 {
        self.inner.extract(1)
    }
    pub fn y(self) -> f32 {
        self.inner.extract(2)
    }
    pub fn z(self) -> f32 {
        self.inner.extract(3)
    }

    pub fn dot(self, other: Quat) -> f32 {
        let p = self.inner * other.inner;
        p.extract(0) + p.extract(1) + p.extract(2) + p.extract(3)
    }

    pub fn scale(self, s: f32) -> Quat {
        Quat {
            inner: self.inner * Vector::new(s),
        }
    }

    /// Normalize. Caller's responsibility to ensure non-zero length.
    pub fn normalize(self) -> Quat {
        self.scale(1.0f32 / f32::sqrt(self.dot(self)))
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
#[expand(derive(Clone, Copy))]
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

    pub fn row0(self) -> Vec3A {
        Vec3A::new(self.c0_x, self.c1_x, self.c2_x)
    }

    pub fn row1(self) -> Vec3A {
        Vec3A::new(self.c0_y, self.c1_y, self.c2_y)
    }

    pub fn row2(self) -> Vec3A {
        Vec3A::new(self.c0_z, self.c1_z, self.c2_z)
    }

    /// `M * M^T` — the result is always symmetric.
    pub fn outer_product_self(self) -> Sym3 {
        let r0 = self.row0();
        let r1 = self.row1();
        let r2 = self.row2();
        Sym3 {
            c00: r0.dot(r0),
            c01: r0.dot(r1),
            c02: r0.dot(r2),
            c11: r1.dot(r1),
            c12: r1.dot(r2),
            c22: r2.dot(r2),
        }
    }
}

/// 2x3 matrix, column-major.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct Mat2x3 {
    pub c0: Vec2,
    pub c1: Vec2,
    pub c2: Vec2,
}

#[cube]
impl Mat2x3 {
    /// `M * N`. Each output column is `M * N.col_i`.
    pub fn mul_mat3(self, n: Mat3) -> Mat2x3 {
        Mat2x3 {
            c0: self.mul_vec3(n.col0()),
            c1: self.mul_vec3(n.col1()),
            c2: self.mul_vec3(n.col2()),
        }
    }

    /// `M * v`.
    pub fn mul_vec3(self, v: Vec3A) -> Vec2 {
        self.c0
            .scale(v.x())
            .add(self.c1.scale(v.y()))
            .add(self.c2.scale(v.z()))
    }

    pub fn row0(self) -> Vec3A {
        Vec3A::new(self.c0.x(), self.c1.x(), self.c2.x())
    }

    pub fn row1(self) -> Vec3A {
        Vec3A::new(self.c0.y(), self.c1.y(), self.c2.y())
    }

    /// `self^T * sym * self` — congruence (2×3)^T × (2×2 sym) × (2×3) → (3×3 sym).
    pub fn transpose_congruence_sym2(self, sym: Sym2) -> Sym3 {
        let sc0 = sym.mul_vec2(self.c0);
        let sc1 = sym.mul_vec2(self.c1);
        let sc2 = sym.mul_vec2(self.c2);
        Sym3 {
            c00: self.c0.dot(sc0),
            c01: self.c0.dot(sc1),
            c02: self.c0.dot(sc2),
            c11: self.c1.dot(sc1),
            c12: self.c1.dot(sc2),
            c22: self.c2.dot(sc2),
        }
    }

    /// `M^T * v`.
    pub fn transpose_mul_vec2(self, v: Vec2) -> Vec3A {
        self.row0().scale(v.x()).add(self.row1().scale(v.y()))
    }

    pub fn gram_matrix(self) -> Sym2 {
        let c00 = self.c0.x() * self.c0.x() + self.c1.x() * self.c1.x() + self.c2.x() * self.c2.x();
        let c01 = self.c0.x() * self.c0.y() + self.c1.x() * self.c1.y() + self.c2.x() * self.c2.y();
        let c11 = self.c0.y() * self.c0.y() + self.c1.y() * self.c1.y() + self.c2.y() * self.c2.y();

        Sym2 { c00, c01, c11 }
    }
}

/// Symmetric 2x2 matrix. Three independent entries: `c00`, `c01`, `c11`.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct Sym2 {
    pub c00: f32,
    pub c01: f32,
    pub c11: f32,
}

#[cube]
impl Sym2 {
    pub fn col0(self) -> Vec2 {
        Vec2::new(self.c00, self.c01)
    }

    pub fn col1(self) -> Vec2 {
        Vec2::new(self.c01, self.c11)
    }

    /// `M * v`.
    pub fn mul_vec2(self, v: Vec2) -> Vec2 {
        self.col0().scale(v.x()).add(self.col1().scale(v.y()))
    }

    pub fn scale(self, s: f32) -> Sym2 {
        Sym2 {
            c00: self.c00 * s,
            c01: self.c01 * s,
            c11: self.c11 * s,
        }
    }

    pub fn max_abs(&self) -> f32 {
        max(
            max(f32::abs(self.c00), f32::abs(self.c11)),
            f32::abs(self.c01),
        )
    }

    /// `M * N`. Each output column is `M * N.col_i`.
    pub fn mul_mat2x3(self, n: Mat2x3) -> Mat2x3 {
        Mat2x3 {
            c0: self.mul_vec2(n.c0),
            c1: self.mul_vec2(n.c1),
            c2: self.mul_vec2(n.c2),
        }
    }

    /// 2x2 inverse of a symmetric matrix, returning the inverse as a `Sym2`.
    /// Returns the zero matrix when `det <= 0` (non-PD guard).
    pub fn inverse(self) -> Sym2 {
        let det = self.c00 * self.c11 - self.c01 * self.c01;
        let invertible = det > 0.0f32;
        let inv_det = select(invertible, 1.0f32 / det, 0.0f32);
        Sym2 {
            c00: self.c11 * inv_det,
            c01: -self.c01 * inv_det,
            c11: self.c00 * inv_det,
        }
    }

    /// 2x2 strict determinant — `ad` and `bc` computed separately so the
    /// compiler can't FMA-fuse them into a single rounding step.
    pub fn det2_strict(self) -> f32 {
        let ad = self.c00 * self.c11;
        let bc = self.c01 * self.c01;
        ad - bc
    }

    pub fn is_finite(self) -> bool {
        is_finite_f32(self.c00) && is_finite_f32(self.c11) && is_finite_f32(self.c01)
    }
}

/// Symmetric 3×3 matrix. Six independent entries: `c{i}{j}` with `i ≤ j`.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct Sym3 {
    pub c00: f32,
    pub c01: f32,
    pub c02: f32,
    pub c11: f32,
    pub c12: f32,
    pub c22: f32,
}

#[cube]
impl Sym3 {
    pub fn row0(self) -> Vec3A {
        Vec3A::new(self.c00, self.c01, self.c02)
    }

    pub fn row1(self) -> Vec3A {
        Vec3A::new(self.c01, self.c11, self.c12)
    }

    pub fn row2(self) -> Vec3A {
        Vec3A::new(self.c02, self.c12, self.c22)
    }

    /// `self * v`.
    pub fn mul_vec3(self, v: Vec3A) -> Vec3A {
        self.row0()
            .scale(v.x())
            .add(self.row1().scale(v.y()))
            .add(self.row2().scale(v.z()))
    }

    pub fn scale(self, s: f32) -> Sym3 {
        Sym3 {
            c00: self.c00 * s,
            c01: self.c01 * s,
            c02: self.c02 * s,
            c11: self.c11 * s,
            c12: self.c12 * s,
            c22: self.c22 * s,
        }
    }

    /// `self * m`, treating self as a full symmetric 3×3 matrix.
    pub fn mul_mat3(self, m: Mat3) -> Mat3 {
        Mat3::from_cols(
            self.mul_vec3(m.col0()),
            self.mul_vec3(m.col1()),
            self.mul_vec3(m.col2()),
        )
    }

    /// `m * self * m^T` — congruence transform. Result is symmetric.
    pub fn congruence(self, m: Mat3) -> Sym3 {
        let sr0 = self.mul_vec3(m.row0());
        let sr1 = self.mul_vec3(m.row1());
        let sr2 = self.mul_vec3(m.row2());
        Sym3 {
            c00: m.row0().dot(sr0),
            c01: m.row0().dot(sr1),
            c02: m.row0().dot(sr2),
            c11: m.row1().dot(sr1),
            c12: m.row1().dot(sr2),
            c22: m.row2().dot(sr2),
        }
    }

    /// `m^T * self * m` — transpose congruence. Result is symmetric.
    pub fn transpose_congruence(self, m: Mat3) -> Sym3 {
        let sc0 = self.mul_vec3(m.col0());
        let sc1 = self.mul_vec3(m.col1());
        let sc2 = self.mul_vec3(m.col2());
        Sym3 {
            c00: m.col0().dot(sc0),
            c01: m.col0().dot(sc1),
            c02: m.col0().dot(sc2),
            c11: m.col1().dot(sc1),
            c12: m.col1().dot(sc2),
            c22: m.col2().dot(sc2),
        }
    }
}

/// 2D bbox in tile coords (inclusive min, exclusive max).
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct TileBbox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

/// 2D pixel bbox as a rect (min/max corners in pixel coords).
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
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

/// `sigma = 0.5 * (cx*dx² + cz*dy²) + cy*dx*dy` for `(dx, dy) = pix - xy`.
#[cube]
pub fn calc_sigma(px: f32, py: f32, conic: Sym2, xy_x: f32, xy_y: f32) -> f32 {
    let dx = px - xy_x;
    let dy = py - xy_y;
    0.5f32 * (conic.c00 * dx * dx + conic.c11 * dy * dy) + conic.c01 * dx * dy
}
