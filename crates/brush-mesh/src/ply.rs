//! Minimal triangle-mesh PLY writer (binary little-endian). The `brush-serde`
//! crate's PLY code is splat-specific (positions + per-Gaussian attributes);
//! mesh output gets its own short writer since the schema is just
//! `vertex { x y z }` + `face { vertex_indices [3] }`.

use std::io::Write;

use crate::Mesh;

/// Write `mesh` to `out` as a binary little-endian PLY. Per-vertex u8
/// `red/green/blue` properties are emitted when `mesh.vertex_colors` is
/// populated (and has the same length as `vertices`).
pub fn write_ply<W: Write>(out: &mut W, mesh: &Mesh) -> std::io::Result<()> {
    let n_verts = mesh.vertices.len();
    let n_faces = mesh.faces.len();
    let has_colors = mesh.vertex_colors.len() == n_verts && n_verts > 0;
    let color_props = if has_colors {
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    } else {
        ""
    };
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\nelement vertex {n_verts}\nproperty float x\nproperty float y\nproperty float z\n{color_props}element face {n_faces}\nproperty list uchar uint vertex_indices\nend_header\n"
    )?;
    for (i, v) in mesh.vertices.iter().enumerate() {
        out.write_all(&v.x.to_le_bytes())?;
        out.write_all(&v.y.to_le_bytes())?;
        out.write_all(&v.z.to_le_bytes())?;
        if has_colors {
            let c = mesh.vertex_colors[i];
            out.write_all(&c)?;
        }
    }
    for f in &mesh.faces {
        out.write_all(&[3u8])?;
        for &i in f {
            out.write_all(&i.to_le_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_header_and_payload() {
        let mesh = Mesh {
            vertices: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
            ],
            vertex_scales: vec![1.0, 1.0, 1.0],
            vertex_colors: Vec::new(),
            faces: vec![[0, 1, 2]],
        };
        let mut buf = Vec::new();
        write_ply(&mut buf, &mesh).unwrap();
        let needle = b"end_header\n";
        let header_end =
            buf.windows(needle.len()).position(|w| w == needle).unwrap() + needle.len();
        let header = std::str::from_utf8(&buf[..header_end]).unwrap();
        assert!(header.contains("element vertex 3"));
        assert!(header.contains("element face 1"));
        // 3 verts * 12 bytes + 1 face * (1 + 12) = 49 bytes payload.
        assert_eq!(buf.len() - header_end, 3 * 12 + 13);
    }
}
