//! Minimal triangle-mesh PLY writer (binary little-endian). The `brush-serde`
//! crate's PLY code is splat-specific (positions + per-Gaussian attributes);
//! mesh output gets its own short writer since the schema is just
//! `vertex { x y z }` + `face { vertex_indices [3] }`.

use std::io::{BufRead, Write};

use crate::Mesh;

/// Read a binary little-endian PLY produced by [`write_ply`] back into a
/// [`Mesh`]. Parses the `vertex { x y z [red green blue] }` +
/// `face { list uchar uint vertex_indices }` schema this module writes;
/// `vertex_scales` is left empty (not persisted by the writer). Faces with
/// a vertex count other than 3 are truncated/zero-padded to a triangle.
pub fn read_ply<R: BufRead>(reader: &mut R) -> std::io::Result<Mesh> {
    let mut n_verts = 0usize;
    let mut n_faces = 0usize;
    let mut has_colors = false;
    let mut in_vertex = false;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "PLY: header ended before end_header",
            ));
        }
        let l = line.trim_end();
        if l == "end_header" {
            break;
        }
        let mut it = l.split_whitespace();
        match it.next() {
            Some("element") => match it.next() {
                Some("vertex") => {
                    n_verts = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    in_vertex = true;
                }
                Some("face") => {
                    n_faces = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                    in_vertex = false;
                }
                _ => in_vertex = false,
            },
            Some("property") if in_vertex => {
                // property uchar red — colour block present.
                if it.last() == Some("red") {
                    has_colors = true;
                }
            }
            _ => {}
        }
    }

    let mut vertices = Vec::with_capacity(n_verts);
    let mut vertex_colors = Vec::with_capacity(if has_colors { n_verts } else { 0 });
    let mut xyz = [0u8; 12];
    let mut rgb = [0u8; 3];
    for _ in 0..n_verts {
        reader.read_exact(&mut xyz)?;
        vertices.push(glam::Vec3::new(
            f32::from_le_bytes([xyz[0], xyz[1], xyz[2], xyz[3]]),
            f32::from_le_bytes([xyz[4], xyz[5], xyz[6], xyz[7]]),
            f32::from_le_bytes([xyz[8], xyz[9], xyz[10], xyz[11]]),
        ));
        if has_colors {
            reader.read_exact(&mut rgb)?;
            vertex_colors.push(rgb);
        }
    }

    let mut faces = Vec::with_capacity(n_faces);
    let mut cnt = [0u8; 1];
    let mut idx = [0u8; 4];
    for _ in 0..n_faces {
        reader.read_exact(&mut cnt)?;
        let mut tri = [0u32; 3];
        for j in 0..cnt[0] as usize {
            reader.read_exact(&mut idx)?;
            if j < 3 {
                tri[j] = u32::from_le_bytes(idx);
            }
        }
        faces.push(tri);
    }

    Ok(Mesh {
        vertices,
        vertex_scales: Vec::new(),
        vertex_colors,
        faces,
    })
}

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
