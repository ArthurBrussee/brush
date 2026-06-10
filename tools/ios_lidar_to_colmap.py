#!/usr/bin/env python3
"""Convert an iPhone/ARKit `MapperInput.pb` (+ a `raw/` frame dir) into a COLMAP
dataset for Brush, with the per-frame LiDAR depth/confidence carried alongside.

Layout produced under <out>:
    images/frame_XXXX.jpg          (symlink to raw/)
    depth/frame_XXXX.bin           (256x144 float32 metres, symlink to raw/)
    depth/frame_XXXX_confidence.bin(256x144 uint8 0/1/2, symlink to raw/)
    sparse/0/{cameras,images,points3D}.txt

ARKit `world_from_kf_prior` is camera->world, already in CV camera convention
(+X right, +Y down, +Z forward — verified by minimizing multi-view AprilTag
reprojection: R_w2c = R_c2wᵀ, no axis flip, gives ~5px median error). So
R_w2c = R_c2wᵀ, t_w2c = -R_w2c·t. World units are millimetres (AprilTag frame);
we divide translations by 1000 so the scene is metric metres, matching the
LiDAR depth.


RF: I Thiiiiink this is unused now?
"""

import math
import os
import subprocess
import sys

MECH = "/Users/arthurbrussee/Code/mech"


def decode(pb_path):
    out = subprocess.run(
        [
            "protoc",
            "--proto_path=proto",
            "--decode=terraform.mapper.MapperInput",
            "proto/terraform/mapper/pipeline.proto",
        ],
        cwd=MECH,
        stdin=open(pb_path, "rb"),
        capture_output=True,
        check=True,
    )
    return out.stdout.decode()


def decode_itm(pb_path):
    """Decode a refined IterativeTagMap (bench_e2e --dump-poses output)."""
    out = subprocess.run(
        [
            "protoc",
            "--proto_path=proto",
            "--decode=terraform.reconstruction.IterativeTagMap",
            "proto/terraform/reconstruction/online.proto",
        ],
        cwd=MECH,
        stdin=open(pb_path, "rb"),
        capture_output=True,
        check=True,
    )
    return out.stdout.decode()


def parse(text):
    """Protobuf text-format -> nested dict (repeated fields become lists)."""
    toks, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "{}:":
            toks.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            toks.append(text[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "{}:":
                j += 1
            toks.append(text[i:j])
            i = j

    pos = 0

    def block():
        nonlocal pos
        d = {}
        while pos < len(toks) and toks[pos] != "}":
            key = toks[pos]
            pos += 1
            if toks[pos] == ":":
                pos += 1
                val = toks[pos]
                pos += 1
                if val.startswith('"'):
                    val = val[1:-1]
                elif val in ("true", "false"):
                    val = val == "true"
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass  # enum / identifier — keep as string

            else:  # '{'
                pos += 1
                val = block()
                assert toks[pos] == "}"
                pos += 1
            d.setdefault(key, [])
            d[key].append(val)
        return d

    return block()


def get(d, *path, default=None):
    cur = d
    for k in path:
        if k not in cur:
            return default
        cur = cur[k][0]
    return cur


def quat_to_mat(x, y, z, w):
    nrm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / nrm, y / nrm, z / nrm, w / nrm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matT(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


def matvec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def mat_to_quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


import struct

DEPTH_MAGIC = b"BRDPTH\x01\x00"  # 8 bytes incl. version


def infer_dims(pixel_count, aspect):
    """width*height == pixel_count with width/height closest to `aspect`."""
    best = None
    w0 = max(1, round(math.sqrt(pixel_count * aspect)))
    for w in range(max(1, w0 - 4), w0 + 5):
        if pixel_count % w == 0:
            h = pixel_count // w
            err = abs((w / h) - aspect)
            if best is None or err < best[0]:
                best = (err, w, h)
    return (best[1], best[2]) if best else (pixel_count, 1)


def write_depth(raw_dir, stem, aspect, out_path):
    """Pack ARKit z-depth (float32 metres) + confidence (u8) into one
    self-describing file: magic, width u32, height u32, depth f32[w*h], conf u8[w*h]."""
    dpath = os.path.join(raw_dir, f"{stem}.bin")
    if not os.path.exists(dpath):
        return False
    depth = open(dpath, "rb").read()
    pc = len(depth) // 4
    w, h = infer_dims(pc, aspect)
    cpath = os.path.join(raw_dir, f"{stem}_confidence.bin")
    conf = open(cpath, "rb").read() if os.path.exists(cpath) else b"\x02" * pc
    if len(conf) < pc:
        conf = conf + b"\x02" * (pc - len(conf))
    conf = conf[:pc]
    with open(out_path, "wb") as f:
        f.write(DEPTH_MAGIC + struct.pack("<II", w, h) + depth + conf)
    return True


def main(pb_path, raw_dir, out_dir):
    root = parse(decode(pb_path))
    kfs = root["keyframes"][0]["keyframes"]

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sparse", "0"), exist_ok=True)

    cam_lines, img_lines = [], []
    n = 0
    for cid, kf in enumerate(kfs, start=1):
        name = get(kf, "filename")
        if name is None:
            continue
        stem = os.path.splitext(name)[0]  # frame_XXXX
        intr = kf["intrinsics"][0]
        fx = get(intr, "fx", default=0.0)
        fy = get(intr, "fy", default=0.0)
        cx = get(intr, "px", default=0.0)
        cy = get(intr, "py", default=0.0)
        w = int(get(kf, "image_width", default=get(intr, "width")))
        h = int(get(kf, "image_height", default=get(intr, "height")))

        rot = kf["world_from_kf_prior"][0]["rotation"][0]
        tr = kf["world_from_kf_prior"][0]["translation"][0]
        r_c2w = quat_to_mat(
            get(rot, "x", default=0.0),
            get(rot, "y", default=0.0),
            get(rot, "z", default=0.0),
            get(rot, "w", default=0.0),
        )
        t_c2w = [
            get(tr, "x", default=0.0) / 1000.0,
            get(tr, "y", default=0.0) / 1000.0,
            get(tr, "z", default=0.0) / 1000.0,
        ]
        r_w2c = matT(r_c2w)
        t_w2c = matvec(r_w2c, t_c2w)
        t_w2c = [-t_w2c[0], -t_w2c[1], -t_w2c[2]]
        qw, qx, qy, qz = mat_to_quat(r_w2c)

        cam_lines.append(f"{cid} PINHOLE {w} {h} {fx} {fy} {cx} {cy}")
        img_lines.append(
            f"{cid} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {cid} {name}"
        )
        img_lines.append("")

        # Symlink the image; pack the LiDAR depth + confidence into one file.
        jpg_dst = os.path.join(out_dir, "images", f"{stem}.jpg")
        jpg_src = os.path.abspath(os.path.join(raw_dir, f"{stem}.jpg"))
        if os.path.exists(jpg_src):
            if os.path.islink(jpg_dst) or os.path.exists(jpg_dst):
                os.remove(jpg_dst)
            os.symlink(jpg_src, jpg_dst)
        write_depth(raw_dir, stem, w / h, os.path.join(out_dir, "depth", f"{stem}.bin"))
        n += 1

    sp = os.path.join(out_dir, "sparse", "0")
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write("# Camera list\n" + "\n".join(cam_lines) + "\n")
    with open(os.path.join(sp, "images.txt"), "w") as f:
        f.write("# Image list\n" + "\n".join(img_lines) + "\n")
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
    print(f"Wrote {n} cameras/images + packed depth (z-metres + conf) to {out_dir}.")


def is_degenerate_distortion(fx, fy, cx, cy, w, h, dist, margin=2.5, max_distort=3.0):
    """True if the OpenCV-rational distortion is unstable within `margin` x the
    corner radius. The tag-BA refine overfits k1..k6 to huge near-cancelling
    values; for some cameras the denominator `1 + k4 r^2 + k5 r^4 + k6 r^6`
    crosses zero (a pole) or gets tiny, so the distortion `num/den` blows up and
    flings splats to infinity. Scanned past the frame edge because splats near
    the FOV boundary project to radii beyond the corner. Rejected when the
    denominator changes sign, or |num/den| exceeds `max_distort` anywhere.
    `dist` is [k1, k2, p1, p2, k3, k4, k5, k6]."""
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        return True
    k1, k2, _p1, _p2, k3, k4, k5, k6 = dist
    rmax = max(math.hypot((u - cx) / fx, (v - cy) / fy) for u in (0, w) for v in (0, h))
    rmax *= margin
    prev = 1.0
    steps = 600
    for i in range(1, steps + 1):
        r2 = (rmax * i / steps) ** 2
        r4 = r2 * r2
        r6 = r4 * r2
        den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        if prev * den < 0.0 or abs(den) < 1e-9:
            return True
        prev = den
        num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        if abs(num / den) > max_distort:
            return True
    return False


def main_refined(itm_path, raw_dir, out_dir):
    """Convert from the mapper's tag-BA refined `IterativeTagMap` (camera_states
    keyed by image name -> world_from_cam Mat4 + refined OpenCV-rational
    intrinsics). Cameras are written as COLMAP FULL_OPENCV so Brush applies the
    BA distortion (RadialTangential8); translations mm -> m."""
    root = parse(decode_itm(itm_path))
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sparse", "0"), exist_ok=True)
    # Clear stale frames so dropped cameras don't leave orphaned files.
    for sub in ("images", "depth"):
        d = os.path.join(out_dir, sub)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    cam_lines, img_lines = [], []
    n, dropped = 0, 0
    cid = 0
    for cs in root.get("camera_states", []):
        key = get(cs, "key")
        if key is None:
            continue
        stem = os.path.splitext(os.path.basename(key))[0]
        val = cs["value"][0]
        intr = val["intrinsics"][0]
        fx, fy = get(intr, "fx", default=0.0), get(intr, "fy", default=0.0)
        cx, cy = get(intr, "px", default=0.0), get(intr, "py", default=0.0)
        w, h = int(get(intr, "width")), int(get(intr, "height"))
        dist = [float(d) for d in intr.get("distortion", [])]
        dist = (dist + [0.0] * 8)[:8]  # k1 k2 p1 p2 k3 k4 k5 k6

        if is_degenerate_distortion(fx, fy, cx, cy, w, h, dist):
            dropped += 1
            continue
        cid += 1

        mat = val["pose"][0]["mat"]  # 16 row-major, world_from_cam
        r_c2w = [
            [mat[0], mat[1], mat[2]],
            [mat[4], mat[5], mat[6]],
            [mat[8], mat[9], mat[10]],
        ]
        c = [mat[3] / 1000.0, mat[7] / 1000.0, mat[11] / 1000.0]
        r_w2c = matT(r_c2w)
        t = matvec(r_w2c, c)
        t = [-t[0], -t[1], -t[2]]
        qw, qx, qy, qz = mat_to_quat(r_w2c)

        cam_lines.append(
            f"{cid} FULL_OPENCV {w} {h} {fx} {fy} {cx} {cy} "
            + " ".join(str(d) for d in dist)
        )
        img_lines.append(
            f"{cid} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cid} {stem}.jpg"
        )
        img_lines.append("")

        jpg_dst = os.path.join(out_dir, "images", f"{stem}.jpg")
        jpg_src = os.path.abspath(os.path.join(raw_dir, f"{stem}.jpg"))
        if os.path.exists(jpg_src):
            if os.path.islink(jpg_dst) or os.path.exists(jpg_dst):
                os.remove(jpg_dst)
            os.symlink(jpg_src, jpg_dst)
        write_depth(raw_dir, stem, w / h, os.path.join(out_dir, "depth", f"{stem}.bin"))
        n += 1

    sp = os.path.join(out_dir, "sparse", "0")
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write("# Camera list\n" + "\n".join(cam_lines) + "\n")
    with open(os.path.join(sp, "images.txt"), "w") as f:
        f.write("# Image list\n" + "\n".join(img_lines) + "\n")
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
    print(
        f"Wrote {n} cameras/images (FULL_OPENCV, tag-BA poses) + depth to {out_dir}. "
        f"Dropped {dropped} cameras with degenerate distortion (pole / extreme distortion)."
    )


if __name__ == "__main__":
    base = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else "~/data/rollable-board-ios"
    )
    raw = os.path.join(base, "raw")
    default_out = (base[:-4] if base.endswith("-ios") else base) + "-colmap"
    out = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else default_out
    refined = os.path.join(base, "refined_poses.pb")
    if os.path.exists(refined):
        main_refined(refined, raw, out)
    else:
        main(os.path.join(base, "source.MapperInput.pb"), raw, out)
