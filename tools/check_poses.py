#!/usr/bin/env python3
"""Find the correct ARKit->COLMAP pose convention by minimizing multi-view tag
reprojection error. Brute-forces (transpose?, flip placement, quat conj?) and
reports the reprojection error for each so we can pick the consistent one."""
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ios_lidar_to_colmap import decode, parse, get  # noqa: E402


def qm(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def T(m): return [[m[j][i] for j in range(3)] for i in range(3)]
def mm(a, b): return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def mv(m, v): return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]
def dot(a, b): return sum(a[i] * b[i] for i in range(3))


FLIP = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]


def build(R, C, variant):
    """Return (R_wc, t_wc) for a pose-construction variant. C = camera centre."""
    transpose, flip, conj = variant
    Rb = T(R) if conj else R          # treat quat as inverse rotation?
    Rc = T(Rb) if transpose else Rb
    Rwc = mm(FLIP, Rc) if flip else Rc
    twc = mv(Rwc, C)
    return Rwc, [-twc[0], -twc[1], -twc[2]]


def center(Rwc, twc): return mv(T(Rwc), [-twc[0], -twc[1], -twc[2]])
def ray(K, Rwc, uv):
    fx, fy, cx, cy = K
    return mv(T(Rwc), [(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0])


def tri(c0, r0, c1, r1):
    d0, d1, d01 = dot(r0, r0), dot(r1, r1), dot(r0, r1)
    w0 = [c0[i] - c1[i] for i in range(3)]
    a, b = dot(r0, w0), dot(r1, w0)
    den = d0 * d1 - d01 * d01 or 1e-9
    s, t = (d01 * b - d1 * a) / den, (d0 * b - d01 * a) / den
    p0 = [c0[i] + s * r0[i] for i in range(3)]
    p1 = [c1[i] + t * r1[i] for i in range(3)]
    return [(p0[i] + p1[i]) / 2 for i in range(3)]


def proj(K, Rwc, twc, P):
    fx, fy, cx, cy = K
    pc = [sum(Rwc[i][k] * P[k] for k in range(3)) + twc[i] for i in range(3)]
    if pc[2] <= 1e-6:
        return None
    return (fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy)


def main(base):
    root = parse(decode(os.path.join(base, "source.MapperInput.pb")))
    raw = []
    for kf in root["keyframes"][0]["keyframes"]:
        intr = kf["intrinsics"][0]
        K = (get(intr, "fx", default=0.0), get(intr, "fy", default=0.0),
             get(intr, "px", default=0.0), get(intr, "py", default=0.0))
        rot = kf["world_from_kf_prior"][0]["rotation"][0]
        tr = kf["world_from_kf_prior"][0]["translation"][0]
        R = qm(get(rot, "x", default=0.0), get(rot, "y", default=0.0),
               get(rot, "z", default=0.0), get(rot, "w", default=0.0))
        C = [get(tr, "x", default=0.0) / 1000, get(tr, "y", default=0.0) / 1000,
             get(tr, "z", default=0.0) / 1000]
        dets = {}
        for d in kf.get("detections", []):
            tid = get(d, "tag", "id")
            c = d.get("corners", [{}])[0].get("top_left", [{}])[0]
            if tid is not None and "x" in c:
                dets[int(tid)] = (c["x"][0], c["y"][0])
        raw.append((K, R, C, dets))
    print("frames", len(raw))

    from collections import defaultdict
    for variant in [(tr, fl, cj) for tr in (False, True) for fl in (False, True) for cj in (False, True)]:
        frames = [(K, *build(R, C, variant), dets) for (K, R, C, dets) in raw]
        seen = defaultdict(list)
        for fi, (K, Rwc, twc, dets) in enumerate(frames):
            for tid in dets:
                seen[tid].append(fi)
        errs = []
        for tid, fl_ in seen.items():
            if len(fl_) < 3:
                continue
            for k in range(0, len(fl_) - 2, max(1, len(fl_) // 5)):
                a, bb, c = fl_[k], fl_[k + 1], fl_[k + 2]
                Ka, Ra, ta, da = frames[a]
                Kb, Rb, tb, db = frames[bb]
                Kc, Rc, tc, dc = frames[c]
                P = tri(center(Ra, ta), ray(Ka, Ra, da[tid]),
                        center(Rb, tb), ray(Kb, Rb, db[tid]))
                pp = proj(Kc, Rc, tc, P)
                if pp:
                    errs.append(math.hypot(pp[0] - dc[tid][0], pp[1] - dc[tid][1]))
        errs.sort()
        med = errs[len(errs) // 2] if errs else float("nan")
        print(f"variant transpose={variant[0]} flip={variant[1]} conj={variant[2]}: "
              f"median reproj err {med:8.1f} px  (n={len(errs)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/rollable-board-ios"))
