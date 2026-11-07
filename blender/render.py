"""Render frames (or the four strip photos) from the built .blend.

    python blender/render.py -- --blend scene.blend --out frames/ --frames 0-434
    python blender/render.py -- --blend scene.blend --out cells/ --cells
    python blender/render.py -- --blend scene.blend --out test/ --frames 5,40,150 --percent 50 --samples 16

Frame numbers are the project's 0-based indices (scene frame = index + 1).
"""
import argparse
import os
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anim import FLASHES  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="")
ap.add_argument("--cells", action="store_true")
ap.add_argument("--percent", type=int, default=100)
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--skip-existing", action="store_true")
args = ap.parse_args(argv)


def parse_frames(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
scene = bpy.context.scene
scene.render.resolution_percentage = args.percent
if args.samples:
    scene.cycles.samples = args.samples
scene.render.threads_mode = "AUTO"
os.makedirs(args.out, exist_ok=True)

if args.cells:
    # the four photos that end up on the printed strip: booth camera, landscape 3:2, flash at peak
    scene.render.resolution_x = 900
    scene.render.resolution_y = 600
    scene.camera = bpy.data.objects["cam_cell_strip"]
    for m in list(scene.timeline_markers):
        m.camera = None
    for i, f in enumerate(FLASHES):
        scene.frame_set(f + 1)
        scene.camera = bpy.data.objects["cam_cell_strip"]
        scene.render.filepath = os.path.join(args.out, f"cell_{i + 1}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"cell {i + 1} (frame {f}) {time.time() - t:.1f}s", flush=True)
    sys.exit(0)

frames = parse_frames(args.frames)
for f in frames:
    path = os.path.join(args.out, f"frame_{f:04d}.png")
    if args.skip_existing and os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    scene.frame_set(f + 1)
    scene.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"frame {f} {time.time() - t:.1f}s", flush=True)
